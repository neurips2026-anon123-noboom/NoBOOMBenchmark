from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class _FeaturePolicy:
    name: str
    kind: str  # "almost_constant" | "extreme_vals" | "robust" | "zero_inflated"
    lower_cutoff: Optional[float] = None
    upper_cutoff: Optional[float] = None

    # Optional pre-transform
    apply_asinh: bool = False
    asinh_scale: Optional[float] = None  # s in asinh(x/s)

    # Zero-inflated expansion
    add_indicator: bool = False  # only used if kind == "zero_inflated"


@dataclass(frozen=True)
class _TransformGroupPlan:
    input_indices: np.ndarray
    output_indices: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    asinh_local_indices: np.ndarray
    asinh_scales: np.ndarray
    indicator_local_indices: Optional[np.ndarray] = None
    indicator_output_indices: Optional[np.ndarray] = None
    lower_cutoffs: Optional[np.ndarray] = None
    upper_cutoffs: Optional[np.ndarray] = None


class MixedRobustPreprocessor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for Pandas DataFrames.

    Feature kinds:
      - almost_constant: center-only (mean), scale=1
      - zero_inflated: (optional) output indicator + magnitude stream
      - extreme_vals: feature flagged by tail heuristic; apply extreme_action to those points
      - robust: everything else

    Scaling mode:
      - use_robust_scaling=True  -> RobustScaler-like (median + quantile_range scale)
      - use_robust_scaling=False -> StandardScaler-like (mean + std)

    Robust-specific options (used only if use_robust_scaling=True):
      - quantile_range: robust scale uses Q(q_max)-Q(q_min), default (25,75) = IQR
      - unit_variance: matches sklearn RobustScaler(unit_variance=True)
          scale = (Q(q_max)-Q(q_min)) / (norm.ppf(q_max/100) - norm.ppf(q_min/100))

    Extreme values handling (NEW; replaces mask_extreme_vals):
      - extreme_action="nan"  : replace extremes with NaN (default)
      - extreme_action="ffill": forward-fill then back-fill (no NaNs left)
      - extreme_action="none" : do nothing to extreme points

    Optional additions:
      (5) Zero-inflated split:
          For features with high zero mass: output indicator + magnitude stream.
      (6) Heavy-tail asinh compression:
          Apply asinh(x/s) before scaling, optionally auto-selected per feature.
      (9) Post-scaling clipping:
          Clip final output to [-clip_value, clip_value] for reconstruction stability.

    Output:
      - np.ndarray float32 with stable column order
      - get_feature_names_out() reflects expanded columns (__is_nonzero, __mag)
    """

    def __init__(
            self,
            *,
            columns: Optional[Sequence[str]] = None,
            drop_features: Optional[Sequence[str]] = None,

            # Scaling mode toggle
            use_robust_scaling: bool = True,

            # RobustScaler-like (only used if use_robust_scaling=True)
            quantile_range: Tuple[float, float] = (25.0, 75.0),
            unit_variance: bool = False,

            # Extreme tail detection (your original logic)
            extreme_side: str = "both",  # "upper" | "lower" | "both"
            q_ref: float = 0.995,
            q_top: float = 0.999,
            max_tail_mass: float = 0.01,
            min_jump_iqr_ratio: float = 10.0,
            min_jump_abs: float = 0.0,

            # Extreme values action (replaces mask_extreme_vals)
            extreme_action: str = "ffill",  # "nan" | "ffill" | "none"

            # Almost-constant detection
            const_eps_abs: float = 1e-8,
            const_eps_rel: float = 1e-6,
            const_unique_max: Optional[int] = 2,
            unique_round_decimals: int = 12,

            logger: Optional[logging.Logger] = None,

            # (5) Zero-inflated split (optional)
            enable_zero_inflated: bool = False,
            zero_frac_thresh: float = 0.50,
            zero_inflated_add_indicator: bool = True,
            zero_inflated_eps: float = 0.0,  # treat |x|<=eps as zero

            # (6) Heavy-tail asinh (optional)
            enable_asinh: bool = False,
            asinh_tail_ratio_thresh: float = 6.0,
            asinh_q_hi: float = 0.99,
            asinh_scale_mode: str = "median_abs",  # "median_abs" | "p50_abs" | "p90_abs" | "iqr"
            asinh_scale_min: float = 1e-6,

            # (9) Post-scaling clipping (optional)
            enable_clip: bool = False,
            clip_value: float = 10.0,
            min_scale: float = 1e-3,
            zero_inflated_stats_on_nonzero: bool = True,
            zero_inflated_asinh_scale_on_nonzero: bool = True,
            # Zero-inflated vs almost-constant disambiguation (NEW)
            zero_inflated_const_check_min_nnz: int = 8,
            zero_inflated_const_check_min_frac: float = 0.01,
    ) -> None:
        """Initialize the mixed robust preprocessing transformer.

        Args:
            columns (Optional[Sequence[str]]): Columns to process. Defaults to None.
            drop_features (Optional[Sequence[str]]): Selected features to remove before fitting.
            use_robust_scaling (bool): Use robust scaling when True. Defaults to True.
            quantile_range (Tuple[float, float]): Quantile range for robust scaling.
                Defaults to (25.0, 75.0).
            unit_variance (bool): Match RobustScaler unit variance behavior. Defaults to False.
            extreme_side (str): Tail side to inspect ("upper", "lower", "both"). Defaults to "both".
            q_ref (float): Reference quantile for tail detection. Defaults to 0.995.
            q_top (float): Top quantile for tail detection. Defaults to 0.999.
            max_tail_mass (float): Maximum tail mass for extreme detection. Defaults to 0.01.
            min_jump_iqr_ratio (float): Minimum jump/IQR ratio for extreme detection. Defaults to 10.0.
            min_jump_abs (float): Minimum absolute jump for extreme detection. Defaults to 0.0.
            extreme_action (str): Action for extreme values ("nan", "ffill", "none"). Defaults to "nan".
            const_eps_abs (float): Absolute epsilon for constant detection. Defaults to 1e-8.
            const_eps_rel (float): Relative epsilon for constant detection. Defaults to 1e-6.
            const_unique_max (Optional[int]): Max unique values to treat as constant. Defaults to 2.
            unique_round_decimals (int): Decimal rounding for uniqueness. Defaults to 12.
            logger (Optional[logging.Logger]): Optional logger. Defaults to None.
            enable_zero_inflated (bool): Enable zero-inflated handling. Defaults to False.
            zero_frac_thresh (float): Threshold for zero inflation. Defaults to 0.50.
            zero_inflated_add_indicator (bool): Add indicator column. Defaults to True.
            zero_inflated_eps (float): Epsilon for zero detection. Defaults to 0.0.
            enable_asinh (bool): Enable asinh transform. Defaults to False.
            asinh_tail_ratio_thresh (float): Tail ratio threshold for asinh. Defaults to 6.0.
            asinh_q_hi (float): Quantile for tail ratio. Defaults to 0.99.
            asinh_scale_mode (str): Scale mode for asinh. Defaults to "median_abs".
            asinh_scale_min (float): Minimum asinh scale. Defaults to 1e-6.
            enable_clip (bool): Enable output clipping. Defaults to False.
            clip_value (float): Clip value when enabled. Defaults to 10.0.
            min_scale (float): Minimum scale floor. Defaults to 1e-3.
            zero_inflated_stats_on_nonzero (bool): Use nonzero stats for ZI. Defaults to True.
            zero_inflated_asinh_scale_on_nonzero (bool): Use nonzero asinh scale. Defaults to True.
            zero_inflated_const_check_min_nnz (int): Min nonzero count for const check. Defaults to 8.
            zero_inflated_const_check_min_frac (float): Min nonzero fraction for const check. Defaults to 0.01.

        Raises:
            ValueError: If configuration values are invalid.
        """
        self.columns = list(columns) if columns is not None else None
        self.drop_features = list(drop_features) if drop_features is not None else None

        self.use_robust_scaling = bool(use_robust_scaling)
        self.quantile_range = tuple(map(float, quantile_range))
        self.unit_variance = bool(unit_variance)

        self.extreme_side = extreme_side
        self.q_ref = float(q_ref)
        self.q_top = float(q_top)
        self.max_tail_mass = float(max_tail_mass)
        self.min_jump_iqr_ratio = float(min_jump_iqr_ratio)
        self.min_jump_abs = float(min_jump_abs)

        self.extreme_action = str(extreme_action)
        if self.extreme_action not in {"nan", "ffill", "none"}:
            raise ValueError("extreme_action must be one of: 'nan', 'ffill', 'none'")

        self.const_eps_abs = float(const_eps_abs)
        self.const_eps_rel = float(const_eps_rel)
        self.const_unique_max = const_unique_max
        self.unique_round_decimals = int(unique_round_decimals)

        self.enable_zero_inflated = bool(enable_zero_inflated)
        self.zero_frac_thresh = float(zero_frac_thresh)
        self.zero_inflated_add_indicator = bool(zero_inflated_add_indicator)
        self.zero_inflated_eps = float(zero_inflated_eps)

        self.enable_asinh = bool(enable_asinh)
        self.asinh_tail_ratio_thresh = float(asinh_tail_ratio_thresh)
        self.asinh_q_hi = float(asinh_q_hi)
        self.asinh_scale_mode = str(asinh_scale_mode)
        self.asinh_scale_min = float(asinh_scale_min)

        self.enable_clip = bool(enable_clip)
        self.clip_value = float(clip_value)
        self.min_scale = float(min_scale)
        if not np.isfinite(self.min_scale) or self.min_scale <= 0.0:
            raise ValueError("min_scale must be a finite positive float")

        self.zero_inflated_stats_on_nonzero = bool(zero_inflated_stats_on_nonzero)
        self.zero_inflated_asinh_scale_on_nonzero = bool(zero_inflated_asinh_scale_on_nonzero)
        self.zero_inflated_const_check_min_nnz = int(zero_inflated_const_check_min_nnz)
        self.zero_inflated_const_check_min_frac = float(zero_inflated_const_check_min_frac)
        if self.zero_inflated_const_check_min_nnz < 1:
            raise ValueError("zero_inflated_const_check_min_nnz must be >= 1")
        if not (0.0 <= self.zero_inflated_const_check_min_frac <= 1.0):
            raise ValueError("zero_inflated_const_check_min_frac must be in [0,1]")

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # learned state
        self.feature_names_in_: List[Any] = []
        self.feature_names_out_: List[str] = []
        self.policies_: List[_FeaturePolicy] = []
        self.center_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}
        self.report_: Optional[pd.DataFrame] = None
        self.zero_frac_: Dict[str, float] = {}
        self._array_input_feature_indices: Optional[np.ndarray] = None
        self._almost_constant_plan: Optional[_TransformGroupPlan] = None
        self._robust_plan: Optional[_TransformGroupPlan] = None
        self._extreme_plan: Optional[_TransformGroupPlan] = None
        self._zero_inflated_plan: Optional[_TransformGroupPlan] = None
        self._transform_output_width = 0

    # -----------------------
    # Public sklearn API
    # -----------------------
    def fit(self, X: Any, y: Any = None) -> "MixedRobustPreprocessor":
        """Fit preprocessing statistics on input data.

        Args:
            X (Any): Input data.
            y (Any): Optional target values. Defaults to None.

        Returns:
            MixedRobustPreprocessor: Fitted transformer instance.
        """
        df = self._to_df(X)
        cols = self._select_columns(df)
        self.feature_names_in_ = cols

        if self.use_robust_scaling:
            self._validate_quantile_range()

        self.logger.info("Fitting %s on %d rows, %d features.", self.__class__.__name__, len(df), len(cols))
        self.logger.debug(
            "Config: use_robust_scaling=%s quantile_range=%s unit_variance=%s extreme_action=%s "
            "extreme_side=%s q_ref=%.6f q_top=%.6f max_tail_mass=%.6f min_jump_iqr_ratio=%.3f min_jump_abs=%.6g "
            "enable_zero_inflated=%s zero_frac_thresh=%.3f enable_asinh=%s asinh_tail_ratio_thresh=%.3f "
            "enable_clip=%s clip_value=%.3f",
            self.use_robust_scaling,
            self.quantile_range,
            self.unit_variance,
            self.extreme_action,
            self.extreme_side,
            self.q_ref,
            self.q_top,
            self.max_tail_mass,
            self.min_jump_iqr_ratio,
            self.min_jump_abs,
            self.enable_zero_inflated,
            self.zero_frac_thresh,
            self.enable_asinh,
            self.asinh_tail_ratio_thresh,
            self.enable_clip,
            self.clip_value,
        )

        report = self._build_report(df[cols])
        self.report_ = report

        almost_const_cols = report.loc[report["almost_constant"], "feature"].tolist()
        extreme_cols = report.loc[report["extreme_vals"], "feature"].tolist()

        zero_inflated_cols: List[str] = []
        if self.enable_zero_inflated:
            zi = report.loc[report["zero_frac"] >= self.zero_frac_thresh, "feature"].tolist()
            zero_inflated_cols = [c for c in zi if c not in almost_const_cols]

        self.logger.info(
            "Feature policy split: almost_constant=%d, zero_inflated=%d, extreme_vals=%d, other=%d",
            len(almost_const_cols),
            len(zero_inflated_cols),
            len(extreme_cols),
            len(cols) - len(almost_const_cols) - len(zero_inflated_cols) - len(extreme_cols),
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("almost_constant: %s", almost_const_cols)
            self.logger.debug("zero_inflated: %s", zero_inflated_cols)
            self.logger.debug("extreme_vals: %s", extreme_cols)

        # cutoffs for extreme features
        cutoffs: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for r in report.loc[report["extreme_vals"]].itertuples(index=False):
            cutoffs[r.feature] = (r.lower_cutoff, r.upper_cutoff)

        self.policies_.clear()
        self.center_.clear()
        self.scale_.clear()
        self.feature_names_out_.clear()
        self.zero_frac_.clear()

        for c in cols:
            x_raw = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)

            # decide kind (priority: almost_constant > zero_inflated > extreme_vals > robust)
            if c in zero_inflated_cols:
                kind = "zero_inflated"
            elif c in almost_const_cols:
                kind = "almost_constant"
            elif c in extreme_cols:
                kind = "extreme_vals"
            else:
                kind = "robust"

            lower_cutoff = None
            upper_cutoff = None
            if kind == "extreme_vals":
                lower_cutoff, upper_cutoff = cutoffs.get(c, (None, None))

            # store zero fraction on raw stream (debug)
            finite = np.isfinite(x_raw)
            if finite.any():
                zmask = finite & (np.abs(x_raw) <= self.zero_inflated_eps)
                self.zero_frac_[c] = float(np.mean(zmask[finite]))
            else:
                self.zero_frac_[c] = 0.0

            # for stats: apply extreme_action only to extreme_vals features
            x_stats = x_raw
            if kind == "extreme_vals":
                x_stats = self._apply_extreme_action(x_stats, lower_cutoff, upper_cutoff)

            # Decide asinh
            apply_asinh, asinh_scale = self._decide_asinh(x_stats, kind)

            # For zero_inflated, recompute asinh scale on nonzeros (recommended)
            if kind == "zero_inflated" and apply_asinh and self.zero_inflated_asinh_scale_on_nonzero:
                nz = np.isfinite(x_raw) & (np.abs(x_raw) > self.zero_inflated_eps)
                if nz.any():
                    asinh_scale = self._compute_asinh_scale(x_raw[nz])

            # Apply asinh for stats
            if apply_asinh:
                x_stats = self._asinh_transform(x_stats, asinh_scale)

            # Compute center/scale
            if kind == "zero_inflated" and self.zero_inflated_stats_on_nonzero:
                nz = np.isfinite(x_raw) & (np.abs(x_raw) > self.zero_inflated_eps)
                x_for_stats = x_stats[nz] if nz.any() else x_stats
            else:
                x_for_stats = x_stats

            center, scale = self._compute_center_scale(x_for_stats)

            # Scale floor
            scale = max(scale, self.min_scale)

            self.center_[c] = center
            self.scale_[c] = scale

            if kind == "zero_inflated":
                self.policies_.append(
                    _FeaturePolicy(
                        c,
                        kind,
                        lower_cutoff,
                        upper_cutoff,
                        apply_asinh=apply_asinh,
                        asinh_scale=asinh_scale,
                        add_indicator=self.zero_inflated_add_indicator,
                    )
                )
                if self.zero_inflated_add_indicator:
                    self.feature_names_out_.append(f"{c}__is_nonzero")
                self.feature_names_out_.append(f"{c}__mag")
            else:
                self.policies_.append(
                    _FeaturePolicy(
                        c,
                        kind,
                        lower_cutoff,
                        upper_cutoff,
                        apply_asinh=apply_asinh,
                        asinh_scale=asinh_scale,
                        add_indicator=False,
                    )
                )
                self.feature_names_out_.append(c)

        self._build_transform_plans()
        return self

    def transform(self, X: Any) -> Any:
        """Transform input data using the fitted preprocessing rules.

        Args:
            X (Any): Input data.

        Returns:
            Any: Transformed numpy array.

        Raises:
            RuntimeError: If the transformer is not fitted.
            KeyError: If required columns are missing.
        """
        self._check_is_fitted()
        input_array, restore_shape = self._prepare_transform_input(X)

        # --- handle empty input (0 rows) ---
        if input_array.shape[0] == 0:
            self.logger.debug("Transform called with empty input; returning empty array.")
            return np.empty((0, len(self.feature_names_out_)), dtype=np.float32)

        out = self._transform_selected_array(input_array)

        if self.enable_clip:
            np.clip(out, -self.clip_value, self.clip_value, out=out)

        if restore_shape is not None:
            out = out.reshape((*restore_shape, -1))
        return out

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names after transformation.

        Args:
            input_features (Any | None): Optional input feature names. Defaults to None.

        Returns:
            np.ndarray: Output feature names.
        """
        self._check_is_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return initialization parameters for hashing and sklearn compatibility.

        Args:
            deep (bool): Unused, present for sklearn compatibility.

        Returns:
            Dict[str, Any]: Initialization parameters.
        """
        return {
            "columns": list(self.columns) if self.columns is not None else None,
            "drop_features": list(self.drop_features) if self.drop_features is not None else None,
            "use_robust_scaling": self.use_robust_scaling,
            "quantile_range": self.quantile_range,
            "unit_variance": self.unit_variance,
            "extreme_side": self.extreme_side,
            "q_ref": self.q_ref,
            "q_top": self.q_top,
            "max_tail_mass": self.max_tail_mass,
            "min_jump_iqr_ratio": self.min_jump_iqr_ratio,
            "min_jump_abs": self.min_jump_abs,
            "extreme_action": self.extreme_action,
            "const_eps_abs": self.const_eps_abs,
            "const_eps_rel": self.const_eps_rel,
            "const_unique_max": self.const_unique_max,
            "unique_round_decimals": self.unique_round_decimals,
            "enable_zero_inflated": self.enable_zero_inflated,
            "zero_frac_thresh": self.zero_frac_thresh,
            "zero_inflated_add_indicator": self.zero_inflated_add_indicator,
            "zero_inflated_eps": self.zero_inflated_eps,
            "enable_asinh": self.enable_asinh,
            "asinh_tail_ratio_thresh": self.asinh_tail_ratio_thresh,
            "asinh_q_hi": self.asinh_q_hi,
            "asinh_scale_mode": self.asinh_scale_mode,
            "asinh_scale_min": self.asinh_scale_min,
            "enable_clip": self.enable_clip,
            "clip_value": self.clip_value,
            "min_scale": self.min_scale,
            "zero_inflated_stats_on_nonzero": self.zero_inflated_stats_on_nonzero,
            "zero_inflated_asinh_scale_on_nonzero": self.zero_inflated_asinh_scale_on_nonzero,
            "zero_inflated_const_check_min_nnz": self.zero_inflated_const_check_min_nnz,
            "zero_inflated_const_check_min_frac": self.zero_inflated_const_check_min_frac,
        }

    # -----------------------
    # Internals
    # -----------------------
    def _check_is_fitted(self) -> None:
        """Validate that the transformer has been fitted.

        Raises:
            RuntimeError: If the transformer is not fitted.
        """
        if not self.feature_names_in_ or not self.policies_ or not self.feature_names_out_:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet. Call fit() first.")

    def _build_transform_plans(self) -> None:
        self._array_input_feature_indices = self._resolve_array_input_feature_indices()
        self._transform_output_width = len(self.feature_names_out_)

        almost_constant_input: List[int] = []
        almost_constant_output: List[int] = []
        robust_input: List[int] = []
        robust_output: List[int] = []
        extreme_input: List[int] = []
        extreme_output: List[int] = []
        zero_inflated_input: List[int] = []
        zero_inflated_output: List[int] = []
        zero_indicator_local: List[int] = []
        zero_indicator_output: List[int] = []

        output_index = 0
        for feature_index, policy in enumerate(self.policies_):
            if policy.kind == "zero_inflated":
                local_index = len(zero_inflated_input)
                zero_inflated_input.append(feature_index)
                if policy.add_indicator:
                    zero_indicator_local.append(local_index)
                    zero_indicator_output.append(output_index)
                    output_index += 1
                zero_inflated_output.append(output_index)
                output_index += 1
                continue

            if policy.kind == "almost_constant":
                almost_constant_input.append(feature_index)
                almost_constant_output.append(output_index)
            elif policy.kind == "extreme_vals":
                extreme_input.append(feature_index)
                extreme_output.append(output_index)
            else:
                robust_input.append(feature_index)
                robust_output.append(output_index)
            output_index += 1

        self._almost_constant_plan = self._make_transform_group_plan(
            almost_constant_input,
            almost_constant_output,
        )
        self._robust_plan = self._make_transform_group_plan(
            robust_input,
            robust_output,
        )
        self._extreme_plan = self._make_transform_group_plan(
            extreme_input,
            extreme_output,
            include_cutoffs=True,
        )
        self._zero_inflated_plan = self._make_transform_group_plan(
            zero_inflated_input,
            zero_inflated_output,
            indicator_local_indices=zero_indicator_local,
            indicator_output_indices=zero_indicator_output,
        )

    def _make_transform_group_plan(
        self,
        input_indices: Sequence[int],
        output_indices: Sequence[int],
        *,
        indicator_local_indices: Optional[Sequence[int]] = None,
        indicator_output_indices: Optional[Sequence[int]] = None,
        include_cutoffs: bool = False,
    ) -> Optional[_TransformGroupPlan]:
        if not input_indices:
            return None

        policies = [self.policies_[index] for index in input_indices]
        centers = np.asarray([self.center_[policy.name] for policy in policies], dtype=np.float32)
        scales = np.asarray([self.scale_[policy.name] for policy in policies], dtype=np.float32)
        asinh_local_indices = np.asarray(
            [index for index, policy in enumerate(policies) if policy.apply_asinh],
            dtype=np.intp,
        )
        asinh_scales = np.asarray(
            [1.0 if policy.asinh_scale is None else policy.asinh_scale for policy in policies],
            dtype=np.float32,
        )

        lower_cutoffs = None
        upper_cutoffs = None
        if include_cutoffs:
            lower_cutoffs = np.asarray(
                [np.nan if policy.lower_cutoff is None else policy.lower_cutoff for policy in policies],
                dtype=np.float32,
            )
            upper_cutoffs = np.asarray(
                [np.nan if policy.upper_cutoff is None else policy.upper_cutoff for policy in policies],
                dtype=np.float32,
            )

        indicator_local_array = None
        indicator_output_array = None
        if indicator_local_indices:
            indicator_local_array = np.asarray(indicator_local_indices, dtype=np.intp)
            indicator_output_array = np.asarray(indicator_output_indices, dtype=np.intp)

        return _TransformGroupPlan(
            input_indices=np.asarray(input_indices, dtype=np.intp),
            output_indices=np.asarray(output_indices, dtype=np.intp),
            centers=centers,
            scales=scales,
            asinh_local_indices=asinh_local_indices,
            asinh_scales=asinh_scales,
            indicator_local_indices=indicator_local_array,
            indicator_output_indices=indicator_output_array,
            lower_cutoffs=lower_cutoffs,
            upper_cutoffs=upper_cutoffs,
        )

    def _resolve_array_input_feature_indices(self) -> Optional[np.ndarray]:
        array_indices: List[int] = []
        for feature_name in self.feature_names_in_:
            if isinstance(feature_name, (int, np.integer)):
                feature_index = int(feature_name)
                if feature_index < 0:
                    return None
                array_indices.append(feature_index)
                continue
            return None
        return np.asarray(array_indices, dtype=np.intp)

    def _prepare_transform_input(self, X: Any) -> Tuple[np.ndarray, Optional[Tuple[int, ...]]]:
        if isinstance(X, pd.DataFrame):
            return self._prepare_transform_input_from_df(X), None

        array = np.asarray(X)
        restore_shape = None
        if array.ndim == 3:
            restore_shape = array.shape[:-1]
            array = array.reshape((-1, array.shape[-1]))
        elif array.ndim == 2:
            pass
        elif array.ndim == 1:
            array = array.reshape((-1, 1))
        else:
            raise ValueError(f"Unsupported input rank for transform: {array.ndim}")

        return self._prepare_transform_input_from_array(array), restore_shape

    def _prepare_transform_input_from_df(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names_in_ if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns at transform time: {missing}")

        selected = df.loc[:, self.feature_names_in_]
        if selected.empty:
            return np.empty((len(selected), len(self.feature_names_in_)), dtype=np.float32)

        if all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
            return selected.to_numpy(dtype=np.float32, copy=False)

        coerced = selected.apply(pd.to_numeric, errors="coerce")
        return coerced.to_numpy(dtype=np.float32, copy=False)

    def _prepare_transform_input_from_array(self, array: np.ndarray) -> np.ndarray:
        if self._array_input_feature_indices is None:
            raise KeyError(f"Missing columns at transform time: {self.feature_names_in_}")

        if self._array_input_feature_indices.size:
            max_index = int(self._array_input_feature_indices.max())
            if array.shape[1] <= max_index:
                missing = [name for name in self.feature_names_in_ if int(name) >= array.shape[1]]
                raise KeyError(f"Missing columns at transform time: {missing}")
            selected = array[:, self._array_input_feature_indices]
        else:
            selected = array[:, :0]

        if np.issubdtype(selected.dtype, np.number):
            return np.asarray(selected, dtype=np.float32)

        coerced = pd.DataFrame(selected).apply(pd.to_numeric, errors="coerce")
        return coerced.to_numpy(dtype=np.float32, copy=False)

    def _transform_selected_array(self, x_selected: np.ndarray) -> np.ndarray:
        out = np.empty((x_selected.shape[0], self._transform_output_width), dtype=np.float32)

        if self._almost_constant_plan is not None:
            plan = self._almost_constant_plan
            out[:, plan.output_indices] = x_selected[:, plan.input_indices] - plan.centers

        if self._robust_plan is not None:
            plan = self._robust_plan
            out[:, plan.output_indices] = self._transform_scaled_group(
                x_selected[:, plan.input_indices],
                plan,
                apply_extreme_action=False,
            )

        if self._extreme_plan is not None:
            plan = self._extreme_plan
            out[:, plan.output_indices] = self._transform_scaled_group(
                x_selected[:, plan.input_indices],
                plan,
                apply_extreme_action=True,
            )

        if self._zero_inflated_plan is not None:
            plan = self._zero_inflated_plan
            x_raw = x_selected[:, plan.input_indices]
            if plan.indicator_local_indices is not None and plan.indicator_output_indices is not None:
                indicators = np.isfinite(x_raw[:, plan.indicator_local_indices]) & (
                    np.abs(x_raw[:, plan.indicator_local_indices]) > self.zero_inflated_eps
                )
                out[:, plan.indicator_output_indices] = indicators.astype(np.float32, copy=False)
            out[:, plan.output_indices] = self._transform_zero_inflated_group(x_raw, plan)

        return out

    def _transform_scaled_group(
        self,
        x_raw: np.ndarray,
        plan: _TransformGroupPlan,
        *,
        apply_extreme_action: bool,
    ) -> np.ndarray:
        transformed = x_raw
        if apply_extreme_action:
            transformed = self._apply_extreme_action_matrix(
                transformed,
                plan.lower_cutoffs,
                plan.upper_cutoffs,
            )
        if plan.asinh_local_indices.size:
            if transformed is x_raw:
                transformed = transformed.copy()
            self._apply_asinh_inplace(
                transformed,
                plan.asinh_local_indices,
                plan.asinh_scales,
            )
        return (transformed - plan.centers) / plan.scales

    def _transform_zero_inflated_group(
        self,
        x_raw: np.ndarray,
        plan: _TransformGroupPlan,
    ) -> np.ndarray:
        transformed = x_raw
        if plan.asinh_local_indices.size:
            transformed = transformed.copy()
            self._apply_asinh_inplace(
                transformed,
                plan.asinh_local_indices,
                plan.asinh_scales,
            )
        return (transformed - plan.centers) / plan.scales

    @staticmethod
    def _apply_asinh_inplace(
        x: np.ndarray,
        local_indices: np.ndarray,
        asinh_scales: np.ndarray,
    ) -> None:
        x[:, local_indices] = np.arcsinh(x[:, local_indices] / asinh_scales[local_indices])

    def _to_df(self, X: Any) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame.

        Args:
            X (Any): Input data.

        Returns:
            pd.DataFrame: DataFrame representation.
        """
        if isinstance(X, pd.DataFrame):
            return X
        if X.ndim == 3:
            X = X.reshape((-1, X.shape[-1]))
        return pd.DataFrame(X)

    def _select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select columns to process from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            List[str]: Selected column names.

        Raises:
            KeyError: If requested columns are missing.
            ValueError: If no columns are selected.
        """
        if self.columns is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = list(self.columns)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"Missing requested columns: {missing}")
        if self.drop_features is not None:
            requested_drops = list(self.drop_features)
            selected = set(cols)
            missing_drops = [c for c in requested_drops if c not in selected]
            if missing_drops:
                self.logger.warning(
                    "Configured drop_features are not selected for preprocessing and will be ignored: %s",
                    missing_drops,
                )
            drop_set = set(requested_drops)
            cols = [c for c in cols if c not in drop_set]
        if not cols:
            raise ValueError("No columns selected for preprocessing.")
        return cols

    def _validate_quantile_range(self) -> None:
        """Validate the quantile range configuration.

        Raises:
            ValueError: If the quantile range is invalid.
        """
        q_min, q_max = self.quantile_range
        if not (0.0 < q_min < q_max < 100.0):
            raise ValueError("quantile_range must satisfy 0 < q_min < q_max < 100")
        if self.unit_variance:
            adjust = float(stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0))
            if not np.isfinite(adjust) or adjust <= 0.0:
                raise ValueError("Invalid quantile_range for unit_variance adjustment")

    def _apply_extreme_action(
            self,
            x: np.ndarray,
            lower_cutoff: Optional[float],
            upper_cutoff: Optional[float],
    ) -> np.ndarray:
        """Apply the configured extreme-value action to the array.

        Args:
            x (np.ndarray): Input array.
            lower_cutoff (Optional[float]): Lower cutoff for extremes.
            upper_cutoff (Optional[float]): Upper cutoff for extremes.

        Returns:
            np.ndarray: Array with extreme values handled.
        """
        x_array = np.asarray(x)
        dtype = x_array.dtype if np.issubdtype(x_array.dtype, np.floating) else np.float64
        lower_cutoffs = np.asarray(
            [np.nan if lower_cutoff is None else lower_cutoff],
            dtype=dtype,
        )
        upper_cutoffs = np.asarray(
            [np.nan if upper_cutoff is None else upper_cutoff],
            dtype=dtype,
        )
        transformed = self._apply_extreme_action_matrix(
            np.asarray(x_array, dtype=dtype).reshape(-1, 1),
            lower_cutoffs,
            upper_cutoffs,
        )
        return transformed[:, 0]

    def _apply_extreme_action_matrix(
        self,
        x: np.ndarray,
        lower_cutoffs: Optional[np.ndarray],
        upper_cutoffs: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.extreme_action == "none" or x.size == 0:
            return x

        x2 = x.copy()
        finite = np.isfinite(x2)
        mask = np.zeros_like(x2, dtype=bool)

        if lower_cutoffs is not None:
            valid_lower = np.isfinite(lower_cutoffs)
            if valid_lower.any():
                mask |= finite & valid_lower[np.newaxis, :] & (x2 < lower_cutoffs[np.newaxis, :])
        if upper_cutoffs is not None:
            valid_upper = np.isfinite(upper_cutoffs)
            if valid_upper.any():
                mask |= finite & valid_upper[np.newaxis, :] & (x2 > upper_cutoffs[np.newaxis, :])

        if not mask.any():
            return x2

        x2[mask] = np.nan
        if self.extreme_action == "nan":
            return x2

        return self._forward_backward_fill_nan_matrix(x2)

    @staticmethod
    def _forward_backward_fill_nan_matrix(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x

        filled = x.copy()
        nan_mask = np.isnan(filled)
        if not nan_mask.any():
            return filled

        row_indices = np.arange(filled.shape[0], dtype=np.intp)[:, np.newaxis]
        col_indices = np.arange(filled.shape[1], dtype=np.intp)

        forward_indices = np.where(~nan_mask, row_indices, 0)
        np.maximum.accumulate(forward_indices, axis=0, out=forward_indices)
        filled = filled[forward_indices, col_indices]

        leading_nan_mask = np.isnan(filled)
        if leading_nan_mask.any():
            backward_indices = np.where(~leading_nan_mask, row_indices, filled.shape[0] - 1)
            backward_indices = np.minimum.accumulate(backward_indices[::-1], axis=0)[::-1]
            filled = filled[backward_indices, col_indices]

        return filled

    def _compute_center_scale(self, x: np.ndarray) -> Tuple[float, float]:
        """Compute center and scale for a feature stream.

        Args:
            x (np.ndarray): Input array.

        Returns:
            Tuple[float, float]: Center and scale values.
        """
        xf = x[np.isfinite(x)]
        if xf.size == 0:
            return 0.0, 1.0

        if self.use_robust_scaling:
            center = float(np.nanmedian(xf))
            scale = self._robust_scale_from_quantiles(xf)
            return center, scale

        center = float(np.nanmean(xf))
        scale = float(np.nanstd(xf, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return center, scale

    def _robust_scale_from_quantiles(self, x_f: np.ndarray) -> float:
        """Compute robust scale from quantiles.

        Args:
            x_f (np.ndarray): Finite values array.

        Returns:
            float: Robust scale value.
        """
        q_min, q_max = self.quantile_range
        lo = float(np.nanquantile(x_f, q_min / 100.0))
        hi = float(np.nanquantile(x_f, q_max / 100.0))
        scale = hi - lo

        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0

        if self.unit_variance:
            adjust = float(stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0))
            if not np.isfinite(adjust) or adjust <= 0.0:
                return 1.0
            scale = scale / adjust

        return float(scale)

    def _decide_asinh(self, x: np.ndarray, kind: str) -> Tuple[bool, Optional[float]]:
        """Decide whether to apply an asinh transform.

        Args:
            x (np.ndarray): Input array.
            kind (str): Feature kind label.

        Returns:
            Tuple[bool, Optional[float]]: Whether to apply asinh and the scale.
        """
        if not self.enable_asinh or kind not in {"robust", "extreme_vals", "zero_inflated"}:
            return False, None

        xr = x[np.isfinite(x)]
        if xr.size == 0:
            return False, None

        q50 = float(np.nanquantile(xr, 0.50))
        qhi = float(np.nanquantile(xr, self.asinh_q_hi))
        q25 = float(np.nanquantile(xr, 0.25))
        q75 = float(np.nanquantile(xr, 0.75))
        iqr = q75 - q25

        tail_ratio = (qhi - q50) / max(float(iqr), 1e-12)
        if np.isfinite(tail_ratio) and tail_ratio >= self.asinh_tail_ratio_thresh:
            return True, self._compute_asinh_scale(xr)
        return False, None

    def _compute_asinh_scale(self, x_finite: np.ndarray) -> float:
        """Compute the asinh scale based on the configured mode.

        Args:
            x_finite (np.ndarray): Finite values array.

        Returns:
            float: Scale value for asinh transform.
        """
        a = np.abs(x_finite[np.isfinite(x_finite)])
        if a.size == 0:
            return self.asinh_scale_min

        mode = self.asinh_scale_mode
        if mode == "median_abs":
            s = float(np.nanmedian(a))
        elif mode == "p50_abs":
            s = float(np.nanquantile(a, 0.50))
        elif mode == "p90_abs":
            s = float(np.nanquantile(a, 0.90))
        elif mode == "iqr":
            q25 = float(np.nanquantile(x_finite, 0.25))
            q75 = float(np.nanquantile(x_finite, 0.75))
            s = float(q75 - q25)
        else:
            raise ValueError("asinh_scale_mode must be one of: 'median_abs', 'p50_abs', 'p90_abs', 'iqr'")

        if not np.isfinite(s) or s < self.asinh_scale_min:
            s = self.asinh_scale_min
        return s

    @staticmethod
    def _asinh_transform(x: np.ndarray, scale: Optional[float]) -> np.ndarray:
        """Apply an asinh transform with a given scale.

        Args:
            x (np.ndarray): Input array.
            scale (Optional[float]): Scale parameter.

        Returns:
            np.ndarray: Transformed array.
        """
        s = 1.0 if (scale is None or not np.isfinite(scale) or scale <= 0.0) else float(scale)
        return np.arcsinh(x / s)

    def _build_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a per-feature report used for preprocessing decisions.

        Args:
            df (pd.DataFrame): Input DataFrame containing numeric features.

        Returns:
            pd.DataFrame: Report of per-feature statistics and policies.

        Raises:
            ValueError: If ``extreme_side`` is invalid.
        """
        side = self.extreme_side
        if side not in {"upper", "lower", "both"}:
            raise ValueError("extreme_side must be one of: 'upper', 'lower', 'both'")

        rows = []
        for c in df.columns:
            x_all = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(x_all)
            x = x_all[finite]
            if x.size == 0:
                continue

            n_total = int(x_all.size)
            n_finite = int(finite.sum())
            med = float(np.nanmedian(x))

            q1 = float(np.nanquantile(x, 0.25))
            q3 = float(np.nanquantile(x, 0.75))
            iqr = q3 - q1

            # almost-constant detection
            # zero fraction (for zero-inflated detection)
            zmask = finite & (np.abs(x_all) <= self.zero_inflated_eps)
            zero_frac = float(np.mean(zmask[finite])) if n_finite else 0.0

            # almost-constant detection
            # Default: compute on ALL finite values
            x_for_const = x

            # If the column is zero-inflated (high zero mass), recompute const-ness on NONZERO support
            # so sparse-but-informative impulse channels are not misclassified as almost-constant.
            if self.enable_zero_inflated and (zero_frac >= self.zero_frac_thresh):
                # non-zero mask defined on finite values
                nz = np.abs(x) > self.zero_inflated_eps
                x_nz = x[nz]

                # only trust the nonzero-only check if there is enough nonzero data
                min_nnz = getattr(self, "zero_inflated_const_check_min_nnz", 8)
                min_frac = getattr(self, "zero_inflated_const_check_min_frac", 0.01)
                if (x_nz.size >= min_nnz) and (x_nz.size / max(1, x.size) >= min_frac):
                    x_for_const = x_nz
                else:
                    # Not enough nonzeros to assess const-ness reliably; treat as NOT almost-constant
                    # (it can still become almost_constant later if truly constant via other logic).
                    x_for_const = None

            if x_for_const is None:
                nunique = None
                mad = 0.0
                const_thr = 0.0
                almost_constant_unique = False
                almost_constant = False
            else:
                med_const = float(np.nanmedian(x_for_const))
                # robust dispersion on chosen support
                mad = float(np.nanmedian(np.abs(x_for_const - med_const)))

                q1_const = float(np.nanquantile(x_for_const, 0.25))
                q3_const = float(np.nanquantile(x_for_const, 0.75))
                iqr_const = q3_const - q1_const

                scale_ref = max(1.0, abs(med_const))
                const_thr = self.const_eps_abs + self.const_eps_rel * scale_ref

                if self.const_unique_max is not None:
                    x_round = np.round(x_for_const, self.unique_round_decimals)
                    nunique = int(np.unique(x_round).size)
                    almost_constant_unique = (nunique <= self.const_unique_max)
                else:
                    nunique = None
                    almost_constant_unique = False

                almost_constant = bool((iqr_const <= const_thr) or (mad <= const_thr) or almost_constant_unique)

            # extreme tail diagnostics
            uq_ref = float(np.nanquantile(x, self.q_ref))
            uq_top = float(np.nanquantile(x, self.q_top))
            u_jump = uq_top - uq_ref
            u_mass = float(np.mean(x > uq_ref))
            u_ratio = (u_jump / iqr) if iqr > 0 else np.inf
            upper_flag = (u_mass <= self.max_tail_mass) and (u_ratio >= self.min_jump_iqr_ratio) and (
                        u_jump >= self.min_jump_abs)

            lq_ref = float(np.nanquantile(x, 1.0 - self.q_ref))
            lq_top = float(np.nanquantile(x, 1.0 - self.q_top))
            l_jump = lq_ref - lq_top
            l_mass = float(np.mean(x < lq_ref))
            l_ratio = (l_jump / iqr) if iqr > 0 else np.inf
            lower_flag = (l_mass <= self.max_tail_mass) and (l_ratio >= self.min_jump_iqr_ratio) and (
                        l_jump >= self.min_jump_abs)

            if side == "upper":
                extreme_vals = bool(upper_flag)
                lower_cutoff = None
                upper_cutoff = uq_ref if extreme_vals else None
                is_ext = finite & (x_all > uq_ref) if extreme_vals else np.zeros_like(finite, dtype=bool)
            elif side == "lower":
                extreme_vals = bool(lower_flag)
                lower_cutoff = lq_ref if extreme_vals else None
                upper_cutoff = None
                is_ext = finite & (x_all < lq_ref) if extreme_vals else np.zeros_like(finite, dtype=bool)
            else:  # both
                extreme_vals = bool(upper_flag or lower_flag)
                lower_cutoff = lq_ref if (extreme_vals and lower_flag) else None
                upper_cutoff = uq_ref if (extreme_vals and upper_flag) else None
                if extreme_vals:
                    is_ext = np.zeros_like(finite, dtype=bool)
                    if upper_cutoff is not None:
                        is_ext |= finite & (x_all > upper_cutoff)
                    if lower_cutoff is not None:
                        is_ext |= finite & (x_all < lower_cutoff)
                else:
                    is_ext = np.zeros_like(finite, dtype=bool)

            n_extreme = int(is_ext.sum())
            frac_extreme = (n_extreme / n_finite) if n_finite else 0.0

            const_support = "all"
            if self.enable_zero_inflated and (zero_frac >= self.zero_frac_thresh):
                const_support = "nonzero" if (
                            x_for_const is not None and x_for_const is not x) else "insufficient_nonzero"

            rows.append(
                dict(
                    feature=c,
                    n_total=n_total,
                    n_finite=n_finite,
                    median=med,
                    iqr=float(iqr),
                    mad=mad,
                    nunique=nunique,
                    const_thr=float(const_thr),
                    almost_constant=almost_constant,
                    zero_frac=zero_frac,
                    extreme_vals=extreme_vals,
                    n_extreme=n_extreme,
                    frac_extreme=float(frac_extreme),
                    lower_cutoff=lower_cutoff,
                    upper_cutoff=upper_cutoff,
                    upper_tail_mass=u_mass,
                    upper_jump=u_jump,
                    upper_jump_iqr_ratio=float(u_ratio),
                    lower_tail_mass=l_mass,
                    lower_jump=l_jump,
                    lower_jump_iqr_ratio=float(l_ratio),
                    const_support=const_support,
                )
            )

        rep = pd.DataFrame(rows).sort_values(
            ["almost_constant", "extreme_vals", "frac_extreme"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        self.logger.info(
            "Report: almost_constant=%d, extreme_vals=%d (of %d).",
            int(rep["almost_constant"].sum()),
            int(rep["extreme_vals"].sum()),
            len(rep),
        )
        return rep
