import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "data_transform.py"
)
SPEC = importlib.util.spec_from_file_location("test_data_transform_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DATA_TRANSFORM_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATA_TRANSFORM_MODULE
SPEC.loader.exec_module(DATA_TRANSFORM_MODULE)
MixedRobustPreprocessor = DATA_TRANSFORM_MODULE.MixedRobustPreprocessor


def _legacy_transform(preprocessor: MixedRobustPreprocessor, X):
    preprocessor._check_is_fitted()
    df = preprocessor._to_df(X)

    if len(df) == 0:
        missing = [c for c in preprocessor.feature_names_in_ if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns at transform time: {missing}")
        return np.empty((0, len(preprocessor.feature_names_out_)), dtype=np.float32)

    missing = [c for c in preprocessor.feature_names_in_ if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns at transform time: {missing}")

    out_cols = []
    for policy in preprocessor.policies_:
        column = policy.name
        x_raw = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)

        x = x_raw
        if policy.kind == "extreme_vals":
            x = preprocessor._apply_extreme_action(x, policy.lower_cutoff, policy.upper_cutoff)

        if policy.kind == "almost_constant":
            out_cols.append(x - preprocessor.center_[column])
            continue

        if policy.kind == "zero_inflated":
            if policy.add_indicator:
                finite = np.isfinite(x_raw)
                indicator = np.zeros_like(x_raw, dtype=np.float64)
                indicator[finite] = (np.abs(x_raw[finite]) > preprocessor.zero_inflated_eps).astype(np.float64)
                out_cols.append(indicator)

            magnitude = x_raw
            if policy.apply_asinh:
                magnitude = preprocessor._asinh_transform(magnitude, policy.asinh_scale)
            magnitude = (magnitude - preprocessor.center_[column]) / preprocessor.scale_[column]
            out_cols.append(magnitude)
            continue

        if policy.apply_asinh:
            x = preprocessor._asinh_transform(x, policy.asinh_scale)
        out_cols.append((x - preprocessor.center_[column]) / preprocessor.scale_[column])

    out = np.column_stack(out_cols).astype(np.float32, copy=False)
    if preprocessor.enable_clip:
        np.clip(out, -preprocessor.clip_value, preprocessor.clip_value, out=out)

    if not isinstance(X, pd.DataFrame) and np.asarray(X).ndim == 3:
        out = out.reshape((*np.asarray(X).shape[:-1], -1))

    return out


def _make_training_array() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 3.0],
            [1.0, 5.0, 1.0, 4.0],
            [1.0, 0.0, 1.0, 5.0],
            [1.0, 6.0, 1.0, 20.0],
            [1.0, 0.0, 1.0, 30.0],
            [1.0, 7.0, 100.0, 40.0],
        ],
        dtype=np.float32,
    )


def _make_transform_array() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0, 2.0],
            [1.0, 9.0, 200.0, 6.0],
            [1.0, 0.0, 1.0, 25.0],
            [1.0, 8.0, 1.0, 45.0],
        ],
        dtype=np.float32,
    )


def _make_preprocessor(columns=None) -> MixedRobustPreprocessor:
    return MixedRobustPreprocessor(
        columns=columns,
        use_robust_scaling=False,
        enable_zero_inflated=True,
        zero_frac_thresh=0.4,
        zero_inflated_add_indicator=True,
        enable_asinh=True,
        asinh_tail_ratio_thresh=1.1,
        extreme_action="ffill",
        extreme_side="upper",
        q_ref=0.75,
        q_top=0.95,
        max_tail_mass=0.4,
        min_jump_iqr_ratio=0.2,
        min_jump_abs=0.1,
        enable_clip=True,
        clip_value=8.0,
    )


def test_mixed_robust_preprocessor_transform_matches_legacy_numpy_path() -> None:
    preprocessor = _make_preprocessor()
    preprocessor.fit(_make_training_array())

    inputs_2d = _make_transform_array()
    expected_2d = _legacy_transform(preprocessor, inputs_2d)
    actual_2d = preprocessor.transform(inputs_2d)

    assert actual_2d.dtype == np.float32
    assert np.allclose(actual_2d, expected_2d, equal_nan=True)

    inputs_3d = inputs_2d.reshape(2, 2, inputs_2d.shape[-1])
    expected_3d = _legacy_transform(preprocessor, inputs_3d)
    actual_3d = preprocessor.transform(inputs_3d)

    assert actual_3d.shape == expected_3d.shape
    assert np.allclose(actual_3d, expected_3d, equal_nan=True)


def test_mixed_robust_preprocessor_transform_matches_legacy_dataframe_path() -> None:
    columns = ["const", "zero_inflated", "extreme", "robust"]
    training_df = pd.DataFrame(_make_training_array(), columns=columns)
    preprocessor = _make_preprocessor(columns=columns)
    preprocessor.fit(training_df)

    transform_df = pd.DataFrame(_make_transform_array(), columns=columns)
    transform_df["ignored"] = [10.0, 11.0, 12.0, 13.0]
    transform_df["extreme"] = ["1.0", "bad", "1.0", "300.0"]

    expected = _legacy_transform(preprocessor, transform_df)
    actual = preprocessor.transform(transform_df)

    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, equal_nan=True)


def test_mixed_robust_preprocessor_transform_raises_for_ndarray_after_named_fit() -> None:
    columns = ["const", "zero_inflated", "extreme", "robust"]
    training_df = pd.DataFrame(_make_training_array(), columns=columns)
    preprocessor = _make_preprocessor(columns=columns)
    preprocessor.fit(training_df)

    try:
        preprocessor.transform(_make_transform_array())
    except KeyError as exc:
        assert "Missing columns at transform time" in str(exc)
    else:
        raise AssertionError("Expected KeyError for ndarray input after fitting with named columns.")


def test_mixed_robust_preprocessor_drops_selected_features_and_warns(caplog) -> None:
    columns = ["const", "zero_inflated", "extreme", "robust"]
    training_df = pd.DataFrame(_make_training_array(), columns=columns)
    preprocessor = MixedRobustPreprocessor(
        columns=columns,
        drop_features=["zero_inflated", "missing_feature"],
        use_robust_scaling=False,
    )

    preprocessor.fit(training_df)

    assert preprocessor.feature_names_in_ == ["const", "extreme", "robust"]
    assert preprocessor.get_params()["drop_features"] == ["zero_inflated", "missing_feature"]
    assert "missing_feature" in caplog.text


def test_mixed_robust_preprocessor_raises_when_drop_features_remove_all_columns() -> None:
    training_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    preprocessor = MixedRobustPreprocessor(drop_features=["a", "b"])

    try:
        preprocessor.fit(training_df)
    except ValueError as exc:
        assert "No columns selected" in str(exc)
    else:
        raise AssertionError("Expected ValueError when all selected columns are dropped.")
