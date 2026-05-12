# Benchmark utilities

## MixedRobustPreprocessor scaling behavior
The `MixedRobustPreprocessor` in `data_transform.py` is the default scaler for tabular features. It converts Pandas DataFrames into a NumPy array with deterministic column order while applying feature-wise policies:

- **Feature categorization**: during `fit`, each column is tagged as `almost_constant`, `zero_inflated`, `extreme_vals`, or `robust` based on its distribution. Almost-constant columns are centered only; zero-inflated columns may be split into an indicator and magnitude stream; the rest follow robust or standard scaling. A feature is **almost constant** when its robust dispersion (IQR or MAD) falls below a threshold or its rounded unique count is tiny; for zero-heavy features this check is recomputed on the non-zero support only if there are enough non-zero samples. A feature is **zero inflated** when its zero mass exceeds `zero_frac_thresh` and it is *not* already almost-constant, meaning it preserves meaningful variation away from zero that should be scaled separately rather than collapsed to a constant stream.
- **Optional asinh compression**: if `enable_asinh` is set, heavy-tailed features can be compressed via `asinh(x / s)` (with `s` chosen per feature using the configured `asinh_scale_mode`) before any centering/scaling is computed.
- **Extreme value handling**: tails are detected using the `extreme_side`, `q_ref`, `q_top`, `max_tail_mass`, and jump thresholds. Depending on `extreme_action`, extreme points are replaced with NaN (`nan`), forward/back-filled (`ffill`), or left unchanged (`none`) before scaling.
- **Scaling modes**:
  - *Robust mode* (`use_robust_scaling=True`): centers with the median and scales by the quantile span `Q(q_max) - Q(q_min)` from `quantile_range`. If `unit_variance` is enabled, the scale is divided by the normal quantile gap `norm.ppf(q_max/100) - norm.ppf(q_min/100)` to mimic sklearn's `RobustScaler(unit_variance=True)`.
  - *Standard mode* (`use_robust_scaling=False`): centers with the mean and scales by the standard deviation (std) of the feature.
  - In either mode, scales are floored at `min_scale` to avoid exploding outputs when the variance is tiny.
- **Zero-inflated treatment**: when `enable_zero_inflated` is active and a column's zero fraction exceeds `zero_frac_thresh`, the transformer can emit an indicator (`__is_nonzero`) plus a magnitude stream (`__mag`) that is scaled using only non-zero values if `zero_inflated_stats_on_nonzero` is set.
- **Post-scaling clipping**: when `enable_clip` is true, the final outputs are clipped to `[-clip_value, clip_value]` to stabilize downstream reconstruction losses.
- **Outputs and feature names**: `transform` returns a `float32` NumPy array whose columns follow the `get_feature_names_out()` ordering, reflecting any indicator/magnitude expansions.

Use `fit_transform` on a training DataFrame to learn per-feature centers/scales, then reuse the fitted instance to transform validation/test splits identically.
