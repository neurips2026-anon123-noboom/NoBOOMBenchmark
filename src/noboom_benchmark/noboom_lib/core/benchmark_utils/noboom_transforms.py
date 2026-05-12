"""NoBoom-specific timesead Transform classes.

These compose with the standard ``timesead.data.transforms`` pipeline. They are
written from scratch (not subclassed from ``FiniteDifferencesTargetTransform``)
because the existing finite-differences transform emits its diffs in the
*target* slot, whereas KAN-AD's CTE step (paper Section 3.2) requires the
diffs to land in the *input* slot so that downstream prediction transforms see
them as the actual signal the model fits.

Two transforms are provided:

* :class:`FirstOrderDifferenceTransform` — replaces inputs with their
  first-order finite differences and shifts per-timestep labels to the
  finite-difference endpoints.
* :class:`PerWindowStandardiseInputTransform` — z-scores the (already
  windowed) input by per-window statistics and applies the same scale/shift to
  any per-timestep target tensors with matching trailing dimension. This
  realises the paper's "subsequently renormalize the differenced data" step
  on top of the global z-score that NoBoom's :class:`MixedRobustPreprocessor`
  applies to raw inputs upstream.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch

from timesead.data.transforms.transform_base import Transform


class FirstOrderDifferenceTransform(Transform):
    """Emit first-order finite differences as the dataset's primary signal.

    For each input tensor ``x`` shaped ``(T, ...)``, returns
    ``dx[k] = x[k+1] - x[k]`` of shape ``(T - 1, ...)``. Per-timestep label
    tensors lose their first entry so label at position ``k + 1`` pairs with
    ``dx[k]``, the endpoint represented by the forward difference.
    """

    def __init__(self, parent: Transform) -> None:
        super().__init__(parent)

    def _truncate_targets(
        self, targets: Tuple[torch.Tensor, ...], reference_len: int
    ) -> Tuple[torch.Tensor, ...]:
        truncated: List[torch.Tensor] = []
        for tensor in targets:
            if tensor.ndim >= 1 and tensor.shape[0] == reference_len + 1:
                truncated.append(tensor[1:])
            else:
                truncated.append(tensor)
        return tuple(truncated)

    def _get_datapoint_impl(
        self, item: int
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)
        new_inputs = tuple(inp[1:] - inp[:-1] for inp in inputs)
        if not new_inputs:
            return new_inputs, targets
        new_targets = self._truncate_targets(targets, new_inputs[0].shape[0])
        return new_inputs, new_targets

    @property
    def seq_len(self) -> Union[int, List[int]]:
        parent_seq_len = self.parent.seq_len
        if isinstance(parent_seq_len, int):
            return max(parent_seq_len - 1, 0)
        return [max(s - 1, 0) for s in parent_seq_len]


class PerWindowStandardiseInputTransform(Transform):
    """Per-window z-score the input tensor; mirror the same scale to compatible targets.

    Statistics are computed from ``inputs[0]`` (treated as the canonical
    signal). The same per-channel mean/std are used to standardise:

    * every entry in ``inputs`` (including duplicates), and
    * every target tensor whose trailing dimension matches ``inputs[0]``'s
      trailing dimension and whose ``ndim`` is the same — these are
      prediction-style targets (e.g. the next-step ``dx`` produced by
      :class:`timesead.data.transforms.PredictionTargetTransform`).

    Per-timestep label tensors (``ndim`` < ``inputs[0].ndim`` or trailing-dim
    mismatch) are passed through untouched so anomaly labels are not
    corrupted.

    The transform is stateless and operates per-item, which keeps it
    composable with the rest of the timesead pipeline. It is *not* a
    substitute for a global z-score over the training set, but it tracks the
    per-window statistics of the differenced signal — closing the same gap
    the paper's "subsequently renormalize the differenced data" step
    targets, while staying inside the existing transform contract.
    """

    def __init__(self, parent: Transform, eps: float = 1e-6) -> None:
        super().__init__(parent)
        self.eps = float(eps)

    def _standardise_target(
        self,
        tensor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        ref_ndim: int,
        ref_trailing: int,
    ) -> torch.Tensor:
        if (
            tensor.ndim == ref_ndim
            and tensor.ndim >= 1
            and tensor.shape[-1] == ref_trailing
        ):
            return (tensor - mean) / std
        return tensor

    def _get_datapoint_impl(
        self, item: int
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)
        if not inputs:
            return inputs, targets

        reference = inputs[0]
        if reference.ndim == 0:
            return inputs, targets

        # Time axis is the leading dimension after windowing/transform stacks.
        mean = reference.mean(dim=0, keepdim=True)
        std = reference.std(dim=0, keepdim=True).clamp_min(self.eps)
        ref_ndim = reference.ndim
        ref_trailing = reference.shape[-1] if reference.ndim >= 1 else 0

        new_inputs = tuple((inp - mean) / std for inp in inputs)
        new_targets = tuple(
            self._standardise_target(t, mean, std, ref_ndim, ref_trailing) for t in targets
        )
        return new_inputs, new_targets
