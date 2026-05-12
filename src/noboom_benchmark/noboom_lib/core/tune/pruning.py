import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from ray import tune
from ..tune_constants import TUNE_PROGRESS_ATTR


logger = logging.getLogger(__name__)
_MEAN_METRIC_EXCLUSIONS = {"alarm_score"}


def has_explicit_hpo_seeds(study_params: Mapping[str, Any]) -> bool:
    return "hpo_seeds" in study_params and study_params.get("hpo_seeds") is not None


def resolve_hpo_seeds(study_params: Mapping[str, Any]) -> List[int]:
    seed_source = (
        study_params.get("hpo_seeds")
        if has_explicit_hpo_seeds(study_params)
        else study_params.get("seeds", [])
    )
    source_name = "hpo_seeds" if has_explicit_hpo_seeds(study_params) else "seeds"
    if (
        seed_source is None
        or isinstance(seed_source, (str, bytes))
        or not isinstance(seed_source, Sequence)
    ):
        raise ValueError(f"study.{source_name} must be a sequence of integer seeds.")

    seeds = [int(seed) for seed in seed_source]
    if has_explicit_hpo_seeds(study_params) and not seeds:
        raise ValueError("study.hpo_seeds must contain at least one seed when provided.")
    return seeds


def resolve_full_seeds(study_params: Mapping[str, Any]) -> List[int]:
    seed_source = study_params.get("seeds", [])
    if (
        seed_source is None
        or isinstance(seed_source, (str, bytes))
        or not isinstance(seed_source, Sequence)
    ):
        raise ValueError("study.seeds must be a sequence of integer seeds.")
    return [int(seed) for seed in seed_source]


def _build_reduced_seed_base_context(study_params: Mapping[str, Any]) -> Dict[str, Any]:
    if not has_explicit_hpo_seeds(study_params):
        return {}

    hpo_seeds = resolve_hpo_seeds(study_params)
    full_seeds = resolve_full_seeds(study_params)
    if hpo_seeds == full_seeds:
        return {}

    return {
        "uses_reduced_hpo_seeds": True,
        "hpo_seeds": hpo_seeds,
        "full_seeds": full_seeds,
        "final_eval_seeds": full_seeds,
        "hpo_seed_count": len(hpo_seeds),
        "full_seed_count": len(full_seeds),
        "final_eval_seed_count": len(full_seeds),
    }


def build_reduced_hpo_seed_context(study_params: Mapping[str, Any]) -> Dict[str, Any]:
    context = _build_reduced_seed_base_context(study_params)
    if not context:
        return {}
    return {
        "seed_evaluation_scope": "hpo",
        **context,
    }


def build_final_eval_seed_context(study_params: Mapping[str, Any]) -> Dict[str, Any]:
    context = _build_reduced_seed_base_context(study_params)
    if not context:
        return {}
    return {
        "seed_evaluation_scope": "final_eval",
        **context,
    }


class TrialPrunedError(RuntimeError):
    def __init__(self, payload: Dict[str, Any]):
        super().__init__("Ray Tune stopped the current trial.")
        self.payload = payload


def aggregate_completed_metrics(
    *,
    metric_history: Dict[str, List[float]],
    best_metric_history: Dict[str, List[float]],
) -> Dict[str, Any]:
    def _metric_greater_is_better(metric_name: str) -> bool:
        return metric_name not in {"aaf"}

    mean_metrics = {
        name: float(np.mean(values))
        for name, values in metric_history.items()
        if values
    }
    best_seed_metrics = {
        f"best_{name}": float(
            np.max(values) if _metric_greater_is_better(name) else np.min(values)
        )
        for name, values in metric_history.items()
        if values
    }
    theory_best_metrics = {
        f"theory_best_{name}": float(np.mean(values))
        for name, values in best_metric_history.items()
        if values
    }

    payload = mean_metrics | best_seed_metrics | theory_best_metrics
    payload.update(
        {
            "mean_" + name: value
            for name, value in mean_metrics.items()
            if name not in _MEAN_METRIC_EXCLUSIONS
        }
    )
    return payload


class SeedAwareMetricReporter:
    def __init__(
        self,
        *,
        tune_metric: str,
        seed_context: Optional[Mapping[str, Any]] = None,
        report_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._tune_metric = tune_metric
        self._seed_context = dict(seed_context or {})
        self._report_fn = report_fn or tune.report
        self._completed_seed_scores: List[float] = []
        self._last_payload: Dict[str, Any] = {}

    @property
    def tune_metric(self) -> str:
        return self._tune_metric

    @property
    def completed_seed_count(self) -> int:
        return len(self._completed_seed_scores)

    @property
    def last_payload(self) -> Dict[str, Any]:
        return dict(self._last_payload)

    def note_completed_seed(self, score: float) -> None:
        self._completed_seed_scores.append(float(score))

    def build_completed_payload(
        self,
        *,
        metric_history: Dict[str, List[float]],
        best_metric_history: Dict[str, List[float]],
        trial_run_id: str,
        seed: Optional[int] = None,
        dataset_name: Optional[str] = None,
        dataset_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = aggregate_completed_metrics(
            metric_history=metric_history,
            best_metric_history=best_metric_history,
        )
        payload.update(
            {
                "n_seeds": self.completed_seed_count - 1,
                "completed_seed_count": self.completed_seed_count,
                "mlflow_run_id": trial_run_id,
                TUNE_PROGRESS_ATTR: self.completed_seed_count,
            }
        )
        payload.update(self._seed_context)
        if seed is not None:
            payload["seed"] = seed
        if dataset_name is not None:
            payload["dataset_name"] = dataset_name
        if dataset_digest is not None:
            payload["dataset_digest"] = dataset_digest
        return payload

    def emit_completed_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._emit(payload)
        return payload

    def _emit(self, payload: Dict[str, Any]) -> None:
        self._last_payload = dict(payload)
        try:
            self._report_fn(payload)
        except SystemExit as exc:
            raise TrialPrunedError(payload) from exc
