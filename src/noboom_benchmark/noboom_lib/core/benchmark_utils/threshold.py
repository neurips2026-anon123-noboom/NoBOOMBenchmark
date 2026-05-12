"""Baseline threshold objective for the NoBoom benchmark."""

from argparse import Namespace
from typing import Dict, List, Tuple

from joblib import delayed, Parallel
import numpy as np
import torch
from timesead.data.transforms import PipelineDataset

from .arrow_runtime import build_dataset_source
from ..metric_utils.alarm_threshold_search import select_threshold
from ..metric_utils.evaluation_postprocessing import (
    evaluate_sequences_with_optional_postprocessing,
    resolve_evaluation_postprocessing_config,
)
from ..metric_utils.metrics import get_metric_by_name, is_metric_binary
from ..data_prep.manifest import PreparedDatasetManifest
from ..data_prep.runtime import manifest_path_for_stage_from_args

#TODO: Replace with AnomalyDetector for consistency

def objective_threshold(args: Namespace, hp_params: dict, metrics: List[str], target_metric: str = 'alarm_score') \
        -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray, float, int]:
    """Optuna objective for the moving-mean threshold baseline.

    Args:
        args (Namespace): Parsed CLI arguments containing dataset info.
        hp_params (dict): Hyperparameters for the threshold detector.
        metrics (List[str]): Metric names to evaluate.
        target_metric (str): Metric to optimize. Defaults to "alarm_score".

    Returns:
        Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray, float, int]:
        Metrics mapping, raw anomaly scores, labels, best threshold, and best feature index.

    Side Effects:
        Loads datasets from disk and runs multi-threaded evaluation.
    """
    from noboom.tsad.baselines import MovingMeanDifferenceAnomalyDetector
    assert target_metric in metrics

    manifest_path = manifest_path_for_stage_from_args(args, "real")
    manifest = PreparedDatasetManifest.load(manifest_path)
    eval_data = build_dataset_source(manifest, mode="test")
    detector = MovingMeanDifferenceAnomalyDetector(hp_params["window_size"], hp_params["use_mean"],
                                                   hp_params["std_coefficient"],
                                                   two_sided=hp_params["two_sided"])

    eval_scores = []
    eval_labels = []
    eval_score_sequences = []
    eval_label_sequences = []
    for ts, labels in PipelineDataset(eval_data):
        scores = detector.score(ts[0])
        eval_scores.extend([torch.zeros((1, scores.shape[1])), scores])
        eval_labels.extend([torch.zeros(1), labels[0]])
        eval_score_sequences.append(scores.cpu().numpy())
        eval_label_sequences.append(labels[0].cpu().numpy())

    score_stack_eval = torch.cat(eval_scores).numpy()
    label_stack_eval = torch.cat(eval_labels).numpy()
    study_params = getattr(args, "study_params", None)
    postprocessing_config = resolve_evaluation_postprocessing_config(
        study_params.get("evaluation_postprocessing")
        if study_params is not None
        else None
    )

    metric_fns = {metric: get_metric_by_name(metric) for metric in metrics}

    def get_labels(metric_name: str, labels_: np.ndarray) -> np.ndarray:
        if is_metric_binary(metric_name):
            return (labels_ > 0).astype(int)
        return labels_

    def worker_func(i: int, scores_eval: np.ndarray, labels_eval:np.ndarray):
        """Evaluate a single feature index for threshold selection.

        Args:
            i (int): Feature index to evaluate.

        Returns:
            tuple: Metric results, best threshold, feature index, predictions.
        """
        scores_eval = scores_eval[:, i]
        labels_eval = labels_eval[-scores_eval.shape[0]:]
        if postprocessing_config.enabled:
            processed = evaluate_sequences_with_optional_postprocessing(
                [sequence[:, i] for sequence in eval_score_sequences],
                eval_label_sequences,
                metric_names=tuple(metrics),
                config=postprocessing_config,
            )
            threshold = processed.threshold
            eval_predictions = processed.predictions
            eval_metric_results = dict(processed.metric_values)
        else:
            threshold = select_threshold(scores_eval, get_labels(target_metric, labels_eval)).best_threshold
            eval_predictions = (scores_eval > threshold).astype(int)
            eval_metric_results = {
                name: metric_fn(eval_predictions, get_labels(name, labels_eval))
                for name, metric_fn in metric_fns.items()
            }
        return eval_metric_results, threshold, i, eval_predictions

    result = Parallel(n_jobs=4, verbose=2)(delayed(worker_func)(i, score_stack_eval, label_stack_eval)
                                for i in range(score_stack_eval.shape[1]))
    best_metric_results, best_threshold, best_feature, best_predictions = (
        sorted(result, key=lambda x: x[0][target_metric], reverse=True))[0]
    best_scores = score_stack_eval[:, best_feature]

    eval_labels = label_stack_eval[-best_predictions.shape[0]:]
    eval_labels_bin = (eval_labels > 0).astype(int)
    for metric_name in best_metric_results.keys():
        metric_fn = metric_fns[metric_name]
        if is_metric_binary(metric_name):
            best_metric = metric_fn(eval_labels_bin, eval_labels_bin)
        else:
            best_metric = metric_fn(eval_labels_bin, eval_labels)
        best_metric_results[metric_name] = (best_metric_results[metric_name], best_metric)

    return best_metric_results, best_scores, label_stack_eval, float(best_threshold), best_feature
