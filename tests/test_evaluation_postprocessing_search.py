from __future__ import annotations

import numpy as np

from noboom_benchmark.noboom_lib.core.metric_utils import evaluation_postprocessing_search as search_module


def test_split_flat_score_payload_uses_finite_score_segments() -> None:
    scores = np.array([-np.inf, 0.1, 0.2, -np.inf, 0.3], dtype=np.float64)
    labels = np.array([0, 1, 0, 0, 1], dtype=np.int64)

    score_sequences, label_sequences = search_module.split_flat_score_payload(scores, labels)

    assert len(score_sequences) == 2
    np.testing.assert_allclose(score_sequences[0], np.array([0.1, 0.2], dtype=np.float64))
    np.testing.assert_allclose(score_sequences[1], np.array([0.3], dtype=np.float64))
    np.testing.assert_array_equal(label_sequences[0], np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(label_sequences[1], np.array([1], dtype=np.int64))


def test_search_evaluation_postprocessing_prefers_better_candidate(monkeypatch) -> None:
    seen_configs = []

    def fake_evaluate_sequences_with_optional_postprocessing(
        score_sequences,
        label_sequences,
        *,
        metric_names,
        config,
        fix_threshold=False,
        dataset_name=None,
    ):
        del score_sequences
        del label_sequences
        del metric_names
        del fix_threshold
        del dataset_name
        seen_configs.append(dict(config.__dict__))
        short_window = int(config.short_window)
        alarm_score = 1.0 if config.enabled and short_window == 7 else float(short_window) / 10.0
        return type(
            "Result",
            (),
            {
                "metric_values": {
                    "alarm_score": alarm_score,
                    "aaf": 0.5,
                    "event_recall": 0.5,
                    "edf": 1.0,
                    "ldf": 1.0,
                },
                "threshold": 0.5,
                "threshold_on": 0.5,
                "threshold_off": 0.4,
                "robust_scale": 0.1,
            },
        )()

    monkeypatch.setattr(
        search_module,
        "evaluate_sequences_with_optional_postprocessing",
        fake_evaluate_sequences_with_optional_postprocessing,
    )

    result = search_module.search_evaluation_postprocessing(
        score_sequences=[np.array([0.1, 0.2], dtype=np.float64)],
        label_sequences=[np.array([0, 1], dtype=np.int64)],
        metric_names=("alarm_score", "aaf", "event_recall", "edf", "ldf"),
        target_metric="alarm_score",
        dataset_name="industry_process",
        initial_config={"enabled": False, "short_window": 5},
        search_space={
            "short_window": (5, 7, 9),
        },
        search_order=("short_window",),
        max_rounds=2,
        fix_threshold=False,
    )

    assert result.baseline_trial.config["enabled"] is False
    assert result.default_enabled_trial.config["enabled"] is True
    assert result.best_trial.config["short_window"] == 7
    assert result.best_trial.metric_values["alarm_score"] == 1.0
    assert any(config["enabled"] is False for config in seen_configs)
