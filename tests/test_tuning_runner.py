from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import textwrap
from typing import Optional

import pytest

from noboom_benchmark.noboom_lib.core import tuning_runner


CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
)


def test_load_study_params_reads_evaluation_postprocessing_from_model_common(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "params").mkdir(parents=True, exist_ok=True)
    (config_dir / "models").mkdir(parents=True, exist_ok=True)

    (config_dir / "params" / "common.yaml").write_text(
        textwrap.dedent(
            """
            study:
              metrics: ["alarm_score"]
              seeds: [42]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "params" / "neutralad.yaml").write_text(
        textwrap.dedent(
            """
            search_space:
              window_size: 128
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "models" / "common.yaml").write_text(
        textwrap.dedent(
            """
            model:
              evaluation_postprocessing:
                enabled: true
                short_window: 9
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "models" / "neutralad.yaml").write_text("model: {}\n", encoding="utf-8")

    study_params, hp_params = tuning_runner._load_study_params(  # noqa: SLF001
        Namespace(config_dir=str(config_dir)),
        "neutralad",
    )

    assert study_params["evaluation_postprocessing"] == {
        "enabled": True,
        "short_window": 9,
    }
    assert hp_params == {"window_size": 128}


def test_load_study_params_merges_dataset_search_space_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "params").mkdir(parents=True, exist_ok=True)
    (config_dir / "models").mkdir(parents=True, exist_ok=True)

    (config_dir / "params" / "common.yaml").write_text(
        textwrap.dedent(
            """
            study:
              metrics: ["alarm_score"]
              seeds: [42]
            dataset_search_spaces:
              batch_dist_ternary_acetone_1_butanol_methanol:
                data:
                  scaler:
                    init_args:
                      drop_features:
                        distribution: categorical
                        choices:
                          - []
                          - ["LS701", "LS702"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "params" / "neutralad.yaml").write_text(
        textwrap.dedent(
            """
            search_space:
              data:
                scaler:
                  init_args:
                    enable_asinh:
                      distribution: categorical
                      choices: [true, false]
              window_size: 128
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "models" / "common.yaml").write_text("model: {}\n", encoding="utf-8")
    (config_dir / "models" / "neutralad.yaml").write_text("model: {}\n", encoding="utf-8")

    _study_params, hp_params = tuning_runner._load_study_params(  # noqa: SLF001
        Namespace(
            config_dir=str(config_dir),
            dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        ),
        "neutralad",
    )

    assert hp_params["data"]["scaler"]["init_args"]["enable_asinh"]["choices"] == [True, False]
    assert hp_params["data"]["scaler"]["init_args"]["drop_features"]["choices"] == [
        [],
        ["LS701", "LS702"],
    ]


def test_max_concurrent_trials_serializes_full_gpu_exclusive_trials() -> None:
    assert (
        tuning_runner._max_concurrent_trials(  # noqa: SLF001
            250,
            {"CPU": 4, "GPU": 1.0, "exclusive": 1.0},
        )
        == 1
    )


def test_max_concurrent_trials_keeps_existing_fractional_policy() -> None:
    assert (
        tuning_runner._max_concurrent_trials(  # noqa: SLF001
            8,
            {"CPU": 4, "GPU": 0.5, "exclusive": 0.001},
        )
        == 4
    )


def test_execute_tuning_workflow_passes_real_trial_resources_to_with_resources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}
    sentinel_resources = {"CPU": 3, "GPU": 0.5, "exclusive": 0.001}
    experiment_path = tmp_path / "experiment"
    experiment_path.mkdir()

    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            return SimpleNamespace(info=SimpleNamespace(run_id="study-run-id", run_name=run_name))

        def download_artifacts(self, _run_id: str, _artifact_path: str, _dst_path: str) -> None:
            return None

        def log_artifacts(self, _run_id: str, _path: str) -> None:
            return None

        def log_dict(self, _run_id: str, _payload, artifact_file: str) -> None:
            captured.setdefault("artifact_files", []).append(artifact_file)

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config) -> None:
            captured["trainable"] = trainable
            captured["param_space"] = param_space
            captured["tune_config"] = tune_config
            captured["run_config"] = run_config

        def fit(self):
            return SimpleNamespace(
                get_best_result=lambda: SimpleNamespace(metrics={"alarm_score": 0.8, "mlflow_run_id": "trial-run-id"}),
                experiment_path=str(experiment_path),
            )

    def fake_with_resources(trainable, resources):
        captured["resources"] = resources
        return ("wrapped-trainable", trainable)

    monkeypatch.setattr(tuning_runner, "_load_study_params", lambda args, model_name: (
        {
            "tune_metric": "alarm_score",
            "direction": "maximize",
            "n_trials": 4,
            "scheduler": {"name": "ASHA", "args": {"max_t": 6, "grace_period": 2}},
        },
        {"lr": 1e-3},
    ))
    monkeypatch.setattr(tuning_runner, "build_tune_search_space", lambda hp_params: ({"lr": [1e-3]}, 8))
    monkeypatch.setattr(tuning_runner, "build_sampler", lambda cfg: "sampler")
    monkeypatch.setattr(tuning_runner, "build_tune_scheduler", lambda cfg: ("scheduler", cfg))
    monkeypatch.setattr(tuning_runner, "OptunaSearch", lambda **kwargs: ("optuna-search", kwargs))
    monkeypatch.setattr(tuning_runner, "MLFlowLoggerCallbackSW", lambda **kwargs: ("mlflow-callback", kwargs))
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    def fake_tune_resource_request_dict(args):
        captured["resource_study_params"] = args.study_params
        return sentinel_resources

    monkeypatch.setattr(tuning_runner, "tune_resource_request_dict", fake_tune_resource_request_dict)
    monkeypatch.setattr(tuning_runner.tune, "with_resources", fake_with_resources)
    monkeypatch.setattr(tuning_runner.tune, "Tuner", FakeTuner)

    args = Namespace(
        optuna_storage_uri="sqlite:///optuna.db",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        experiment_id="exp-1",
        tracking_uri="file:///tmp/mlruns",
        storage_path=str(tmp_path / "storage"),
        temp_dir=str(tmp_path),
        config_dir=str(CONFIG_DIR),
        mlflow_enable_logged_models=False,
        mlflow_enable_evaluation=False,
        mlflow_enable_tables=False,
    )

    tuning_runner.execute_tuning_workflow(args)

    assert captured["resources"] == sentinel_resources
    assert captured["resource_study_params"]["tune_metric"] == "alarm_score"
    assert captured["tune_config"].scheduler == ("scheduler", {"name": "ASHA", "args": {"max_t": 6, "grace_period": 2}})


def test_optuna_storage_from_uri_supports_memory_and_sqlite(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    monkeypatch.setattr(
        tuning_runner.optuna.storages,
        "RDBStorage",
        lambda url: captured.update({"url": url}) or ("storage", url),
    )

    assert tuning_runner._optuna_storage_from_uri("memory://") is None  # noqa: SLF001

    storage = tuning_runner._optuna_storage_from_uri(f"sqlite:///{tmp_path / 'optuna.db'}")  # noqa: SLF001

    assert storage == ("storage", f"sqlite:///{tmp_path / 'optuna.db'}")
    assert captured["url"] == f"sqlite:///{tmp_path / 'optuna.db'}"


def test_execute_local_tuning_workflow_uses_optuna_memory_and_local_trial_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {
        "trial_hparams": [],
        "snapshots": [],
    }

    class FakeClient:
        def __init__(self) -> None:
            self._run_index = 0

        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            self._run_index += 1
            return SimpleNamespace(
                info=SimpleNamespace(
                    run_id=f"run-{self._run_index}",
                    run_name=run_name,
                )
            )

        def set_terminated(self, run_id: str) -> None:
            captured.setdefault("terminated", []).append(run_id)

    def fake_run_local_hpo_trial(*, args, client, hp_params, trial_name, trial_root_dir):
        del args, client, trial_root_dir
        captured["trial_hparams"].append((trial_name, dict(hp_params)))
        score = 0.4 if hp_params["window_size"] == 32 else 0.8
        return {"alarm_score": score, "mlflow_run_id": f"{trial_name}-run"}

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: (
            {
                "tune_metric": "alarm_score",
                "direction": "maximize",
                "n_trials": 2,
                "metrics": ["alarm_score"],
                "seeds": [42],
                "sampler": {"name": "RandomSampler", "args": {"seed": 11}},
            },
            {"window_size": {"distribution": "categorical", "choices": [32, 64]}},
        ),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "_run_local_hpo_trial", fake_run_local_hpo_trial)
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda client, *, args, result, resolved_hparams, status, partial_result, result_source, write_mlflow: captured[
            "snapshots"
        ].append(
            {
                "result": dict(result),
                "resolved_hparams": dict(resolved_hparams),
                "status": status,
                "partial_result": partial_result,
                "result_source": result_source,
            }
        ),
    )
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: {
            **dict(result),
            "hp_params": dict(resolved_hparams),
        },
    )

    args = Namespace(
        optuna_storage_uri="memory://",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        experiment_id="exp-1",
        tracking_uri="file:///tmp/mlruns",
        storage_path=str(tmp_path / "storage" / "local" / "cont_reactive_ome__neutralad"),
        temp_dir=str(tmp_path),
        config_dir=str(CONFIG_DIR),
        execution_backend="local",
        deployment_mode="local",
        mlflow_controller_run_id=None,
        mlflow_enable_logged_models=False,
        mlflow_enable_evaluation=False,
        mlflow_enable_tables=False,
        tune=True,
        hpo_seeds=None,
    )

    _model_name, _dataset_name, result = tuning_runner.execute_local_tuning_workflow(args)

    assert len(captured["trial_hparams"]) == 2
    assert captured["snapshots"]
    assert captured["snapshots"][-1]["result_source"] == "incremental_best_trial"
    assert result["alarm_score"] in {0.4, 0.8}
    assert result["hp_params"]["window_size"] in {32, 64}


def test_execute_tuning_workflow_runs_enabled_final_full_seed_eval_after_reduced_hpo(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {
        "seed_calls": [],
        "snapshots": [],
        "saved_metadata": [],
    }
    experiment_path = tmp_path / "ray_experiment"
    experiment_path.mkdir()
    best_config = {"window_size": 64, "lr": 2e-3}

    study_params = {
        "tune_metric": "alarm_score",
        "direction": "maximize",
        "n_trials": 1,
        "metrics": ["alarm_score"],
        "seeds": [42, 43, 44],
        "hpo_seeds": [42],
        "final_eval_after_hpo": True,
    }

    class FakeClient:
        def __init__(self) -> None:
            self._run_index = 0

        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            self._run_index += 1
            if run_name.endswith("__single_train"):
                run_id = "final-trial-run-id"
            elif self._run_index == 1:
                run_id = "hpo-study-run-id"
            else:
                run_id = f"final-study-run-id-{self._run_index}"
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config) -> None:
            del trainable, param_space, tune_config, run_config

        def fit(self):
            return SimpleNamespace(
                get_best_result=lambda: SimpleNamespace(
                    metrics={"alarm_score": 0.7, "mlflow_run_id": "hpo-trial-run-id"},
                    config=best_config,
                ),
                experiment_path=str(experiment_path),
            )

    class FakeRunSeedTask:
        def options(self, **kwargs):
            del kwargs
            return self

        def remote(
            self,
            resolved_hparams,
            args,
            seed,
            trial_root_dir,
            trial_run_id,
            trial_name,
        ):
            del trial_root_dir, trial_run_id, trial_name
            captured["seed_calls"].append(
                {
                    "resolved_hparams": dict(resolved_hparams),
                    "seed": seed,
                    "tune": getattr(args, "tune", None),
                    "study_params": dict(args.study_params),
                }
            )
            return {"alarm_score": (0.8 + (seed - 42) * 0.01, 0.95)}

    def fake_persist_pair_snapshot(
        client,
        *,
        args,
        result,
        resolved_hparams,
        status,
        partial_result,
        result_source,
        write_mlflow,
    ):
        del client, write_mlflow
        payload = dict(result)
        if getattr(args, "tune", False):
            payload = tuning_runner.build_reduced_hpo_seed_context(args.study_params) | payload
        elif isinstance(getattr(args, "result_seed_context", None), dict):
            payload = dict(args.result_seed_context) | payload
        else:
            payload.setdefault("seed_evaluation_scope", "final_eval")
            payload.setdefault("uses_reduced_hpo_seeds", False)
        if resolved_hparams is not None:
            payload["hp_params"] = dict(resolved_hparams)
        captured["snapshots"].append(
            {
                "payload": payload,
                "status": status,
                "partial_result": partial_result,
                "result_source": result_source,
                "tune": getattr(args, "tune", None),
            }
        )
        return payload

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: (dict(study_params), {"window_size": 128, "lr": 1e-3}),
    )
    monkeypatch.setattr(tuning_runner.optuna.storages, "RDBStorage", lambda url: ("storage", url))
    monkeypatch.setattr(tuning_runner, "build_tune_search_space", lambda hp_params: ({"lr": [1e-3]}, 1))
    monkeypatch.setattr(tuning_runner, "build_sampler", lambda cfg: "sampler")
    monkeypatch.setattr(tuning_runner, "build_tune_scheduler", lambda cfg: ("scheduler", cfg))
    monkeypatch.setattr(tuning_runner, "OptunaSearch", lambda **kwargs: ("optuna-search", kwargs))
    monkeypatch.setattr(tuning_runner, "MLFlowLoggerCallbackSW", lambda **kwargs: ("mlflow-callback", kwargs))
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(
        tuning_runner,
        "tune_resource_request_dict",
        lambda args: {"CPU": 1.0, "GPU": 0.5, "memory": 16 * 1024**3},
    )
    monkeypatch.setattr(tuning_runner.tune, "with_resources", lambda trainable, resources: (trainable, resources))
    monkeypatch.setattr(tuning_runner.tune, "Tuner", FakeTuner)
    monkeypatch.setattr(tuning_runner, "run_seed_task", FakeRunSeedTask())
    monkeypatch.setattr(tuning_runner, "NodeAffinitySchedulingStrategy", lambda node_id, soft: (node_id, soft))
    monkeypatch.setattr(
        tuning_runner.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-id"),
    )
    monkeypatch.setattr(tuning_runner.ray, "get", lambda future: future)
    monkeypatch.setattr(
        tuning_runner.ray,
        "wait",
        lambda futures, num_returns=1: (futures[:num_returns], futures[num_returns:]),
    )
    monkeypatch.setattr(tuning_runner.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tuning_runner,
        "save_metadata",
        lambda args, artifact_dir, hp_params: captured["saved_metadata"].append(
            {
                "tune": getattr(args, "tune", None),
                "study_params": dict(args.study_params),
                "hp_params": dict(hp_params),
            }
        ),
    )
    monkeypatch.setattr(tuning_runner, "_persist_pair_snapshot", fake_persist_pair_snapshot)

    args = Namespace(
        optuna_storage_uri="sqlite:///optuna.db",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        experiment_id="exp-1",
        tracking_uri="file:///tmp/mlruns",
        storage_path=str(tmp_path / "storage"),
        temp_dir=str(tmp_path),
        config_dir=str(CONFIG_DIR),
        mlflow_enable_logged_models=False,
        mlflow_enable_evaluation=False,
        mlflow_enable_tables=False,
        tune=True,
        hpo_seeds=None,
    )

    _model_name, _dataset_name, result = tuning_runner.execute_tuning_workflow(args)

    assert [call["seed"] for call in captured["seed_calls"]] == [42, 43, 44]
    assert all(call["resolved_hparams"] == best_config for call in captured["seed_calls"])
    assert all(call["tune"] is False for call in captured["seed_calls"])
    assert captured["saved_metadata"] == [
        {
            "tune": False,
            "study_params": study_params,
            "hp_params": best_config,
        }
    ]
    assert captured["snapshots"][0]["payload"]["seed_evaluation_scope"] == "hpo"
    assert captured["snapshots"][0]["payload"]["uses_reduced_hpo_seeds"] is True
    assert captured["snapshots"][-1]["payload"]["seed_evaluation_scope"] == "final_eval"
    assert captured["snapshots"][-1]["payload"]["final_eval_seeds"] == [42, 43, 44]
    assert captured["snapshots"][-1]["result_source"] == "final_full_seed_eval_after_hpo"
    assert result["seed_evaluation_scope"] == "final_eval"
    assert result["completed_seed_count"] == 3
    assert result["hp_params"] == best_config


@pytest.mark.parametrize(
    ("study_params", "expected_alarm_score"),
    [
        (
            {
                "tune_metric": "alarm_score",
                "direction": "maximize",
                "n_trials": 1,
                "metrics": ["alarm_score"],
                "seeds": [42, 43],
                "hpo_seeds": [42],
            },
            0.7,
        ),
        (
            {
                "tune_metric": "alarm_score",
                "direction": "maximize",
                "n_trials": 1,
                "metrics": ["alarm_score"],
                "seeds": [42, 43],
                "final_eval_after_hpo": True,
            },
            0.7,
        ),
        (
            {
                "tune_metric": "alarm_score",
                "direction": "maximize",
                "n_trials": 1,
                "metrics": ["alarm_score"],
                "seeds": [42, 43],
                "hpo_seeds": [42, 43],
                "final_eval_after_hpo": True,
            },
            0.7,
        ),
    ],
)
def test_execute_tuning_workflow_skips_final_full_seed_eval_without_enabled_reduced_hpo(
    monkeypatch,
    tmp_path: Path,
    study_params: dict,
    expected_alarm_score: float,
) -> None:
    experiment_path = tmp_path / "ray_experiment"
    experiment_path.mkdir()

    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            return SimpleNamespace(info=SimpleNamespace(run_id="hpo-study-run-id", run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config) -> None:
            del trainable, param_space, tune_config, run_config

        def fit(self):
            return SimpleNamespace(
                get_best_result=lambda: SimpleNamespace(
                    metrics={"alarm_score": expected_alarm_score, "mlflow_run_id": "hpo-trial-run-id"},
                    config={"window_size": 64},
                ),
                experiment_path=str(experiment_path),
            )

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: (dict(study_params), {"window_size": 128}),
    )
    monkeypatch.setattr(tuning_runner.optuna.storages, "RDBStorage", lambda url: ("storage", url))
    monkeypatch.setattr(tuning_runner, "build_tune_search_space", lambda hp_params: ({"window_size": [64]}, 1))
    monkeypatch.setattr(tuning_runner, "build_sampler", lambda cfg: "sampler")
    monkeypatch.setattr(tuning_runner, "build_tune_scheduler", lambda cfg: ("scheduler", cfg))
    monkeypatch.setattr(tuning_runner, "OptunaSearch", lambda **kwargs: ("optuna-search", kwargs))
    monkeypatch.setattr(tuning_runner, "MLFlowLoggerCallbackSW", lambda **kwargs: ("mlflow-callback", kwargs))
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "tune_resource_request_dict", lambda args: {"CPU": 1.0})
    monkeypatch.setattr(tuning_runner.tune, "with_resources", lambda trainable, resources: (trainable, resources))
    monkeypatch.setattr(tuning_runner.tune, "Tuner", FakeTuner)
    monkeypatch.setattr(
        tuning_runner,
        "execute_single_train_workflow",
        lambda *args, **kwargs: pytest.fail("final full-seed eval should not run"),
    )
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: dict(result),
    )

    args = Namespace(
        optuna_storage_uri="sqlite:///optuna.db",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        experiment_id="exp-1",
        tracking_uri="file:///tmp/mlruns",
        storage_path=str(tmp_path / "storage"),
        temp_dir=str(tmp_path),
        config_dir=str(CONFIG_DIR),
        mlflow_enable_logged_models=False,
        mlflow_enable_evaluation=False,
        mlflow_enable_tables=False,
        tune=True,
        hpo_seeds=None,
    )

    _model_name, _dataset_name, result = tuning_runner.execute_tuning_workflow(args)

    assert result["alarm_score"] == expected_alarm_score


def test_execute_tuning_workflow_does_not_run_final_full_seed_eval_when_hpo_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            return SimpleNamespace(info=SimpleNamespace(run_id="hpo-study-run-id", run_name=run_name))

    class FailingTuner:
        def __init__(self, trainable, param_space, tune_config, run_config) -> None:
            del trainable, param_space, tune_config, run_config

        def fit(self):
            raise RuntimeError("hpo failed")

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: (
            {
                "tune_metric": "alarm_score",
                "direction": "maximize",
                "n_trials": 1,
                "metrics": ["alarm_score"],
                "seeds": [42, 43],
                "hpo_seeds": [42],
                "final_eval_after_hpo": True,
            },
            {"window_size": 128},
        ),
    )
    monkeypatch.setattr(tuning_runner.optuna.storages, "RDBStorage", lambda url: ("storage", url))
    monkeypatch.setattr(tuning_runner, "build_tune_search_space", lambda hp_params: ({"window_size": [64]}, 1))
    monkeypatch.setattr(tuning_runner, "build_sampler", lambda cfg: "sampler")
    monkeypatch.setattr(tuning_runner, "build_tune_scheduler", lambda cfg: ("scheduler", cfg))
    monkeypatch.setattr(tuning_runner, "OptunaSearch", lambda **kwargs: ("optuna-search", kwargs))
    monkeypatch.setattr(tuning_runner, "MLFlowLoggerCallbackSW", lambda **kwargs: ("mlflow-callback", kwargs))
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "tune_resource_request_dict", lambda args: {"CPU": 1.0})
    monkeypatch.setattr(tuning_runner.tune, "with_resources", lambda trainable, resources: (trainable, resources))
    monkeypatch.setattr(tuning_runner.tune, "Tuner", FailingTuner)
    monkeypatch.setattr(
        tuning_runner,
        "execute_single_train_workflow",
        lambda *args, **kwargs: pytest.fail("final full-seed eval should not run after failed HPO"),
    )

    args = Namespace(
        optuna_storage_uri="sqlite:///optuna.db",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        experiment_id="exp-1",
        tracking_uri="file:///tmp/mlruns",
        storage_path=str(tmp_path / "storage"),
        temp_dir=str(tmp_path),
        config_dir=str(CONFIG_DIR),
        mlflow_enable_logged_models=False,
        mlflow_enable_evaluation=False,
        mlflow_enable_tables=False,
        tune=True,
        hpo_seeds=None,
    )

    with pytest.raises(RuntimeError, match="hpo failed"):
        tuning_runner.execute_tuning_workflow(args)


def test_persist_pair_snapshot_distinguishes_hpo_and_final_full_seed_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logged_artifacts = []

    class FakeClient:
        def log_dict(self, run_id: str, payload, artifact_file: str) -> None:
            logged_artifacts.append((run_id, artifact_file, dict(payload)))

    monkeypatch.setattr(tuning_runner, "update_score_excel_one", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_runner, "excel_score_table_has_metrics", lambda *args, **kwargs: False)

    study_params = {
        "tune_metric": "alarm_score",
        "direction": "maximize",
        "metrics": ["alarm_score"],
        "seeds": [42, 43],
        "hpo_seeds": [42],
    }
    storage_path = str(tmp_path / "storage" / "ray" / "cont_reactive_ome__neutralad")
    base_args = {
        "model_name": "neutralad",
        "dataset_name": "cont_reactive_ome",
        "requested_dataset_name": "cont_reactive_ome",
        "storage_path": storage_path,
        "study_run_id": "study-run-id",
        "study_params": study_params,
    }

    hpo_payload = tuning_runner._persist_pair_snapshot(  # noqa: SLF001
        FakeClient(),
        args=Namespace(**base_args, tune=True),
        result={"alarm_score": 0.7, "completed_seed_count": 1},
        resolved_hparams={"window_size": 64},
        status="SUCCEEDED",
        partial_result=False,
        result_source="hpo_final",
        write_mlflow=True,
    )
    final_payload = tuning_runner._persist_pair_snapshot(  # noqa: SLF001
        FakeClient(),
        args=Namespace(**base_args, tune=False),
        result={
            "alarm_score": 0.81,
            "completed_seed_count": 2,
            "seed_evaluation_scope": "final_eval",
        },
        resolved_hparams={"window_size": 64},
        status="SUCCEEDED",
        partial_result=False,
        result_source="final_full_seed_eval_after_hpo",
        write_mlflow=True,
    )

    assert hpo_payload["seed_evaluation_scope"] == "hpo"
    assert hpo_payload["uses_reduced_hpo_seeds"] is True
    assert hpo_payload["hpo_seeds"] == [42]
    assert hpo_payload["full_seeds"] == [42, 43]
    assert final_payload["seed_evaluation_scope"] == "final_eval"
    assert final_payload["uses_reduced_hpo_seeds"] is True
    assert final_payload["hpo_seeds"] == [42]
    assert final_payload["final_eval_seeds"] == [42, 43]
    assert final_payload["result_source"] == "final_full_seed_eval_after_hpo"
    assert [entry[1] for entry in logged_artifacts] == [
        "summary/result.json",
        "summary/result.json",
    ]


def test_finalize_study_outputs_skips_logged_models_and_logs_tables(
    monkeypatch,
) -> None:
    captured = {
        "tags": [],
        "artifact_files": [],
        "table_files": [],
    }

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            captured["tags"].append((run_id, key, value))

        def log_dict(self, run_id: str, payload, artifact_file: str) -> None:
            captured["artifact_files"].append((run_id, artifact_file, payload))
    monkeypatch.setattr(
        tuning_runner,
        "seed_rows_for_trial",
        lambda *args, **kwargs: [{"seed": 42, "alarm_score": 0.9}],
    )
    monkeypatch.setattr(
        tuning_runner,
        "trial_rows_for_study",
        lambda *args, **kwargs: [{"run_id": "trial-run-id", "alarm_score": 0.9}],
    )
    monkeypatch.setattr(
        tuning_runner,
        "log_table_artifact",
        lambda client, run_id, *, artifact_file, rows: captured["table_files"].append(
            (run_id, artifact_file, list(rows))
        ),
    )
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda client, *, args, result, resolved_hparams, status, partial_result, result_source, write_mlflow: captured[
            "artifact_files"
        ].append(
            (
                args.study_run_id,
                "summary/result.json",
                {
                    **dict(result),
                    "hp_params": {"window_size": 108, "model": {"dropout": 0.1}},
                },
            )
        )
        or {
            **dict(result),
            "hp_params": {"window_size": 108, "model": {"dropout": 0.1}},
        },
    )

    args = Namespace(
        experiment_id="exp-1",
        study_run_id="study-run-id",
        mlflow_enable_logged_models=True,
        mlflow_enable_evaluation=True,
        mlflow_enable_tables=True,
    )

    result = tuning_runner._finalize_study_outputs(  # noqa: SLF001
        FakeClient(),
        args=args,
        trial_run_id="trial-run-id",
        trial_artifact_dir=None,
        resolved_hparams={"window_size": 108, "model": {"dropout": 0.1}},
        result={"alarm_score": 0.9, "mlflow_run_id": "trial-run-id"},
    )

    assert result == {
        "alarm_score": 0.9,
        "mlflow_run_id": "trial-run-id",
        "hp_params": {"window_size": 108, "model": {"dropout": 0.1}},
    }
    assert captured["tags"] == []
    assert captured["artifact_files"][0][1] == "summary/result.json"
    assert captured["artifact_files"][0][2]["hp_params"] == {
        "window_size": 108,
        "model": {"dropout": 0.1},
    }
    assert [artifact_file for _run_id, artifact_file, _rows in captured["table_files"]] == [
        "tables/seeds.json",
        "tables/trials.json",
    ]


def test_best_trial_snapshot_callback_persists_only_on_improvement(monkeypatch) -> None:
    captured = []
    callback = tuning_runner.BestTrialSnapshotCallback(
        args=Namespace(
            study_params={"tune_metric": "alarm_score", "metrics": ["alarm_score"]},
            model_name="neutralad",
            dataset_name="cont_reactive_ome",
            requested_dataset_name="cont_reactive_ome",
            storage_path="/tmp/storage/ray/cont_reactive_ome__neutralad",
            study_run_id="study-run-id",
        ),
        client=SimpleNamespace(),
        mode="max",
    )

    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda client, *, args, result, resolved_hparams, status, partial_result, result_source, write_mlflow: captured.append(
            {
                "result": dict(result),
                "resolved_hparams": dict(resolved_hparams) if resolved_hparams is not None else None,
                "status": status,
                "partial_result": partial_result,
                "result_source": result_source,
                "write_mlflow": write_mlflow,
            }
        ),
    )

    trial = SimpleNamespace(config={"window_size": 64})
    callback.on_trial_result(0, [], trial, {"alarm_score": 0.8, "mlflow_run_id": "trial-1"})
    callback.on_trial_result(1, [], trial, {"alarm_score": 0.7, "mlflow_run_id": "trial-2"})
    callback.on_trial_result(2, [], trial, {"alarm_score": 0.85, "mlflow_run_id": "trial-3"})

    assert [entry["result"]["mlflow_run_id"] for entry in captured] == ["trial-1", "trial-3"]
    assert all(entry["status"] == "RUNNING" for entry in captured)
    assert all(entry["partial_result"] is True for entry in captured)
    assert all(entry["result_source"] == "incremental_best_trial" for entry in captured)


def test_run_tune_or_train_defers_manifest_preparation_for_tune_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    runtime_config = SimpleNamespace(
        validate=lambda: None,
        mlflow_tracking_uri="file:///tmp/mlruns",
        deployment_mode="docker",
        mlflow_enable_controller_lineage=True,
        mlflow_enable_dataset_tracking=True,
        mlflow_enable_logged_models=True,
        mlflow_enable_evaluation=True,
        mlflow_enable_tables=True,
        mlflow_enable_system_metrics=True,
        mlflow_enable_registry=False,
        mlflow_controller_run_id="controller-run-id",
    )

    def fake_prepare_model_dataset_manifests(**kwargs):
        raise AssertionError(f"tune dispatch should not eagerly prepare manifests: {kwargs}")

    def fake_execute_tuning_workflow(args: Namespace):
        captured["prepared_manifest_paths"] = getattr(args, "prepared_manifest_paths", None)
        captured["data_manifest_path"] = getattr(args, "data_manifest_path", None)
        captured["dataset_name"] = args.dataset_name
        captured["requested_dataset_name"] = args.requested_dataset_name
        captured["config_dir"] = args.config_dir
        return args.model_name, args.dataset_name, {"alarm_score": 0.9}

    monkeypatch.setattr(
        tuning_runner.RuntimeConfig,
        "from_env_and_args",
        classmethod(lambda cls, args: runtime_config),
    )
    monkeypatch.setattr(
        tuning_runner,
        "prepare_model_dataset_manifests",
        fake_prepare_model_dataset_manifests,
    )
    monkeypatch.setattr(tuning_runner, "execute_tuning_workflow", fake_execute_tuning_workflow)
    monkeypatch.setattr(tuning_runner.ray, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_runner.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(tuning_runner.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tuning_runner.mlflow, "set_experiment", lambda **_kwargs: None)

    args = Namespace(
        dataset_name="cont_reactive_ome_tsst",
        model_name="neutralad",
        config_dir=str(CONFIG_DIR),
        temp_dir=str(tmp_path),
        verbose=0,
        tune=True,
        experiment_id="exp-1",
    )

    result = tuning_runner.run_tune_or_train(args)

    assert result == ("neutralad", "cont_reactive_ome", {"alarm_score": 0.9})
    assert captured["prepared_manifest_paths"] is None
    assert captured["data_manifest_path"] is None
    assert captured["dataset_name"] == "cont_reactive_ome"
    assert captured["requested_dataset_name"] == "cont_reactive_ome_tsst"


def test_execute_single_train_workflow_preserves_resolved_window_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            run_id = "study-run-id" if run_name == "neutralad__cont_reactive_ome__20260311_000000" else "trial-run-id"
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeRunSeedTask:
        def options(self, **kwargs):
            captured["options"] = kwargs
            return self

        def remote(
            self,
            resolved_hparams,
            args,
            seed,
            trial_root_dir,
            trial_run_id,
            trial_name,
        ):
            del args, trial_root_dir, trial_run_id, trial_name
            captured["resolved_hparams"] = dict(resolved_hparams)
            captured.setdefault("seeds", []).append(seed)
            return {"alarm_score": (0.75, 0.9)}

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: (
            {"seeds": [42, 43], "hpo_seeds": [42]},
            {"window_size": 128, "lr": 1e-3},
        ),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "run_seed_task", FakeRunSeedTask())
    monkeypatch.setattr(tuning_runner, "NodeAffinitySchedulingStrategy", lambda node_id, soft: (node_id, soft))
    monkeypatch.setattr(
        tuning_runner.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-id"),
    )
    monkeypatch.setattr(tuning_runner.ray, "get", lambda future: future)
    monkeypatch.setattr(
        tuning_runner.ray,
        "wait",
        lambda futures, num_returns=1: (futures[:num_returns], futures[num_returns:]),
    )
    monkeypatch.setattr(
        tuning_runner,
        "tune_resource_request_dict",
        lambda args: {"CPU": 1.0, "GPU": 0.5, "memory": 16 * 1024**3},
    )
    monkeypatch.setattr(tuning_runner.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tuning_runner,
        "save_metadata",
        lambda args, trial_root_dir, resolved_hparams: captured.setdefault(
            "saved_hparams",
            dict(resolved_hparams),
        ),
    )
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: result,
    )
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda *args, **kwargs: captured.setdefault("snapshots", []).append(dict(kwargs["result"])),
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        source_experiment_id=None,
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        storage_path=str(tmp_path / "storage"),
        config_dir=str(CONFIG_DIR),
    )

    tuning_runner.execute_single_train_workflow(
        args,
        hparams_payload={"config": {"window_size": 58, "lr": 1e-3}},
    )

    assert captured["resolved_hparams"] == {"window_size": 58, "lr": 1e-3}
    assert captured["saved_hparams"] == {"window_size": 58, "lr": 1e-3}
    assert captured["seeds"] == [42, 43]
    assert captured["options"]["memory"] == 16 * 1024**3
    assert "memory" not in captured["options"]["resources"]
    assert [snapshot["completed_seed_count"] for snapshot in captured["snapshots"]] == [1, 2]


def test_execute_single_train_workflow_local_backend_runs_seed_loop_without_ray(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {
        "seeds": [],
        "snapshots": [],
    }

    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            run_id = "study-run-id" if run_name == "neutralad__cont_reactive_ome__20260311_000000" else "trial-run-id"
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    def fake_run_seed_task_local(
        resolved_hparams,
        args,
        seed,
        trial_root_dir,
        trial_run_id,
        trial_name,
    ):
        del resolved_hparams, args, trial_root_dir, trial_run_id, trial_name
        captured["seeds"].append(seed)
        return {"alarm_score": (0.75 if seed == 42 else 0.85, 0.95)}

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: ({"seeds": [42, 43], "metrics": ["alarm_score"]}, {"window_size": 128}),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "run_seed_task_local", fake_run_seed_task_local)
    monkeypatch.setattr(
        tuning_runner.ray,
        "get_runtime_context",
        lambda: pytest.fail("ray runtime context should not be used in local mode"),
    )
    monkeypatch.setattr(tuning_runner.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_runner, "save_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: result,
    )
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda *args, **kwargs: captured["snapshots"].append(dict(kwargs["result"])),
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        source_experiment_id=None,
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        storage_path=str(tmp_path / "storage" / "local" / "cont_reactive_ome__neutralad"),
        config_dir=str(CONFIG_DIR),
        execution_backend="local",
        deployment_mode="local",
    )

    _model_name, _dataset_name, result = tuning_runner.execute_single_train_workflow(
        args,
        hparams_payload={"config": {"window_size": 58}},
    )

    assert captured["seeds"] == [42, 43]
    assert [snapshot["completed_seed_count"] for snapshot in captured["snapshots"]] == [1, 2]
    assert result["alarm_score"] == pytest.approx(0.8)


def test_execute_single_train_workflow_uses_source_experiment_lookup_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            run_id = "study-run-id" if run_name == "neutralad__cont_reactive_ome__20260311_000000" else "trial-run-id"
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeRunSeedTask:
        def options(self, **kwargs):
            del kwargs
            return self

        def remote(
            self,
            resolved_hparams,
            args,
            seed,
            trial_root_dir,
            trial_run_id,
            trial_name,
        ):
            del args, seed, trial_root_dir, trial_run_id, trial_name
            captured["resolved_hparams"] = dict(resolved_hparams)
            return {"alarm_score": (0.75, 0.9)}

    def fake_load_study_result_from_mlflow(
        experiment_id: str,
        model_name: str,
        dataset_name: str,
        artifact_path: str = "summary/result.json",
        tracking_uri: Optional[str] = None,
        min_completed_seeds: int = 2,
    ):
        del artifact_path, tracking_uri
        captured["source_lookup"] = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "min_completed_seeds": min_completed_seeds,
        }
        return {"hp_params": {"window_size": 77, "lr": 1e-4}}, "source-study-run-id"

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: ({"seeds": [42]}, {"window_size": 128, "lr": 1e-3}),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "load_study_result_from_mlflow", fake_load_study_result_from_mlflow)
    monkeypatch.setattr(tuning_runner, "run_seed_task", FakeRunSeedTask())
    monkeypatch.setattr(tuning_runner, "NodeAffinitySchedulingStrategy", lambda node_id, soft: (node_id, soft))
    monkeypatch.setattr(
        tuning_runner.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-id"),
    )
    monkeypatch.setattr(tuning_runner.ray, "get", lambda future: future)
    monkeypatch.setattr(
        tuning_runner.ray,
        "wait",
        lambda futures, num_returns=1: (futures[:num_returns], futures[num_returns:]),
    )
    monkeypatch.setattr(tuning_runner, "tune_resource_request_dict", lambda args: {"CPU": 1.0, "GPU": 0.5})
    monkeypatch.setattr(tuning_runner.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_runner, "save_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: result,
    )
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda *args, **kwargs: None,
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        source_experiment_id="source-exp-1",
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        storage_path=str(tmp_path / "storage"),
        config_dir=str(CONFIG_DIR),
    )

    tuning_runner.execute_single_train_workflow(args)

    assert captured["source_lookup"] == {
        "experiment_id": "source-exp-1",
        "model_name": "neutralad",
        "dataset_name": "cont_reactive_ome",
        "min_completed_seeds": 2,
    }
    assert captured["resolved_hparams"] == {"window_size": 77, "lr": 1e-4}


def test_execute_single_train_workflow_reports_best_alarm_score_from_seed_max(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

        def create_run(self, _experiment_id: str, tags=None, run_name: str = ""):
            del tags
            run_id = "study-run-id" if run_name == "neutralad__cont_reactive_ome__20260311_000000" else "trial-run-id"
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))

        def set_terminated(self, _run_id: str) -> None:
            return None

    class FakeRunSeedTask:
        def options(self, **kwargs):
            del kwargs
            return self

        def remote(
            self,
            resolved_hparams,
            args,
            seed,
            trial_root_dir,
            trial_run_id,
            trial_name,
        ):
            del resolved_hparams, args, trial_root_dir, trial_run_id, trial_name
            if seed == 42:
                return {"alarm_score": (0.75, 0.9), "aaf": (0.2, 0.2)}
            return {"alarm_score": (0.4, 0.9), "aaf": (0.1, 0.2)}

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: ({"seeds": [42, 43]}, {"window_size": 128, "lr": 1e-3}),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(tuning_runner, "run_seed_task", FakeRunSeedTask())
    monkeypatch.setattr(tuning_runner, "NodeAffinitySchedulingStrategy", lambda node_id, soft: (node_id, soft))
    monkeypatch.setattr(
        tuning_runner.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-id"),
    )
    monkeypatch.setattr(tuning_runner.ray, "get", lambda future: future)
    monkeypatch.setattr(
        tuning_runner.ray,
        "wait",
        lambda futures, num_returns=1: (futures[:num_returns], futures[num_returns:]),
    )
    monkeypatch.setattr(tuning_runner, "tune_resource_request_dict", lambda args: {"CPU": 1.0, "GPU": 0.5})
    monkeypatch.setattr(tuning_runner.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_runner, "save_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tuning_runner,
        "_finalize_study_outputs",
        lambda client, *, args, trial_run_id, trial_artifact_dir, resolved_hparams, result: result,
    )
    snapshot_payloads = []
    monkeypatch.setattr(
        tuning_runner,
        "_persist_pair_snapshot",
        lambda *args, **kwargs: snapshot_payloads.append(dict(kwargs["result"])),
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        source_experiment_id=None,
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        storage_path=str(tmp_path / "storage"),
        config_dir=str(CONFIG_DIR),
    )

    _model_name, _dataset_name, result = tuning_runner.execute_single_train_workflow(
        args,
        hparams_payload={"config": {"window_size": 58, "lr": 1e-3}},
    )

    assert result["alarm_score"] == pytest.approx(0.575)
    assert result["best_alarm_score"] == pytest.approx(0.75)
    assert result["best_aaf"] == pytest.approx(0.1)
    assert result["mean_aaf"] == pytest.approx(0.15)
    assert "mean_alarm_score" not in result
    assert result["theory_best_alarm_score"] == pytest.approx(0.9)
    assert result["theory_best_aaf"] == pytest.approx(0.2)
    assert [payload["completed_seed_count"] for payload in snapshot_payloads] == [1, 2]


def test_execute_single_train_workflow_rejects_invalid_lnt_window_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def get_experiment(self, _experiment_id: str):
            return SimpleNamespace(name="fake-experiment")

    monkeypatch.setattr(
        tuning_runner,
        "_load_study_params",
        lambda args, model_name: ({"seeds": [42]}, {"window_size": 128, "lr": 1e-3}),
    )
    monkeypatch.setattr(tuning_runner, "MlflowClient", lambda *_args, **_kwargs: FakeClient())

    args = Namespace(
        model_name="lnt",
        dataset_name="industry_process",
        timestamp="20260318_000000",
        source_experiment_id=None,
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        storage_path=str(tmp_path / "storage"),
        config_dir=str(CONFIG_DIR),
    )

    with pytest.raises(ValueError, match=r"window_size >= 41"):
        tuning_runner.execute_single_train_workflow(
            args,
            hparams_payload={
                "config": {
                    "window_size": 38,
                    "model": {
                        "network": {
                            "init_args": {
                                "encoder_type": "bosch_cpc",
                            }
                        }
                    },
                }
            },
        )
