from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from noboom_benchmark.noboom_lib.core.tune_constants import TUNE_PROGRESS_ATTR
from noboom_benchmark.noboom_lib.core.tune_utils import (
    InvalidHyperparameterConfiguration,
    build_tune_scheduler,
    create_extra_params_for_lightning,
    validate_model_hparams,
)


CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
)


def test_build_tune_scheduler_scales_asha_from_seed_units_to_report_units() -> None:
    scheduler = build_tune_scheduler({"name": "ASHA", "args": {"max_t": 6, "grace_period": 2}})

    assert scheduler._time_attr == TUNE_PROGRESS_ATTR
    assert scheduler._max_t == 6
    assert scheduler._brackets[0]._rungs[0][0] == 2


def test_create_extra_params_for_lightning_uses_stage_manifest_mapping(tmp_path: Path) -> None:
    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        temp_dir=str(tmp_path),
        study_params={"metrics": ["alarm_score"]},
        prepared_manifest_paths={
            "real": "/prepared/real/manifest.json",
            "synthetic": "/prepared/synthetic/manifest.json",
        },
    )

    hp_path = create_extra_params_for_lightning(
        args,
        logger_params={"run_id": "seed-run-id"},
        seed=7,
        hp_params={"data.data_source": "synthetic"},
    )
    payload = yaml.safe_load(Path(hp_path).read_text(encoding="utf-8"))

    assert payload["data.data_manifest_path"] == "/prepared/synthetic/manifest.json"


def test_create_extra_params_for_lightning_includes_evaluation_postprocessing(tmp_path: Path) -> None:
    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        temp_dir=str(tmp_path),
        study_params={
            "metrics": ["alarm_score"],
            "evaluation_postprocessing": {
                "enabled": True,
                "short_window": 7,
            },
        },
        prepared_manifest_paths={"real": "/prepared/real/manifest.json"},
    )

    hp_path = create_extra_params_for_lightning(
        args,
        logger_params={"run_id": "seed-run-id"},
        seed=11,
        hp_params={},
    )
    payload = yaml.safe_load(Path(hp_path).read_text(encoding="utf-8"))

    assert payload["model.evaluation_postprocessing"] == {
        "enabled": True,
        "short_window": 7,
    }


def test_create_extra_params_for_lightning_skips_lightning_batches_for_source_backed_models(tmp_path: Path) -> None:
    args = Namespace(
        model_name="pca",
        dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        temp_dir=str(tmp_path),
        study_params={"metrics": ["alarm_score"]},
        prepared_manifest_paths={"real": "/prepared/real/manifest.json"},
    )

    hp_path = create_extra_params_for_lightning(
        args,
        logger_params={"run_id": "seed-run-id"},
        seed=13,
        hp_params={},
    )
    payload = yaml.safe_load(Path(hp_path).read_text(encoding="utf-8"))

    assert payload["trainer.max_epochs"] == 1
    assert payload["trainer.limit_train_batches"] == 0
    assert payload["trainer.limit_val_batches"] == 0


def test_validate_model_hparams_rejects_invalid_lnt_bosch_window_size() -> None:
    with pytest.raises(
        InvalidHyperparameterConfiguration,
        match=r"window_size >= 41",
    ):
        validate_model_hparams(
            "lnt",
            {
                "window_size": 38,
                "model": {
                    "network": {
                        "init_args": {
                            "encoder_type": "bosch_cpc",
                        }
                    }
                },
            },
            config_dir=str(CONFIG_DIR),
        )
