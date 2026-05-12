# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project conventions

- Use `Optional[...]` instead of `| None` in type annotations (see `AGENTS.md`).
- Python target is `>=3.12`; the native bootstrap pins 3.13 for the managed venv.
- Dependency manager is `uv`. Always invoke entry points via `uv run …`.

## Commands

### Environment setup

```bash
# Launcher-only (control plane + cluster CLI)
uv sync --extra cluster

# Full benchmark dependencies (workers / local benchmark code)
uv sync --extra benchmark

# Test deps
uv sync --extra test
```

### Tests / lint / smoke

```bash
uv run pytest                          # full suite
uv run pytest tests/test_orchestration.py::TEST_NAME   # single test
uv run python -m compileall src        # CI smoke compile

# CI lints a curated allow-list (see .github/workflows/tests.yml). Mirror that
# when adding files to the lint pass:
uv run ruff check <files>
```

The CI workflow installs deps with `uv pip sync --torch-backend cpu` from a frozen
`uv export`. Tests are written to run on CPU; do not assume GPU.

### Cluster CLI (primary entrypoint)

```bash
uv run noboom-run-cluster --machine-ip-file ./inventory.yaml \
  --ssh-user <u> --ssh-key <key> --root-dir <remote_root> \
  --dataset <name> --model <name> [--tune] [--gpus-per-run 0.25] \
  [--deployment-mode docker|native] [--experiment-id <src_exp>] \
  [--exclusive] [--force-restart] [-v]

uv run noboom-run-cluster update-allowed --machine-ip-file ./inventory.yaml ...
```

`--dataset` / `--model` accept multiple values (legacy `--model a b c` is normalized
to repeated `--model X` flags inside `noboom_cluster/cli.py`). Bundled inventories
`a100.yaml` and `jarvis.yaml` ship inside `src/noboom_cluster/`.

### Native bootstrap (per-host one-time setup)

```bash
python ./scripts/bootstrap_native_remote.py --local --root-dir <root>           # worker
python ./scripts/bootstrap_native_remote.py --local --root-dir <root> --head    # head
python ./scripts/bootstrap_native_remote.py --host <ip> --ssh-user <u> ...      # remote
python ./scripts/bootstrap_native_remote.py ... --rotate-postgres-password      # rotate
python ./scripts/bootstrap_native_remote.py ... --uninstall                     # remove
```

`POSTGRES_PASSWORD` must be exported in the launcher shell for any head bootstrap,
password rotation, or `--deployment-mode native` launch — there is no fallback.
`scripts/bootstrap_native_remote.sh` is a thin wrapper around the Python script.

## Architecture

The repo splits into two installable packages under `src/`:

- **`noboom_cluster`** — the launcher / control plane. Its CLI (`noboom-run-cluster`,
  defined in `noboom_cluster/cli.py`) is what users run on their local machine. It
  never trains a model itself; it builds a runtime bundle from the working tree,
  pushes it to remote nodes over SSH, starts/joins a Ray cluster, brings up
  PostgreSQL + SeaweedFS + MLflow on the head node, forwards Ray (8265) and MLflow
  (5001) ports back to localhost, then submits a Ray job that imports
  `noboom_benchmark` on the workers.
- **`noboom_benchmark`** — the actual benchmark library that runs on the Ray workers.

### Two deployment modes, one launcher

`--deployment-mode docker` (default) renders a Ray-Docker stack from
`noboom_cluster/noboom_cli_lib/templates/` and uses `--runtime=nvidia` containers.
`--deployment-mode native` requires each node to be pre-bootstrapped via
`scripts/bootstrap_native_remote.py`, which installs a managed `.venv`, builds
`treeple` (and optionally `torch-cluster`) from source, and (with `--head`)
installs PostgreSQL and SeaweedFS under `<root_dir>/postgres` and
`<root_dir>/seaweedfs`. Ownership is recorded in `<root_dir>/.noboom-native-bootstrap/state.json`
— the script will refuse to overwrite unmanaged paths under that root.

The first node in the inventory is always the Ray head; the rest are workers.

### Two CLIs, two entry points

`noboom_benchmark/run_tune.py` is a deprecated argparse shim around
`noboom_lib.core.tune.orchestration.run_tune`. New work flows through
`noboom_cluster/cli.py` → `ray_lifecycle.run_cluster_job` → Ray job submission →
workers call `noboom_lib.core.tune.orchestration.run_tune`. The shim is kept for
direct-script execution and tests; do not add new flags to it.

### Tune vs. evaluate

`--tune` triggers Optuna hyperparameter search. Without `--tune`:
- with `--experiment-id`, the controller looks up the latest study-level
  `summary/result.json` for each `(model, dataset)` pair in that source MLflow
  experiment and reuses the resolved hyperparameters for a single-train run;
- without `--experiment-id`, defaults from `cluster_files/configs/models/<model>.yaml`
  are used.

`--experiment-id` is the *source* experiment to read from — every launch creates
a new destination experiment from `controller_settings.experiment_name`.

### Pair-job orchestration (`noboom_lib/core/tune/`)

`orchestration.run_tune` is the controller. It:
1. validates `ControllerSettings` / `RuntimeConfig` (env-driven via
   `noboom_cluster/noboom_cli_lib/settings.py`),
2. resolves the dependency manifest (`DependencyResolver`),
3. enumerates `(dataset, model)` pairs and submits one Ray job per pair through
   `PairJobController` (`tune/job_submission.py`), bounded by `--max-in-flight`,
4. on each pair, the worker calls `pair_execution.run_pair_spec` →
   `tuning_runner.run_tune_or_train`, which builds the Optuna study, the Lightning
   `BenchmarkCLI`, and the Ray Tune trainable.

Pair-level outputs feed back through `MlflowSeafilePairOutputCallback`
(`tune/callbacks.py`), which logs to MLflow and (optionally) syncs artifacts to
Seafile via `tune/storage.upload_to_seafile`.

### Configs

Model configs live at `src/noboom_cluster/cluster_files/configs/models/<model>.yaml`,
search/study configs at `…/configs/params/<model>.yaml`. `common.yaml` in each
folder is merged in first. The `--model` CLI value is the YAML basename (e.g.
`gdn`, `lstm_ae`, `neutralad`, `timesnet`, `hpad`, `physdiff`).

### Datasets and suffixes

`--dataset` accepts a NoBOOM dataset name plus an optional suffix:
- `_tsst` — TSST synthetic flow; data comes from Seafile/WebDAV (requires
  `SEAFILE_USERNAME` + `SEAFILE_PASS`); training is two-stage (synthetic, then
  real) with reporting from the real stage.
- `_ext` — OME-only; reuses the `cont_reactive_ome` Kaggle download, intersects
  features with the extension CSVs, and appends extension sequences to *training*
  only.
- `_red` — OME-only; same feature intersection as `_ext` but no extension
  sequences appended; reduced feature set across train/val/test.

`_tsst` combinations with `_ext`/`_red` are not a supported documented path.

### Required environment variables

| Need | Vars |
| --- | --- |
| Kaggle dataset download | `KAGGLE_API_TOKEN` |
| `_tsst` synthetic data | `SEAFILE_USERNAME`, `SEAFILE_PASS` |
| Seafile result sync | also `SEAFILE_ROOT_PATH` (and optional `NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS=1`) |
| Native head bootstrap / native launches / postgres rotation | `POSTGRES_PASSWORD` |

`EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, `OPTUNA_STORAGE_URI`,
`NOBOOM_MAPPED_STORAGE`, `NOBOOM_S3_BUCKET` are launcher-internal — do not
require users to set them.

### Outputs

- MLflow study/trial/seed/stage runs at `http://127.0.0.1:5001` (forwarded).
- Ray dashboard at `http://127.0.0.1:8265` (forwarded).
- Per-experiment data on the head node under `<root_dir>/experiment_data/<experiment_name>/`,
  with pair working dirs at `…/experiment_data/<experiment_name>/ray/`.
- Study-level `summary/result.json` artifact + `noboom_experiments.xlsx` in the
  experiment directory.

## Test layout notes

`tests/test_benchmark_cli.py` (and similar) install lightweight `sys.modules`
stubs to import `noboom_lib.core.benchmark_utils.benchmark_cli` without pulling
in heavy ML deps. When adding tests that touch the Lightning/Ray surfaces, follow
that pattern instead of importing the full module tree.
