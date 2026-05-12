# NoBoom Benchmark

NoBoom Benchmark evaluates multivariate time-series anomaly detection models on the NoBOOM chemical-process datasets.

For single-machine sequential runs, use the local non-Ray path first. It runs
one or more dataset/model pairs without Ray, Seaweed/S3, Seafile, or Postgres,
and writes results under `.noboom_local`. The dedicated non-Ray Docker image is
the simplest way to smoke-test GPU/runtime changes without starting the
distributed stack.

There are two user-facing launch scripts:

- `scripts/run_cluster.py` for Ray/distributed runs
- `scripts/run_local.py` for non-Ray sequential runs

Both scripts use `--deployment-mode native|docker`.

Native local example:

```bash
cd NoBoomBenchmark
uv sync --extra benchmark

uv run python scripts/run_local.py \
  cont_reactive_ome:neutralad \
  --deployment-mode native \
  --tune
```

Docker local example:

```bash
cd NoBoomBenchmark
python scripts/run_local.py \
  cont_reactive_ome:neutralad \
  --deployment-mode docker \
  --tune
```

The helper always uses the published non-Ray image:
`ghcr.io/denix56/noboom-benchmark-non-ray:latest`. It does not build an image
locally.

Expected local outputs:

- native: `.noboom_local/NoBoomBenchmark__local_native/noboom_experiments.xlsx`
- Docker: `.noboom_local/NoBoomBenchmark__local_docker/noboom_experiments.xlsx`
- pair artifacts: `.noboom_local/NoBoomBenchmark__<timestamp>/local/<dataset>__<model>/`

See [Local non-Ray runs](docs/non_ray_local.md) for the dedicated Docker image,
SQLite Optuna state, expected outputs, and validation commands.

The distributed workflow is:

1. Use your local machine as the launcher.
2. Point it at one or more remote GPU hosts over SSH.
3. Start the cluster in either Docker mode or native mode.
4. Run tuning or single-train evaluation through the same CLI.

The launcher forwards the Ray dashboard to `http://127.0.0.1:8265` and the MLflow UI to `http://127.0.0.1:5001`.

## Workflow model

- Your local machine is the control plane only. It renders the runtime bundle, connects over SSH, starts Ray, submits jobs, and forwards ports.
- The actual workloads run on the remote head and worker nodes.
- `docker` and `native` are deployment modes for the remote nodes, not two separate local development workflows.
- The first entry in the inventory file becomes the Ray head node. All later entries are workers.
- Model configs live in `src/noboom_cluster/cluster_files/configs/models/`.
- Search and study configs live in `src/noboom_cluster/cluster_files/configs/params/`.
- `--model` values are the YAML basenames from `src/noboom_cluster/cluster_files/configs/models/`, for example `gdn`, `lstm_ae`, `neutralad`, or `timesnet`.
- `--dataset` values are NoBOOM dataset names plus the supported suffixes described in [Dataset suffixes](#dataset-suffixes).

## Before you start

### Inventory file

Create a YAML inventory file on the launcher machine:

```yaml
nodes:
  - ip: "203.0.113.10"
    devices: "0,1"
  - ip: "203.0.113.11"
    devices: "0"
```

- `ip` is required.
- `devices` is optional.
- If `devices` is set, the launcher writes it into `CUDA_VISIBLE_DEVICES` for that node.
- If `devices` is omitted, the node keeps its default GPU visibility.
- `--machine-ip-file` can point either to your own YAML file or to a bundled inventory shipped with the package, for example `a100.yaml` or `jarvis.yaml`.

### Credential matrix

For `scripts/run_local.py`, the local launcher clears remote storage settings and
does not require Postgres, Seaweed/S3, or Seafile credentials.

For `scripts/run_cluster.py`, export only the credentials that apply to your
Ray/distributed run:

| Purpose | Variables | Required when |
| --- | --- | --- |
| Kaggle dataset download | `KAGGLE_API_TOKEN` | When the requested datasets need to be downloaded from Kaggle. |
| TSST synthetic datasets | `SEAFILE_USERNAME`, `SEAFILE_PASS` | Only when any dataset name ends with `_tsst`. |
| Seafile result sync | `NOBOOM_SEAFILE_UPLOAD_RESULTS=1`, `SEAFILE_USERNAME`, `SEAFILE_PASS`, `SEAFILE_ROOT_PATH` | Only when you want automatic artifact sync to Seafile. Disabled by default. |
| Ray PostgreSQL auth | `POSTGRES_PASSWORD` | Required for `scripts/run_cluster.py` in both Docker and native deployment modes, and for native head bootstraps/password rotation. |

Notes:

- Seafile result sync is opt-in. `_tsst` runs may use Seafile credentials to
  read synthetic input data without uploading benchmark results to Seafile.
- `POSTGRES_PASSWORD` is launcher input, not an optional default. The Ray
  launcher uses it for the managed PostgreSQL-backed MLflow and Optuna stores.
- You do not need to provide `EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, `OPTUNA_STORAGE_URI`, `NOBOOM_MAPPED_STORAGE`, or `NOBOOM_S3_BUCKET` for normal usage. The launcher derives those internally.

## Common setup

### 1. Clone the repository

```bash
git clone https://github.com/neurips2026-anon123-noboom/NoBOOMBenchmark.git
cd NoBoomBenchmark
```

### 2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell or make sure `uv` is on `PATH`.

### 3. Install launcher dependencies

The launcher machine only needs the cluster CLI and its control-plane dependencies:

```bash
uv sync --extra cluster
```

You can then run the launcher without manually activating the virtual environment:

```bash
uv run python scripts/run_cluster.py --help
```

### 4. Export only the credentials you need

Example for a normal Ray Kaggle-backed run:

```bash
export KAGGLE_API_TOKEN="<kaggle_api_token>"
export POSTGRES_PASSWORD="<postgres_password>"
```

Example for a Ray `_tsst` run with Seafile result sync:

```bash
export KAGGLE_API_TOKEN="<kaggle_api_token>"
export POSTGRES_PASSWORD="<postgres_password>"
export SEAFILE_USERNAME="<seafile_username>"
export SEAFILE_PASS="<seafile_password>"
export SEAFILE_ROOT_PATH="<remote_seafile_results_folder>"
export NOBOOM_SEAFILE_UPLOAD_RESULTS="1"
# Optional: keep checkpoint files in Seafile too. Disabled by default.
export NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS="1"
```

## Docker mode

Docker mode is the default deployment mode. Use it when the remote hosts can run GPU-enabled Docker containers.

### Remote host assumptions

The launcher can prepare parts of the Docker environment for you, but it does not remove these base requirements:

- Passwordless SSH access from the launcher machine to every remote node.
- `sudo` access for the SSH user on the remote nodes.
- NVIDIA GPU drivers already installed on the remote nodes.
- NVIDIA container runtime support on the remote nodes, because the launcher starts Ray containers with `--runtime=nvidia`.

### What the launcher does automatically

When you start a Docker-mode run, the launcher:

- builds a temporary runtime bundle from the current repo state
- renders the cluster YAML and environment files
- prepares Docker and Docker Compose on the remote nodes
- starts the Ray cluster
- starts PostgreSQL, SeaweedFS, and MLflow on the head node
- forwards Ray and MLflow ports back to your launcher machine
- submits the benchmark job through the Ray job server

### Docker tuning run

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir> \
  --deployment-mode docker \
  --dataset <dataset_name> \
  --model <model_name> \
  --tune \
  --gpus-per-run <gpus_per_single_train_run>
```

Example:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user cloud \
  --ssh-key ~/.ssh/id_ed25519 \
  --root-dir ~/noboom \
  --deployment-mode docker \
  --dataset cont_reactive_ome \
  --model gdn \
  --tune \
  --gpus-per-run 0.25
```

### Docker evaluation run

Single-train evaluation is the default when you omit `--tune`.

Reuse the latest study result from a previous MLflow experiment:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir> \
  --deployment-mode docker \
  --dataset <dataset_name> \
  --model <model_name> \
  --experiment-id <source_mlflow_experiment_id> \
  --gpus-per-run <gpus_per_single_train_run>
```

Run single-train evaluation with the default model parameters from the in-repo configs:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir> \
  --deployment-mode docker \
  --dataset <dataset_name> \
  --model <model_name> \
  --gpus-per-run <gpus_per_single_train_run>
```

## Native mode

Native mode runs the benchmark directly on the remote hosts without Docker. Use it when the target machines cannot or should not run the containerized stack.

### Native prerequisites

Before you use `--deployment-mode native`, each target root must be bootstrapped. The bootstrap script assumes the target machine has:

- `curl`
- `git`
- `python3`, with Python 3.13 available to `uv`
- `tar`
- `make`
- `cmake`
- `g++`
- a CUDA toolkit with `nvcc` available under `/usr/local/cuda` or `/usr/local/cuda-*`
- `tmux` on the head node, because native SeaweedFS and MLflow services are started in tmux sessions
- `ss` on the head node, because native PostgreSQL startup checks and clears the bound port before restart

For remote bootstrapping, you also need passwordless SSH access from the launcher machine.

### What the bootstrap script installs

`python ./scripts/bootstrap_native_remote.py` installs a managed runtime under the chosen `root_dir` and records ownership in `root_dir/.noboom-native-bootstrap/state.json`. The legacy `scripts/bootstrap_native_remote.sh` entrypoint is only a thin wrapper around the same Python implementation. When run interactively, the Python bootstrap shows step progress, streams remote bootstrap logs back to the local console, and writes persistent logs under `root_dir/.noboom-native-bootstrap/logs/`.

Run the bootstrap on every native node. By default it installs the shared worker runtime. Add `--head` only on the node that will become the Ray head node, which is the first entry in the inventory file.

For head bootstraps and native launches, export `POSTGRES_PASSWORD` in your local shell first. The bootstrap no longer falls back to a hardcoded database password.

On every node it:

- creates `root_dir/.venv`
- syncs the locked benchmark dependencies from `uv.lock`
- builds `treeple` and verifies that it imports successfully before moving on
- optionally builds `torch-cluster` and verifies that its extension imports and a small op runs successfully
- creates the managed metadata under `root_dir/.noboom-native-bootstrap`

With `--head`, it additionally:

- installs PostgreSQL under `root_dir/postgres`
- installs SeaweedFS under `root_dir/seaweedfs`
- links the managed head binaries into `root_dir/.venv/bin`, including `psql`, `pg_ctl`, `initdb`, `postgres`, and `weed`

Safety behavior:

- if the root already contains managed bootstrap paths, the bootstrap removes those managed paths before reprovisioning
- if the root already contains a valid bootstrap-managed install, a new install removes the previously managed assets first and then reprovisions the root
- reruns preserve an existing head bootstrap automatically, and worker roots can be reprovisioned with `--head` when needed
- `--head` is only for the Ray head node; worker nodes should be bootstrapped without it so they only get the shared Python runtime and native libraries
- `--skip-torch-cluster` skips the `torch-cluster` build and validation for shorter smoke tests
- per-stage skip flags are available: `--skip-uv`, `--skip-venv`, `--skip-dependency-sync`, `--skip-treeple`, `--skip-torch-cluster`, `--skip-postgres`, `--skip-seaweed`, and `--skip-symlinks`
- `--rotate-postgres-password` updates the managed PostgreSQL password in place for an existing head bootstrap and refreshes the managed env files under `root_dir`
- `--uninstall` removes only assets recorded in the bootstrap state
- do not point the bootstrap at a root that contains unrelated `.venv`, `postgres`, `seaweedfs`, `treeple`, or `cluster` content you want to keep, because those managed paths are treated as bootstrap-owned during reprovisioning

### Bootstrap the current machine in place

Use `--local` when the current machine itself is the native target:

```bash
python ./scripts/bootstrap_native_remote.py \
  --local \
  --root-dir ~/noboom-native-local
```

For a local head node:

```bash
python ./scripts/bootstrap_native_remote.py \
  --local \
  --root-dir ~/noboom-native-local-head \
  --head
```

For a shorter smoke test:

```bash
python ./scripts/bootstrap_native_remote.py \
  --local \
  --root-dir ~/noboom-native-local \
  --skip-torch-cluster
```

### Bootstrap a remote native host

Worker node:

```bash
python ./scripts/bootstrap_native_remote.py \
  --host 203.0.113.10 \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir>/noboom-native-smoke
```

Head node:

```bash
python ./scripts/bootstrap_native_remote.py \
  --host 203.0.113.10 \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir>/noboom-native-head \
  --head
```

For a shorter smoke test:

```bash
python ./scripts/bootstrap_native_remote.py \
  --host 203.0.113.10 \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir>/noboom-native-smoke \
  --skip-torch-cluster
```

If you want to isolate experiments cleanly, use a dedicated root per native environment, for example `<remote_root_dir>/test_noboom`.

### Remove a bootstrap-managed native install

Local uninstall:

```bash
python ./scripts/bootstrap_native_remote.py \
  --local \
  --root-dir ~/noboom-native-local \
  --uninstall
```

Remote uninstall:

```bash
python ./scripts/bootstrap_native_remote.py \
  --host 203.0.113.10 \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir>/noboom-native-smoke \
  --uninstall
```

### Rotate the managed PostgreSQL password in place

Export the new `POSTGRES_PASSWORD` locally first, then run the bootstrap in password-rotation mode against the existing head root.

Local head root:

```bash
python ./scripts/bootstrap_native_remote.py \
  --local \
  --root-dir ~/noboom-native-local-head \
  --head \
  --rotate-postgres-password
```

Remote head root:

```bash
python ./scripts/bootstrap_native_remote.py \
  --host 203.0.113.10 \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir>/noboom-native-head \
  --head \
  --rotate-postgres-password
```

This mode is only for an already bootstrapped head root. It does not reinstall the runtime.

### Launch a native run

After the target root is bootstrapped, run the launcher with `--deployment-mode native`. The launcher renders fresh env files into both `root_dir/.env` and `root_dir/mnt/.env`, restarts the managed PostgreSQL instance from `root_dir/postgres/pgsql/data`, starts SeaweedFS and MLflow on the head node, forwards the local Ray dashboard and MLflow ports, and then submits the benchmark job through the Ray jobs API.

For native launches, export `POSTGRES_PASSWORD` in the launcher shell before you
run `scripts/run_cluster.py`.

Generic example:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <bootstrapped_remote_root_dir> \
  --deployment-mode native \
  --dataset <dataset_name> \
  --model <model_name> \
  --tune \
  --gpus-per-run <gpus_per_single_train_run>
```

Example using the bundled `a100.yaml` inventory and a dedicated native root:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file a100.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --deployment-mode native \
  --root-dir <remote_root_dir>/test_noboom \
  --ray-temp-dir <remote_root_dir>/test_noboom/tmp/ray \
  --dataset cont_reactive_ome \
  --model neutralad \
  --tune \
  --gpus-per-run 0.25 \
  --exclusive \
  --force-restart \
  -v
```

Single-train native evaluation with a source experiment:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <bootstrapped_remote_root_dir> \
  --deployment-mode native \
  --dataset <dataset_name> \
  --model <model_name> \
  --experiment-id <source_mlflow_experiment_id> \
  --gpus-per-run <gpus_per_single_train_run>
```

## Evaluation

### Tuning vs. evaluation

- `--tune` means hyperparameter search.
- Omitting `--tune` means single-train evaluation.

### Evaluation with `--experiment-id`

When you run without `--tune` and provide `--experiment-id`, the benchmark:

- treats that value as a source MLflow experiment ID
- looks up the latest study-level `summary/result.json` for each `(model, dataset)` pair in that source experiment
- reuses the resolved hyperparameters for the new single-train run

Important:

- `--experiment-id` is not the destination for the new run.
- The new run still goes into a newly created experiment name for the current launcher submission.

### Evaluation without `--experiment-id`

When you run without `--tune` and without `--experiment-id`, the benchmark executes a single-train run using the default model parameters from the in-repo configs under `src/noboom_cluster/cluster_files/configs/`.

### Typical evaluation commands

Reuse best parameters from a previous experiment:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir> \
  --deployment-mode docker \
  --dataset <dataset_name> \
  --model <model_name> \
  --experiment-id <source_mlflow_experiment_id> \
  --gpus-per-run <gpus_per_single_train_run>
```

Run single-train with default parameters:

```bash
uv run python scripts/run_cluster.py \
  --machine-ip-file ./inventory.yaml \
  --ssh-user <ssh_user> \
  --ssh-key <path_to_private_ssh_key> \
  --root-dir <remote_root_dir> \
  --deployment-mode docker \
  --dataset <dataset_name> \
  --model <model_name> \
  --gpus-per-run <gpus_per_single_train_run>
```

### Evaluation outputs

Each run writes or exposes results in three main places:

- MLflow study, trial, seed, and stage runs
- `summary/result.json` in the study-level MLflow artifacts
- `noboom_experiments.xlsx` in the experiment output directory on the head node

## Dataset suffixes

### `_tsst`

Use `_tsst` when you want the TSST synthetic dataset flow.

Behavior:

- the synthetic data is downloaded from WebDAV/Seafile, not Kaggle
- training runs in two stages: synthetic first, then real
- final reporting is taken from the real stage
- Seafile credentials are required because the synthetic archive comes from the `NOBOOM_TSST` WebDAV location

Examples:

- `cont_reactive_ome_tsst`
- `industry_process_tsst`

### `_ext`

`_ext` is currently an OME-only variant.

Behavior:

- it reuses the base Kaggle dataset download path for `cont_reactive_ome`
- it prepares the support directory `_cont_reactive_ome_extension`
- it computes the feature intersection between the base OME dataset and the extension CSV files
- it appends the extension sequences only to the training split
- validation and test stay on the base dataset

Example:

- `cont_reactive_ome_ext`

### `_red`

`_red` is also an OME-only variant.

Behavior:

- it reuses the base Kaggle dataset download path for `cont_reactive_ome`
- it uses the same feature intersection that `_ext` derives from the extension support data
- it does not append extension sequences to training
- it keeps the reduced feature set across the base train, validation, and test splits

Example:

- `cont_reactive_ome_red`

### Shared notes for `_ext` and `_red`

- Both variants reuse the base `cont_reactive_ome` download path.
- Both variants reuse the same dataset fraction used for evaluation thresholding as `cont_reactive_ome`.

### Unsupported suffix combinations

Do not rely on `_tsst` combinations with `_ext` or `_red` in this repository documentation. The current orchestration documents `_tsst`, `_ext`, and `_red` separately, and the combined suffix forms are not a supported README path.

## Outputs and troubleshooting

### Where results go

- On the head node, experiment outputs are stored under `<root_dir>/experiment_data/<experiment_name>/`.
- Pair-job working directories are stored under `<root_dir>/experiment_data/<experiment_name>/ray/`.
- The forwarded MLflow UI is available on the launcher machine at `http://127.0.0.1:5001`.
- The forwarded Ray dashboard is available on the launcher machine at `http://127.0.0.1:8265`.

### Common failure modes

- Docker mode fails before Ray workers come up:
  the remote nodes are missing passwordless SSH, `sudo`, GPU drivers, or NVIDIA container runtime support for `--runtime=nvidia`
- Native bootstrap aborts because managed paths already exist:
  choose a fresh `root_dir`, or uninstall the matching managed install first
- `_tsst` run fails during dataset setup:
  `SEAFILE_USERNAME` and `SEAFILE_PASS` are missing or invalid

## License

This project is licensed under the MIT License.
