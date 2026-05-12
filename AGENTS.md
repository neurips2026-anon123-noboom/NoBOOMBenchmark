# Agent Instructions

- Use `Optional[...]` instead of `| None` in type annotations.

## Launching Benchmark Runs

- Before launching any benchmark run, confirm the exact models and datasets unless the user has already specified both in the current request or immediately preceding context.
- Prefer the existing PyCharm run configuration requested by the user when one is named. For the `a100` run configuration, use the same working directory and base CLI shape:

  ```bash
  cd <repo_root>/src/noboom_cluster
  export PATH="<repo_root>/.venv/bin:$PATH"
  <repo_root>/.venv/bin/python -m noboom_cluster.cli \
    --dataset <dataset_name> \
    --model <model_name> \
    --machine-ip-file a100.yaml \
    --ssh-user <cluster_user> \
    --ssh-key <path_to_private_ssh_key> \
    --deployment-mode native \
    --root-dir <remote_root_dir> \
    --ray-temp-dir <remote_root_dir>/tmp/ray \
    --exclusive \
    --force-restart \
    -v \
    --tune \
    --gpus-per-run=0.33
  ```

- Replace `<dataset_name>` and `<model_name>` with only the datasets and models confirmed by the user. Do not broaden a requested run to additional models or datasets.
- Run benchmark commands with the repository virtualenv prepended to `PATH`: `<repo_root>/.venv/bin`.
- Preserve required environment variables from the selected run configuration or shell environment, especially `POSTGRES_PASSWORD`, but do not write secret values into logs, summaries, or documentation.
- After launching, report the timestamped Ray job names, the local log path if one was created, and any dashboard or MLflow forwarding URL that is active.
