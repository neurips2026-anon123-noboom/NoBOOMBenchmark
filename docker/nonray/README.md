# Non-Ray Local Benchmark Image

This image is a thin runtime layer over `ghcr.io/denix56/noboom:latest`. It
reuses the published benchmark virtual environment at `/workspace/noboom/.venv`
and expects the repository source to be bind-mounted at `/workspace/noboom-source`.
Keeping those paths separate prevents the source mount from hiding the venv.

It does not start Ray, SeaweedFS/S3, Seafile, or Postgres services. Local mode
defaults are set in the image environment so `run_tune` writes artifacts under
`.noboom_local` unless overridden.

Users should run the published image through the helper:

```bash
python scripts/run_local.py DATASET:MODEL --deployment-mode docker --tune
```

The published runtime image is
`ghcr.io/denix56/noboom-benchmark-non-ray:latest`.

Maintainers can rebuild it from the repository root:

```bash
docker build -f docker/nonray/Dockerfile -t noboom-benchmark-non-ray:local .
```

On Apple Silicon or other non-amd64 hosts, build for the CUDA image platform:

```bash
docker build --platform linux/amd64 -f docker/nonray/Dockerfile -t noboom-benchmark-non-ray:local .
```

Equivalent direct container command for maintainers and debugging:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/noboom-source" \
  -w /workspace/noboom-source \
  ghcr.io/denix56/noboom-benchmark-non-ray:latest \
  python -m noboom_benchmark.run_tune \
    --pair DATASET:MODEL \
    --gpus-per-run 1.0 \
    --timestamp local_docker \
    --config-dir src/noboom_cluster/cluster_files/configs \
    --execution-backend local \
    --artifact-storage-backend local \
    --local-storage-path .noboom_local \
    --optuna-storage-backend memory \
    --tune
```

Use `--optuna-storage-backend sqlite --optuna-sqlite-path .noboom_local/optuna.db`
if you want persisted Optuna state.
