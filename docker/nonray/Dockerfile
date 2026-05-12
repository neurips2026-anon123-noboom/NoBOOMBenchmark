FROM ghcr.io/denix56/noboom:latest

LABEL org.opencontainers.image.title="NoBoom non-Ray local benchmark runtime" \
      org.opencontainers.image.description="Runs noboom_benchmark.run_tune sequentially with local artifact and Optuna storage defaults."

# Keep bind-mounted source separate from the published image's benchmark venv.
ENV PATH="/workspace/noboom/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/noboom/.venv" \
    PYTHONPATH="/workspace/noboom-source/src" \
    NOBOOM_EXECUTION_BACKEND="local" \
    NOBOOM_DEPLOYMENT_MODE="local" \
    NOBOOM_ARTIFACT_STORAGE_BACKEND="local" \
    NOBOOM_OPTUNA_STORAGE_BACKEND="memory" \
    NOBOOM_MAPPED_STORAGE="/workspace/noboom-source/.noboom_local" \
    MLFLOW_TRACKING_URI="file:///workspace/noboom-source/.noboom_local/mlruns" \
    OPTUNA_STORAGE_URI="memory://" \
    NOBOOM_S3_BUCKET="" \
    NOBOOM_PREPARED_DATASET_S3_PATH="" \
    S3_ENDPOINT_URL="" \
    SEAFILE_USERNAME="" \
    SEAFILE_PASS="" \
    SEAFILE_ROOT_PATH="" \
    NOBOOM_SEAFILE_UPLOAD_RESULTS="0" \
    NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS="0"

WORKDIR /workspace/noboom-source

CMD ["python", "-m", "noboom_benchmark.run_tune", "--execution-backend", "local", "--artifact-storage-backend", "local", "--optuna-storage-backend", "memory", "--help"]
