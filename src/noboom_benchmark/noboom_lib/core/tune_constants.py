from pathlib import Path

CPU_ONLY_MODELS = {"threshold", "kmeans", "eif", "hmm", "gmmhmm", "ocsvm", "pca", "hbos"}
DETECTOR_MANAGED_OPTIMIZER_MODELS = {
    "carots",
    "dada",
}
ARTIFACT_DIR = Path("artifacts")
CKPTS_DIR = ARTIFACT_DIR / Path("ckpts")
CKPT_NAME = "model.ckpt"
TORCH_FLOAT_PREC = "high"
NUM_CPUS_PER_JOB = 2
TUNE_PROGRESS_ATTR = "prune_progress"


def uses_lightning_optimizer(model_name: str) -> bool:
    return model_name not in CPU_ONLY_MODELS and model_name not in DETECTOR_MANAGED_OPTIMIZER_MODELS
