"""Optuna helpers for loading search spaces and sampling parameters."""

from typing import Optional, Tuple

from omegaconf import OmegaConf
import optuna


# ---------- Load YAML ----------
def load_config(common_path: str, model_path: str) -> Tuple[dict, dict]:
    """Load shared and model-specific Optuna configuration files.

    Args:
        common_path (str): Path to the shared config YAML.
        model_path (str): Path to the model-specific config YAML.

    Returns:
        Tuple[dict, dict]: Tuple of (study_config, search_space_config).
    """
    # Load YAML via OmegaConf
    common_cfg = OmegaConf.load(common_path)
    model_cfg = OmegaConf.load(model_path)

    common_cfg = OmegaConf.to_container(common_cfg, resolve=True)
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    return common_cfg['study'], model_cfg['search_space']


# ---------- Factory for sampler/pruner from YAML ----------
def build_sampler(cfg_sampler: Optional[dict]):
    """Construct an Optuna sampler instance from a configuration mapping.

    Args:
        cfg_sampler (dict | None): Sampler configuration mapping. Defaults to None.

    Returns:
        optuna.samplers.BaseSampler | None: Configured sampler or None.

    Raises:
        ValueError: If the sampler name is unknown.
    """
    if cfg_sampler is None:
        return None

    name = cfg_sampler["name"]
    args = cfg_sampler.get("args", {})

    if name == "TPESampler":
        return optuna.samplers.TPESampler(**args)
    if name == "RandomSampler":
        return optuna.samplers.RandomSampler(**args)
    if name == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(**args)

    raise ValueError(f"Unknown sampler: {name}")
