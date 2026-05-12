from __future__ import annotations

from typing import Tuple


def ensure_cache_services() -> Tuple[object, object]:
    from .dataset_helper_actors import create_preprocessed_cache_registry, create_tune_lock

    registry = create_preprocessed_cache_registry()
    tune_lock = create_tune_lock()
    return registry, tune_lock
