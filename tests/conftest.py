from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Dict, Tuple

import pytest


_TIMESEAD_MODULE_NAMES = (
    "timesead",
    "timesead.models",
    "timesead.models.common",
    "timesead.models.common.anomaly_detector",
    "timesead.optim",
    "timesead.optim.loss",
    "timesead.data",
    "timesead.data.transforms",
    "timesead.data.transforms.transform_base",
)
_TIMESEAD_ATTR_NAMES = (
    ("timesead.models", "BaseModel"),
    ("timesead.models.common", "AnomalyDetector"),
    ("timesead.models.common.anomaly_detector", "AnomalyDetector"),
    ("timesead.optim.loss", "Loss"),
    ("timesead.data.transforms.transform_base", "Transform"),
)

_REAL_TIMESEAD_MODULES: Dict[str, ModuleType] = {}
_REAL_TIMESEAD_ATTRS: Dict[Tuple[str, str], Any] = {}


def _capture_real_timesead() -> None:
    modules: Dict[str, ModuleType] = {}
    attrs: Dict[Tuple[str, str], Any] = {}
    try:
        for module_name in _TIMESEAD_MODULE_NAMES:
            modules[module_name] = importlib.import_module(module_name)
    except ImportError:
        return

    for module_name, attr_name in _TIMESEAD_ATTR_NAMES:
        module = modules.get(module_name)
        if module is not None and hasattr(module, attr_name):
            attrs[(module_name, attr_name)] = getattr(module, attr_name)

    _REAL_TIMESEAD_MODULES.update(modules)
    _REAL_TIMESEAD_ATTRS.update(attrs)


def _restore_real_timesead() -> None:
    if not _REAL_TIMESEAD_MODULES:
        return

    for module_name, module in _REAL_TIMESEAD_MODULES.items():
        sys.modules[module_name] = module

    for module_name, module in _REAL_TIMESEAD_MODULES.items():
        parent_name, _, child_name = module_name.rpartition(".")
        if parent_name:
            parent = _REAL_TIMESEAD_MODULES.get(parent_name) or sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)

    for (module_name, attr_name), value in _REAL_TIMESEAD_ATTRS.items():
        setattr(_REAL_TIMESEAD_MODULES[module_name], attr_name, value)


_capture_real_timesead()


def pytest_collection_finish(session: pytest.Session) -> None:
    del session
    _restore_real_timesead()


@pytest.fixture(autouse=True)
def restore_timesead_modules():
    _restore_real_timesead()
    yield
    _restore_real_timesead()
