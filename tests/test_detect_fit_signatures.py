"""Audit `detect_fit` signatures against the SourceBackedAnomalyDetector adapter contract.

The adapter introspects each detector's `detect_fit` and supplies an empty test_data
frame when the underlying signature requires two positional args. This test pins the
arity per detector and verifies the helper used by the adapter classifies them
correctly.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPO_ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "models"
    / "_windowed_external.py"
)


def _load_required_positional() -> callable:
    """Load the adapter helper without importing torch-heavy submodules."""
    src = ADAPTER_PATH.read_text()
    namespace: dict = {"inspect": inspect}
    # Extract just the helper function and its imports.
    start = src.index("def _detect_fit_required_positional")
    end = src.index("\n\n\ndef ", start)
    snippet = src[start:end]
    # The helper imports `Callable` from typing already in the file; provide it.
    from typing import Any, Callable  # noqa: F401
    namespace["Callable"] = Callable
    namespace["Any"] = Any
    exec(snippet, namespace)  # noqa: S102 - controlled local source
    return namespace["_detect_fit_required_positional"]


_required_positional = _load_required_positional()


# Audit: (logical name, file path relative to repo, class name, expected required positional after self)
DETECTORS = [
    (
        "anomaly_transformer",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/Anomaly_trans/AnomalyTransformer.py",
        "AnomalyTransformer",
        2,
    ),
    (
        "catch",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/catch/CATCH.py",
        "CATCH",
        2,
    ),
    (
        "dcdetector",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/DCdetector/DCdetector.py",
        "DCdetector",
        2,
    ),
    (
        "carots",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/CAROTS/CAROTS.py",
        "CAROTS",
        1,
    ),
    (
        "dada",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/DADA/DADA.py",
        "DADA",
        1,
    ),
    (
        "oraclead",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/OracleAD/OracleAD.py",
        "OracleAD",
        1,
    ),
    (
        "paano",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/PaAno/PaAno.py",
        "PaAno",
        1,
    ),
    (
        "rtdetector",
        "src/noboom_benchmark/noboom_lib/core/models/_source/baselines/self_impl/RTdetector/RTdetector.py",
        "RTdetector",
        1,
    ),
]


def _required_args_from_source(path: Path, class_name: str) -> Optional[int]:
    """Parse the source for class.detect_fit and count required positional args after self.

    Falls back to None if the method cannot be located.
    """
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "detect_fit":
                    args = item.args
                    pos_or_kw = args.args  # includes 'self'
                    defaults = args.defaults  # rightmost defaults
                    n_required = len(pos_or_kw) - len(defaults) - 1  # exclude self
                    return max(0, n_required)
    return None


@pytest.mark.parametrize("name,relpath,cls,expected", DETECTORS)
def test_detect_fit_required_positional_count(name: str, relpath: str, cls: str, expected: int) -> None:
    path = REPO_ROOT / relpath
    actual = _required_args_from_source(path, cls)
    assert actual == expected, f"{name}: expected {expected} required positional args, got {actual}"


def test_helper_classifies_one_arg_callable() -> None:
    def f(train_data):  # noqa: ANN001
        return train_data

    assert _required_positional(f) == 1


def test_helper_classifies_two_arg_callable() -> None:
    def f(train_data, test_data):  # noqa: ANN001
        return train_data, test_data

    assert _required_positional(f) == 2


def test_helper_ignores_keyword_only_and_var_args() -> None:
    def f(train_data, *args, **kwargs):  # noqa: ANN001
        return train_data, args, kwargs

    assert _required_positional(f) == 1


def test_helper_excludes_defaulted_params() -> None:
    def f(train_data, train_label=None):  # noqa: ANN001
        return train_data, train_label

    assert _required_positional(f) == 1
