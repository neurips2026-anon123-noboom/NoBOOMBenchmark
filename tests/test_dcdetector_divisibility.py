"""Pin DCdetector's win_size % patch_size divisibility constraint.

The upstream `rearrange` requires `win_size % patch_size == 0`. This test verifies:
1. The runtime guard in DCdetector.__init__ raises ValueError for invalid combos.
2. Every (window_size, patch_size) pair in the curated Optuna grid is divisible.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DCDETECTOR_PATH = (
    REPO_ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "models"
    / "dcdetector"
    / "__init__.py"
)
PARAMS_YAML = REPO_ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params" / "dcdetector.yaml"


def _load_grid() -> Tuple[List[int], List[List[int]]]:
    with PARAMS_YAML.open() as fh:
        cfg = yaml.safe_load(fh)
    window_sizes: List[int] = cfg["search_space"]["window_size"]["choices"]
    patch_choices: List[List[int]] = (
        cfg["search_space"]["model"]["network"]["init_args"]["patch_size"]["choices"]
    )
    return window_sizes, patch_choices


def test_yaml_grid_all_combos_divisible() -> None:
    window_sizes, patch_choices = _load_grid()
    bad = []
    for ws in window_sizes:
        for ps_list in patch_choices:
            for ps in ps_list:
                if ws % ps != 0:
                    bad.append((ws, ps))
    assert not bad, f"YAML grid has non-divisible (window_size, patch_size) pairs: {bad}"


def test_init_contains_divisibility_guard() -> None:
    """Source-level check that DCdetector.__init__ contains the ValueError guard."""
    src = DCDETECTOR_PATH.read_text()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DCdetector":
            init = next(
                (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                None,
            )
            assert init is not None, "DCdetector.__init__ not found"
            init_src = ast.unparse(init)
            assert "_validate_divisible" in init_src, (
                "Expected divisibility check (`win_size % ps != 0`) in DCdetector.__init__"
            )
            return
    raise AssertionError("DCdetector class not found in source")
