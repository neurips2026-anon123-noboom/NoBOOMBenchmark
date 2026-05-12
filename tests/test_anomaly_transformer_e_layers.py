"""Pin that AnomalyTransformer honors `config.e_layers` at the construction call site.

Prior to the fix, the call site at AnomalyTransformer.py:221 hard-coded `e_layers=3`,
which silently ignored Optuna search-space proposals. This test inspects the source
to assert the call site forwards `self.config.e_layers`.
"""

from __future__ import annotations

import ast
from pathlib import Path

ANOMALY_TRANSFORMER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "models"
    / "_source"
    / "baselines"
    / "self_impl"
    / "Anomaly_trans"
    / "AnomalyTransformer.py"
)


def test_construction_forwards_e_layers_from_config() -> None:
    src = ANOMALY_TRANSFORMER_PATH.read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "AnomalyTransformer_model":
            for kw in node.keywords:
                if kw.arg == "e_layers":
                    found = True
                    # Must be self.config.e_layers, not a literal int.
                    assert isinstance(kw.value, ast.Attribute), (
                        "AnomalyTransformer_model(e_layers=...) must be self.config.e_layers, "
                        f"got literal {ast.dump(kw.value)}"
                    )
                    assert kw.value.attr == "e_layers"
                    inner = kw.value.value
                    assert isinstance(inner, ast.Attribute) and inner.attr == "config"
    assert found, "Did not find AnomalyTransformer_model(e_layers=...) call site"
