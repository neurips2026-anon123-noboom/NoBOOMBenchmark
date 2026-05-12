from __future__ import annotations

from typing import Dict, List, Optional


CONTROL_PLANE_UV_PACKAGES: List[str] = [
    "pydantic>=2.10",
    "pydantic-settings>=2.6",
    "python-dotenv",
    "tenacity",
]


def build_uv_runtime(packages: Optional[List[str]] = None) -> Dict[str, object]:
    resolved_packages = list(CONTROL_PLANE_UV_PACKAGES)
    if packages:
        resolved_packages.extend(packages)

    return {
        "packages": resolved_packages,
        "uv_pip_install_options": ["--compile", "--no-cache"],
    }
