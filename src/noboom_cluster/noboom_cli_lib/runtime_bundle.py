from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

STYLE_TRANSFER_CHECKPOINT_RELATIVE_PATH = (
    Path("noboom_benchmark")
    / "noboom_lib"
    / "core"
    / "diffstylets"
    / "checkpoints"
    / "model.pth"
)


@dataclass(frozen=True)
class RuntimeBundle:
    root_dir: Path
    runtime_dir: Path
    pair_specs_dir: Path
    mount_files_dir: Path
    head_files_dir: Path
    cluster_config_path: Path
    env_base_path: Path
    machine_file_path: Path
    s3_config_path: Path


def _copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def build_runtime_bundle(project_root: Optional[Path] = None, deployment_mode: str = "docker") -> RuntimeBundle:
    resolved_project_root = project_root or Path(__file__).resolve().parents[3]
    bundle_root = Path(tempfile.mkdtemp(prefix="noboom-runtime-"))

    shutil.copytree(
        resolved_project_root / "src" / "noboom_benchmark",
        bundle_root / "noboom_benchmark",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        resolved_project_root / "src" / "noboom_cluster",
        bundle_root / "noboom_cluster",
        dirs_exist_ok=True,
    )
    _copy_tree_contents(resolved_project_root / "src" / "noboom_cluster" / "cluster_files", bundle_root)

    staged_style_transfer_checkpoint = bundle_root / STYLE_TRANSFER_CHECKPOINT_RELATIVE_PATH
    if not staged_style_transfer_checkpoint.exists():
        raise FileNotFoundError(
            "Style transfer checkpoint is missing from the runtime bundle: "
            f"'{staged_style_transfer_checkpoint}'."
        )

    runtime_dir = bundle_root / ".noboom_runtime"
    pair_specs_dir = runtime_dir / "pair_specs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pair_specs_dir.mkdir(parents=True, exist_ok=True)

    mount_files_dir = bundle_root / "noboom_cluster" / "noboom_cli_lib" / "scripts" / "mount_files"
    head_files_dir = bundle_root / "noboom_cluster" / "noboom_cli_lib" / "scripts" / "head_files"
    cluster_config_path = runtime_dir / f"ray-cluster-{deployment_mode}.yaml"
    env_base_path = mount_files_dir / ".env.base"
    machine_file_path = mount_files_dir / "machine_nodes.yaml"
    s3_config_path = head_files_dir / "s3.json"

    return RuntimeBundle(
        root_dir=bundle_root,
        runtime_dir=runtime_dir,
        pair_specs_dir=pair_specs_dir,
        mount_files_dir=mount_files_dir,
        head_files_dir=head_files_dir,
        cluster_config_path=cluster_config_path,
        env_base_path=env_base_path,
        machine_file_path=machine_file_path,
        s3_config_path=s3_config_path,
    )


def render_template(
    template_name: str,
    output_path: Path,
    context: Dict[str, Any],
    *,
    templates_root: Optional[Path] = None,
) -> Path:
    root = templates_root or Path(__file__).resolve().parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(str(root)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    template = environment.get_template(template_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(**context) + "\n", encoding="utf-8")
    return output_path
