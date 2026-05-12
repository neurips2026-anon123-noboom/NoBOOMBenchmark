from __future__ import annotations

import getpass
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SOURCE = (
    REPO_ROOT / "src/noboom_cluster/noboom_cli_lib/scripts/mount_files/scripts/filter_busy_gpus.py"
)


def test_filter_busy_gpus_excludes_devices_used_by_other_users(tmp_path: Path) -> None:
    script_path = tmp_path / "filter_busy_gpus.py"
    shutil.copyfile(SCRIPT_SOURCE, script_path)
    env_file = tmp_path / ".env"
    env_file.write_text("CUDA_VISIBLE_DEVICES=0,1,2\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *\"--query-gpu=index,uuid\"* ]]; then\n"
        "  printf '0, GPU-a\\n1, GPU-b\\n2, GPU-c\\n'\n"
        "elif [[ \"$*\" == *\"--query-compute-apps=gpu_uuid,pid\"* ]]; then\n"
        "  printf 'GPU-a, 111\\nGPU-b, 222\\n'\n"
        "elif [[ \"$*\" == *\"--query-gpu=index\"* ]]; then\n"
        "  printf '0\\n1\\n2\\n'\n"
        "else\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "pid=\"${@: -1}\"\n"
        "case \"$pid\" in\n"
        "  111) printf '%s\\n' \"$CURRENT_TEST_USER\" ;;\n"
        "  222) printf 'other_user\\n' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CURRENT_TEST_USER"] = getpass.getuser()

    result = subprocess.run(
        [sys.executable, str(script_path), "--env-file", str(env_file)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,2" in env_file.read_text(encoding="utf-8")
    assert "Excluding GPU 1" in result.stderr
