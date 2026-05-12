from __future__ import annotations

import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Optional, List

from tqdm import tqdm
from webdav3.client import Client  # pip install webdavclient3
import zstandard as zstd

import logging

logger = logging.getLogger(__name__)


# ----------------------------
# Part detection & sorting
# ----------------------------

@dataclass(frozen=True)
class DavFile:
    path: str   # WebDAV-relative path for webdavclient3
    name: str   # basename


def part_sort_key(name: str) -> tuple:
    """
    Supports: base.tar.zst.part-000, part-001, ...
    Also keeps support for a few other common patterns as fallback.
    """
    # Your pattern: .part-000
    m = re.search(r"\.part-(\d+)$", name, flags=re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))

    # Fallbacks
    m = re.search(r"\.part(\d+)$", name, flags=re.IGNORECASE)
    if m:
        return (1, int(m.group(1)))

    m = re.search(r"\.(\d+)$", name)
    if m:
        return (2, int(m.group(1)))

    return (9, name.lower())


def _infer_archive_base(name: str) -> Optional[str]:
    # Your pattern: X.tar.zst.part-000  -> base = X.tar.zst
    m = re.match(r"^(?P<base>.+\.tar\.zst)\.part-\d+$", name, flags=re.IGNORECASE)
    if m:
        return m.group("base")
    raise ValueError(f"Could not infer archive base, {name}")


def _archive_name_to_dirname(name: str) -> str:
    for suffix in (".tar.zst", ".tzst", ".tar.zstd", ".tzstd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _dir_has_contents(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


# ----------------------------
# Reassembly + extraction
# ----------------------------

def _concatenate_files(part_paths: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".building")
    with tmp.open("wb") as out:
        for p in tqdm(part_paths, desc="Concatenating parts", unit="part"):
            with p.open("rb") as f:
                while True:
                    buf = f.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
    tmp.replace(out_path)


def _extract_tar_zst(archive_path: Path, extract_root: Path) -> Path:
    out_dir = extract_root / _archive_name_to_dirname(archive_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_tar = extract_root / f"{archive_path.stem}.tar"

    with archive_path.open("rb") as src, tmp_tar.open("wb") as dst:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(src) as reader:
            shutil.copyfileobj(reader, dst, length=1024 * 1024)

    with tarfile.open(tmp_tar, mode="r:") as tf:
        for member in tf.getmembers():
            target = out_dir / member.name
            if not _is_within_directory(out_dir, target):
                raise RuntimeError(f"Unsafe path in tar (path traversal): {member.name}")
        tf.extractall(path=out_dir)

    try:
        tmp_tar.unlink()
    except OSError:
        pass

    return out_dir


# ----------------------------
# Public API
# ----------------------------

def download_from_webdav(
    *,
    webdav_hostname: str,
    webdav_login: str,
    webdav_password: str,
    remote_folder: str,
    out_dir: str | Path = ".",
    archive_base: Optional[str] = None,
    keep_parts: bool = False,
) -> str:
    """
    Downloads split parts like:
      tsst_data.tar.zst.part-000
      tsst_data.tar.zst.part-001
      ...

    Reassembles into tsst_data.tar.zst, then extracts into:
      <out_dir>/tsst_data/

    Returns:
      Absolute path to extracted directory.

    Notes:
      Archive name is inferred from the first file in ``remote_folder``.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    #remote_folder = remote_folder if remote_folder.startswith("/") else "/" + remote_folder
    #remote_folder = remote_folder.rstrip("/") + "/"

    client = Client(
        {
            "webdav_hostname": webdav_hostname.rstrip("/"),
            "webdav_login": webdav_login,
            "webdav_password": webdav_password,
        }
    )

    names = client.list(remote_folder)
    names = [n for n in names if n not in (".", "..")]
    if not names:
        raise RuntimeError(f"No files found in WebDAV folder '{remote_folder}'.")

    archive_base = _infer_archive_base(os.path.basename(names[1].rstrip("/")))
    archive_base_norm = _archive_name_to_dirname(archive_base)
    extracted_dir = out_dir / archive_base_norm / archive_base_norm
    if _dir_has_contents(extracted_dir):
        return str(extracted_dir)

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        download_root = Path(tmp_dir)
        with tqdm(desc="Downloading WebDAV", unit="file") as pbar:
            def _progress(current: int, total: Optional[int]) -> None:
                if total:
                    pbar.total = total
                pbar.update(max(0, current - pbar.n))

            client.download_directory(
                remote_path=remote_folder,
                local_path=str(download_root),
                progress=_progress,
            )

        if download_root.is_dir():
            entries = [p for p in download_root.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                download_root = entries[0]

        downloaded_files: List[Path] = []
        for path in download_root.rglob("*"):
            if path.is_file():
                downloaded_files.append(path)

        downloaded_files = sorted(downloaded_files, key=lambda f: part_sort_key(f.name))
        if not downloaded_files:
            raise RuntimeError("No files downloaded from WebDAV.")

        if keep_parts:
            persisted_root = out_dir / f"{archive_base}.download"
            if persisted_root.exists():
                shutil.rmtree(persisted_root)
            shutil.copytree(download_root, persisted_root)

        # Concatenate into full archive
        local_archive = out_dir / archive_base
        _concatenate_files(downloaded_files, local_archive)

        # Extract archive into folder named after archive
        extracted = _extract_tar_zst(local_archive, out_dir)
        return str(extracted / archive_base_norm)
