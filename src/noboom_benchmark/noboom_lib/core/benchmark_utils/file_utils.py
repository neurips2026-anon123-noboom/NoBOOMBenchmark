"""Utilities for safely creating and reusing metadata files across processes."""

import json
import os
import time
from typing import Any, Callable, Dict


def get_or_create_metadata(
    meta_path: str,
    compute_metadata_fn: Callable[[], Dict[str, Any]],
    poll_interval: float = 0.1,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Ensure a JSON metadata file exists at ``meta_path``.

    Args:
        meta_path (str): Path to the metadata JSON file.
        compute_metadata_fn (Callable[[], Dict[str, Any]]): Function that computes metadata.
        poll_interval (float): Sleep time between checks while waiting. Defaults to 0.1.
        force_recompute (bool): Whether to force recomputation. Defaults to False.

    Returns:
        Dict[str, Any]: Metadata loaded from the JSON file.

    Raises:
        TypeError: If ``compute_metadata_fn`` returns a non-dict.

    Side Effects:
        Creates lock files and writes JSON metadata to disk.
    """
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    lock_path = meta_path + ".lock"

    # Fast path: if file already exists and is readable, just use it.
    if not force_recompute:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("metadata.json must contain a JSON object.")
            return data
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    # Try to become the creator by creating a lock file atomically.
    try:
        # O_CREAT | O_EXCL => fails if file already exists (atomic)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        i_am_creator = True
    except FileExistsError:
        i_am_creator = False
        lock_fd = None  # just to keep the name defined

    if not i_am_creator:
        # Another process is (or was) creating the metadata.
        # Wait until the file exists and is readable.
        while True:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("metadata.json must contain a JSON object.")
                return data
            except FileNotFoundError:
                # File not there yet, wait.
                time.sleep(poll_interval)
            except (json.JSONDecodeError, ValueError):
                # Probably still being written or incomplete, wait.
                time.sleep(poll_interval)

    # We are the creator: compute and write atomically via a temp file.
    try:
        metadata = compute_metadata_fn()
        if not isinstance(metadata, dict):
            raise TypeError("compute_metadata_fn must return a dict.")

        tmp_path = f"{meta_path}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace: other processes will see either no file or a complete file.
        os.replace(tmp_path, meta_path)
        return metadata

    finally:
        # Release the lock
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
