"""Storage utilities for tune workflows."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Optional, Sequence, Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
import zstandard

logger = logging.getLogger(__name__)


def create_s3_client():
    """Create a boto3 S3 client using environment configuration.

    Returns:
        boto3.client: Configured S3 client.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def ensure_bucket_exists(bucket: str) -> None:
    """Ensure an S3 bucket exists, creating it if needed.

    Args:
        bucket (str): Bucket name to ensure.

    Returns:
        None: Bucket is ensured to exist.

    Raises:
        botocore.exceptions.ClientError: If an unexpected S3 error occurs.

    Side Effects:
        Makes S3 API calls to check and possibly create the bucket.
    """
    seaweed_client = create_s3_client()
    try:
        response = seaweed_client.head_bucket(Bucket=bucket)
        logger.info("SeaweedFS bucket '%s' already exists.", bucket)
        logger.debug(str(response))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchBucket"}:
            logger.info("Creating SeaweedFS bucket '%s'.", bucket)
            seaweed_client.create_bucket(Bucket=bucket)
        else:
            raise


def upload_to_seafile(
    storage_path: Union[str, Path],
    remote_storage_path: Union[str, Path],
    extra_args: Optional[list] = None,
    compressed: bool = False,
    archive_name: Optional[str] = None,
) -> None:
    """Sync a local path to Seafile using rclone.

    Args:
        storage_path (str | Path): Local path to sync.
        remote_storage_path (str | Path): Destination path in Seafile.
        extra_args (Optional[list]): Extra rclone arguments. Defaults to None.

    Returns:
        None: Sync command is executed.

    Side Effects:
        Executes an rclone subprocess to sync files.
    """
    logger.info("Starting Seafile sync via rclone (if configured).")

    try:
        seafile_root = os.getenv("SEAFILE_ROOT_PATH")
        if not seafile_root:
            logger.info("Skipping Seafile sync because SEAFILE_ROOT_PATH is not configured.")
            return
        if compressed:
            storage_str = str(storage_path).rstrip("/")
            if "://" in storage_str:
                parsed = urlparse(storage_str)
                base_name = Path(parsed.path.rstrip("/")).name or parsed.netloc or "payload"
            else:
                base_name = Path(storage_str).name or "payload"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                staging_dir = tmp_path / base_name
                staging_dir.mkdir(parents=True, exist_ok=True)

                copy_cmd = [
                    "rclone",
                    "sync",
                    str(storage_path),
                    str(staging_dir),
                    "-v",
                ]
                if extra_args is not None:
                    copy_cmd.extend(extra_args)

                subprocess.run(copy_cmd, check=True)
                archive_name = archive_name if archive_name is not None else base_name
                tar_name = f"{archive_name}.tar.zst"
                tar_path = tmp_path / tar_name
                with open(tar_path, "wb") as tar_stream:
                    compressor = zstandard.ZstdCompressor(level=15, threads=4)
                    with compressor.stream_writer(tar_stream) as compressed:
                        with tarfile.open(fileobj=compressed, mode="w|") as tar:
                            tar.add(staging_dir, arcname=archive_name)

                upload_cmd = [
                    "rclone",
                    "sync",
                    str(tar_path),
                    f"seafile:{seafile_root}/{str(remote_storage_path)}/",
                    "-v",
                    "--progress"
                ]
                if extra_args is not None:
                    upload_cmd.extend(extra_args)

                subprocess.run(upload_cmd, check=True)
        else:
            cmd = [
                "rclone",
                "sync",
                str(storage_path),
                f"seafile:{seafile_root}/{str(remote_storage_path)}",
                "-v",
            ]

            if extra_args is not None:
                cmd.extend(extra_args)
            subprocess.run(cmd, check=True)
        logger.info("Seafile sync completed successfully.")
    except subprocess.CalledProcessError as exc:
        logger.warning("rclone sync failed on node: %s", exc)


def delete_by_extension(
    bucket: str,
    prefix: str,
    extensions: Optional[Sequence[str]] = None,
    *,
    case_insensitive: bool = False,
) -> int:
    """Delete S3 objects under a prefix, optionally filtering by extension.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix to delete under.
        extensions (Optional[Sequence[str]]): File extensions to filter by. Defaults to None.
        case_insensitive (bool): Whether to match extensions case-insensitively.
            Defaults to False.

    Returns:
        int: Number of deleted objects reported by S3.

    Raises:
        ValueError: If ``prefix`` is empty.

    Side Effects:
        Performs delete operations against S3.
    """
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if not prefix.endswith("/"):
        prefix = prefix + "/"

    # Normalize extensions (if provided)
    norm_exts = None
    if extensions:
        tmp = []
        for ext in extensions:
            ext = ext.strip()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = "." + ext
            tmp.append(ext.lower() if case_insensitive else ext)
        norm_exts = tmp if tmp else None

    seaweed_client = create_s3_client()
    paginator = seaweed_client.get_paginator("list_objects_v2")

    deleted = 0
    batch = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if norm_exts is not None:
                key_cmp = key.lower() if case_insensitive else key
                if not any(key_cmp.endswith(ext) for ext in norm_exts):
                    continue

            batch.append({"Key": key})

            if len(batch) == 1000:
                resp = seaweed_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                deleted += len(resp.get("Deleted", []))
                batch.clear()

    if batch:
        resp = seaweed_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(resp.get("Deleted", []))

    return deleted
