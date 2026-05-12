"""Ray job submission for per-pair tune runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shlex
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from ray.job_submission import JobStatus, JobSubmissionClient
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    JobStatus = None  # type: ignore[assignment]
    JobSubmissionClient = None  # type: ignore[assignment]

from noboom_cluster.noboom_cli_lib.runtime_env import build_uv_runtime
from noboom_cluster.noboom_cli_lib.specs import DependencyManifest, PairResult, PairRunSpec

from .interfaces import PairOutputCallback
from .node_ram_guard import (
    PAIR_JOB_ROLE,
    PAIR_JOB_ROLE_ENV,
    PAIR_JOB_ROLE_ENV_ALIASES,
    PairRamGuard,
    RayNodeRamGuard,
)

logger = logging.getLogger(__name__)


def study_suffix(dataset_name: str, model_name: str) -> str:
    return f"{dataset_name}__{model_name}"


def build_pair_run_spec(
    *,
    experiment_id: str,
    source_experiment_id: Optional[str],
    model_name: str,
    dataset_name: str,
    timestamp: str,
    local_storage_path: Path,
    optuna_storage_uri: str,
    tracking_uri: str,
    config_dir: str,
    temp_dir: str,
    data_manifest_path: Optional[str],
    verbose: int,
    tune_mode: bool,
    gpus_per_run: float,
    s3_endpoint_url: Optional[str],
    prepared_dataset_s3_path: Optional[str],
    env_file: Optional[str],
    save_checkpoints: bool = False,
    hpo_seeds: Optional[List[int]] = None,
    execution_backend: str = "ray",
    artifact_storage_backend: str = "remote",
    optuna_storage_backend: str = "configured",
) -> PairRunSpec:
    backend_storage_dir = "local" if execution_backend == "local" else "ray"
    storage_path = str(local_storage_path / backend_storage_dir / study_suffix(dataset_name, model_name))

    return PairRunSpec(
        experiment_id=experiment_id,
        source_experiment_id=source_experiment_id,
        model_name=model_name,
        dataset_name=dataset_name,
        timestamp=timestamp,
        storage_path=storage_path,
        gpus_per_run=gpus_per_run,
        optuna_storage_uri=optuna_storage_uri,
        config_dir=config_dir,
        temp_dir=temp_dir,
        data_manifest_path=data_manifest_path,
        verbose=verbose,
        tune=tune_mode,
        env_file=env_file,
        tracking_uri=tracking_uri,
        s3_endpoint_url=s3_endpoint_url,
        prepared_dataset_s3_path=prepared_dataset_s3_path,
        save_checkpoints=save_checkpoints,
        hpo_seeds=hpo_seeds,
        execution_backend=execution_backend,
        artifact_storage_backend=artifact_storage_backend,
        optuna_storage_backend=optuna_storage_backend,
    )


@contextmanager
def _temporary_env(env_vars: Mapping[str, str]):
    previous: Dict[str, Optional[str]] = {}
    for key, value in env_vars.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PairJobController:
    def __init__(
        self,
        *,
        dependency_manifest: DependencyManifest,
        pair_specs_dir: Path,
        callback: PairOutputCallback,
        env_vars: Mapping[str, str],
        max_in_flight: int = 4,
        client: Optional[JobSubmissionClient] = None,
        ram_guard: Optional[PairRamGuard] = None,
    ) -> None:
        self._dependency_manifest = dependency_manifest
        self._pair_specs_dir = pair_specs_dir
        self._callback = callback
        self._env_vars = {
            key: value
            for key, value in env_vars.items()
            if value is not None and value != ""
        }
        self._max_in_flight = max_in_flight
        if JobSubmissionClient is None:
            raise RuntimeError("Ray is required for PairJobController; use LocalPairJobController.")
        self._client = client or JobSubmissionClient("auto")
        self._ram_guard = ram_guard
        if self._ram_guard is None:
            self._ram_guard = RayNodeRamGuard.from_env(client=self._client)

    def build_runtime_env(self, pair_spec: Optional[PairRunSpec] = None) -> Dict[str, object]:
        env_vars = dict(self._env_vars)
        env_vars[PAIR_JOB_ROLE_ENV] = PAIR_JOB_ROLE
        for env_name in PAIR_JOB_ROLE_ENV_ALIASES:
            env_vars[env_name] = PAIR_JOB_ROLE
        if pair_spec is not None:
            env_vars["NOBOOM_PAIR_SUBMISSION_ID"] = pair_spec.submission_id
            env_vars["NOBOOM_PAIR_ID"] = pair_spec.pair_id
        return {
            "working_dir": ".",
            "env_vars": env_vars,
            "uv": build_uv_runtime(self._dependency_manifest.to_uv_packages()),
        }

    def submit_jobs(self, pair_specs: Iterable[PairRunSpec]) -> List[PairResult]:
        specs = list(pair_specs)
        results: List[PairResult] = []
        pending: Dict[str, PairRunSpec] = {}
        callback_errors: List[BaseException] = []
        callback_futures = []
        pair_iter = iter(specs)

        def notify_start(pair_spec: PairRunSpec) -> None:
            try:
                self._callback.on_job_start(pair_spec)
            except Exception:  # noqa: BLE001
                logger.warning("Pair start callback failed for %s.", pair_spec.submission_id, exc_info=True)

        def submit_one() -> bool:
            try:
                pair_spec = next(pair_iter)
            except StopIteration:
                return False

            pair_spec_payload = pair_spec.to_base64()
            entrypoint = (
                "python -m noboom_benchmark.tuning_workflow "
                f"--pair-spec-b64 {shlex.quote(pair_spec_payload)}"
            )
            logger.info("Submitting pair job %s.", pair_spec.submission_id)
            self._client.submit_job(
                entrypoint=entrypoint,
                runtime_env=self.build_runtime_env(pair_spec),
                submission_id=pair_spec.submission_id,
            )
            notify_start(pair_spec)
            pending[pair_spec.submission_id] = pair_spec
            return True

        with ThreadPoolExecutor(max_workers=1) as callback_executor:
            def _log_callback_exception(future: object) -> None:
                try:
                    future.result()  # type: ignore[union-attr]
                except Exception as exc:  # noqa: BLE001
                    callback_errors.append(exc)
                    logger.exception("Pair output callback failed")

            for _ in range(self._max_in_flight):
                if not submit_one():
                    break

            poll_interval = 2.0
            while pending:
                finished_jobs: List[tuple[str, Any, PairRunSpec]] = []
                jobs = self._batch_jobs(list(pending.keys()))
                for submission_id, pair_spec in list(pending.items()):
                    job_info = jobs.get(submission_id)
                    status = getattr(job_info, "status", None)
                    if status is not None and status.is_terminal():
                        finished_jobs.append((submission_id, job_info, pair_spec))

                if not finished_jobs:
                    if self._ram_guard is not None:
                        self._ram_guard.stop_hot_pending_jobs(pending)
                    time.sleep(poll_interval)
                    poll_interval = min(30.0, poll_interval * 1.5)
                    continue

                poll_interval = 2.0
                for submission_id, job_info, pair_spec in finished_jobs:
                    pending.pop(submission_id, None)
                    status = job_info.status
                    message = getattr(job_info, "message", None)
                    if JobStatus is not None:
                        future = callback_executor.submit(
                            self._callback.on_job_terminal,
                            pair_spec,
                            status.name,
                            message,
                        )
                        callback_futures.append(future)
                        future.add_done_callback(_log_callback_exception)
                    else:
                        logger.error(
                            "Pair job %s finished with status %s.",
                            submission_id,
                            status,
                        )
                    submit_one()

            callback_executor.shutdown(wait=True, cancel_futures=False)

        if callback_errors:
            raise RuntimeError("One or more pair output callbacks failed.") from callback_errors[0]

        for future in callback_futures:
            results.append(future.result())

        return results

    def _batch_jobs(self, submission_ids: List[str]) -> Dict[str, Any]:
        jobs_info = self._client.list_jobs()
        jobs: Dict[str, Any] = {}
        for job_info in jobs_info:
            submission_id = job_info.submission_id
            status = job_info.status
            if submission_id in submission_ids and status is not None:
                jobs[submission_id] = job_info
        return jobs


class LocalPairJobController:
    """Sequential local pair executor with the same callback contract as Ray jobs."""

    def __init__(
        self,
        *,
        dependency_manifest: DependencyManifest,
        pair_specs_dir: Path,
        callback: PairOutputCallback,
        env_vars: Mapping[str, str],
        max_in_flight: int = 1,
        runner: Optional[Any] = None,
    ) -> None:
        del dependency_manifest, max_in_flight
        self._pair_specs_dir = pair_specs_dir
        self._callback = callback
        self._env_vars = {
            key: value
            for key, value in env_vars.items()
            if value is not None and value != ""
        }
        self._runner = runner

    def submit_jobs(self, pair_specs: Iterable[PairRunSpec]) -> List[PairResult]:
        self._pair_specs_dir.mkdir(parents=True, exist_ok=True)
        results: List[PairResult] = []
        if self._runner is None:
            from .pair_execution import run_pair_spec

            runner = run_pair_spec
        else:
            runner = self._runner

        with _temporary_env(self._env_vars):
            for pair_spec in pair_specs:
                spec_path = self._pair_specs_dir / f"{pair_spec.submission_id}.json"
                pair_spec.write_json(str(spec_path))
                status = "SUCCEEDED"
                message = None
                try:
                    logger.info("Running local pair job %s.", pair_spec.submission_id)
                    try:
                        self._callback.on_job_start(pair_spec)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "Pair start callback failed for %s.",
                            pair_spec.submission_id,
                            exc_info=True,
                        )
                    runner(pair_spec)
                except Exception as exc:  # noqa: BLE001
                    status = "FAILED"
                    message = str(exc)
                    logger.exception("Local pair job %s failed.", pair_spec.submission_id)

                pair_result = self._callback.on_job_terminal(pair_spec, status, message)
                if status == "FAILED":
                    pair_result.status = "FAILED"
                    pair_result.partial_result = True
                    pair_result.job_status_message = message
                results.append(pair_result)

        return results
