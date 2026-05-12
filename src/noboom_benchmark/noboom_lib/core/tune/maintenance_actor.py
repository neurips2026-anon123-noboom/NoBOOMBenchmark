#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import ray

from .tune_helpers import RAY_NAMESPACE

logger = logging.getLogger(__name__)


@dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


def _run_cmd(cmd: List[str], env: Optional[dict[str, str]] = None) -> CmdResult:
    """Run a subprocess and capture timing and output details.

    Args:
        cmd (List[str]): Command and arguments to execute.
        env (Optional[dict[str, str]]): Environment overrides. Defaults to None.

    Returns:
        CmdResult: Captured command results.

    Side Effects:
        Executes a subprocess.
    """
    t0 = time.time()
    p = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    return CmdResult(
        cmd=cmd,
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        duration_s=time.time() - t0,
    )


@ray.remote(label_selector= {"role": "head"})
class SeaweedRWLock:
    """
    Shared worker lock + exclusive maintenance lock.

    - acquire_worker()/release_worker(): any number of concurrent holders allowed
      BUT blocked while maintenance is active OR maintenance is pending (priority).

    - run_maintenance(): waits until all workers exit, then runs maintenance exclusively.
    """

    def __init__(
        self,
        weed_bin: str = "weed",
        master: str = "127.0.0.1:9333",
        garbage_threshold: float = 0.02,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the SeaweedFS maintenance lock actor.

        Args:
            weed_bin (str): Path to the SeaweedFS binary. Defaults to "weed".
            master (str): SeaweedFS master address. Defaults to "127.0.0.1:9333".
            garbage_threshold (float): Garbage threshold for vacuum. Defaults to 0.02.
            extra_env (Optional[Dict[str, str]]): Extra environment vars for maintenance.
                Defaults to None.
        """
        self.weed_bin = weed_bin
        self.master = master
        self.garbage_threshold = garbage_threshold
        self.extra_env = extra_env or {}

        self._cv = asyncio.Condition()
        self._worker_holders: int = 0

        self._maintenance_active: bool = False
        self._maintenance_pending: int = 0  # count of waiting maintenance requests

        self._maintenance_since: Optional[float] = None

    # -------------------------
    # Shared worker lock
    # -------------------------
    async def _wait_for(self, predicate, timeout_s: Optional[float] = None) -> bool:
        """Wait for a predicate to become true under a condition variable.

        Args:
            predicate (Callable[[], bool]): Predicate to evaluate.
            timeout_s (Optional[float]): Optional timeout in seconds. Defaults to None.

        Returns:
            bool: True if predicate became true, False on timeout.
        """
        if timeout_s is None:
            while not predicate():
                #logger.debug(f"{self.maintenance_active}, {self._maintenance_pending}, {self._worker_holders}")
                await self._cv.wait()
            return True

        deadline = time.time() + timeout_s
        while not predicate():
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._cv.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return False
        return True

    async def acquire_worker(self, timeout_s: Optional[float] = None) -> bool:
        """Acquire the shared worker lock.

        Args:
            timeout_s (Optional[float]): Optional timeout in seconds. Defaults to None.

        Returns:
            bool: True if acquired, False on timeout.
        """
        async with self._cv:
            # Wait until no maintenance is active or pending.
            ok = await self._wait_for(
                lambda: not self._maintenance_active and self._maintenance_pending == 0,
                timeout_s,
            )
            if not ok:
                return False

            self._worker_holders += 1
            return True

    async def release_worker(self) -> None:
        """Release a shared worker lock.

        Returns:
            None: Lock count is decremented.

        Raises:
            RuntimeError: If no worker locks are held.
        """
        async with self._cv:
            if self._worker_holders <= 0:
                raise RuntimeError("release_worker() called but no worker locks are held")
            self._worker_holders -= 1
            # If maintenance is waiting, wake it when last worker leaves.
            self._cv.notify_all()

    # -------------------------
    # Introspection
    # -------------------------
    async def worker_holders(self) -> int:
        """Return the number of active worker holders.

        Returns:
            int: Active worker holder count.
        """
        return self._worker_holders

    async def maintenance_active(self) -> bool:
        """Check whether maintenance is currently active.

        Returns:
            bool: True if maintenance is active.
        """
        return self._maintenance_active

    async def maintenance_locked_for_s(self) -> Optional[float]:
        """Return the duration of the current maintenance lock.

        Returns:
            Optional[float]: Seconds since maintenance started, or None.
        """
        if not self._maintenance_active or self._maintenance_since is None:
            return None
        return time.time() - self._maintenance_since

    # -------------------------
    # Exclusive maintenance
    # -------------------------
    async def _run_shell(self, timeout_s: Optional[float]) -> subprocess.CompletedProcess:
        """Run the SeaweedFS maintenance script in a worker thread.

        Args:
            timeout_s (Optional[float]): Timeout in seconds for the subprocess.

        Returns:
            subprocess.CompletedProcess: Completed process results.

        Side Effects:
            Executes the SeaweedFS shell subprocess.
        """
        env = os.environ.copy()
        env.update(self.extra_env)

        commands = [
            "lock",
            "fs.mergeVolumes -apply",
            f"volume.vacuum -garbageThreshold {self.garbage_threshold}",
            "volume.deleteEmpty -quietFor 1s -apply",
            "unlock",
            "exit",
        ]
        script = "\n".join(commands).rstrip() + "\n"
        cmd = [self.weed_bin, "shell", f"-master={self.master}"]

        return await asyncio.to_thread(
            subprocess.run,
            cmd,
            input=script,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )

    async def run_maintenance(self, timeout_s: Optional[float] = 300) -> Dict[str, Any]:
        """Run SeaweedFS maintenance under an exclusive lock.

        Args:
            timeout_s (Optional[float]): Timeout in seconds. Defaults to 300.

        Returns:
            Dict[str, Any]: Result payload with execution status and output.

        Side Effects:
            Blocks worker acquisition, runs SeaweedFS maintenance commands.
        """
        # 1) Mark maintenance pending, then wait until we can become active.
        #logger.debug(f"{self._maintenance_active}, {self._maintenance_pending}, {self._worker_holders}")
        async with self._cv:
            self._maintenance_pending += 1
            try:
                #logger.debug(f"{self._maintenance_active}, {self._maintenance_pending}, {self._worker_holders}")
                ok = await self._wait_for(
                    lambda: not self._maintenance_active and self._worker_holders == 0,
                    timeout_s,
                )
                #logger.debug(f"ok {ok}")
                if not ok:
                    return {
                        "ran": False,
                        "ok": False,
                        "error": "timeout_waiting_for_workers",
                    }
                # We now become the active maintenance holder.
                self._maintenance_active = True
                self._maintenance_since = time.time()
            finally:
                # No longer "pending" (we either became active, or raised).
                self._maintenance_pending -= 1
                # Wake any waiting workers in case this maintenance attempt timed out
                # before acquiring the exclusive lock (otherwise they would remain
                # blocked waiting for _maintenance_pending to drop to zero).
                self._cv.notify_all()

        # 2) Run the maintenance outside the CV lock so workers can wait concurrently.
        try:
            try:
                r = await self._run_shell(timeout_s)
                ok = r.returncode == 0
            except subprocess.TimeoutExpired as exc:
                ok = False
                r = exc
            return {
                "ran": True,
                "ok": ok,
                "master": self.master,
                "garbage_threshold": self.garbage_threshold,
                "results": {
                    "returncode": getattr(r, "returncode", None),
                    "stdout": getattr(r, "stdout", ""),
                    "stderr": getattr(r, "stderr", ""),
                    "timeout": isinstance(r, subprocess.TimeoutExpired),
                }
            }

        finally:
            # 3) Release maintenance lock and wake everyone.
            async with self._cv:
                self._maintenance_active = False
                self._maintenance_since = None
                self._cv.notify_all()


def get_or_create_maintenance_actor(
    name: str = "seaweed_rw_lock",
    namespace: str = RAY_NAMESPACE,
    **actor_kwargs: Any,
):
    """Get or create the SeaweedFS maintenance actor.

    Args:
        name (str): Actor name. Defaults to "seaweed_rw_lock".
        namespace (str): Ray namespace. Defaults to RAY_NAMESPACE.
        **actor_kwargs: Extra kwargs passed to the actor constructor.

    Returns:
        ray.actor.ActorHandle: Handle to the maintenance actor.
    """
    return SeaweedRWLock.options(
        name=name,
        namespace=namespace,
        lifetime="detached",
        get_if_exists=True,
    ).remote(**actor_kwargs)


def run_maintenance():
    """Trigger SeaweedFS maintenance if SeaweedFS is enabled.

    SeaweedFS maintenance is enabled when NOBOOM_USE_SEAWEED=1.

    Returns:
        None: Maintenance is triggered asynchronously.

    Side Effects:
        Calls a Ray actor method when SeaweedFS is enabled.
    """
    if os.getenv("NOBOOM_USE_SEAWEED") == "1":
        master = f"{os.environ.get('HEAD_LOCAL_IP', '127.0.0.1')}:9333"
        lock = get_or_create_maintenance_actor(
            master=master,
            garbage_threshold=float(os.environ.get("WEED_GC_THRESHOLD", "0.02")),
        )
        lock.run_maintenance.remote()

@dataclass
class worker_lock:
    """
    Sync context manager for the shared "worker" lock.
    It resolves the Ray actor internally by (name, RAY_NAMESPACE).

    Usage:
        with worker_lock(name="seaweed_rw_lock", timeout_s=600):
            ...
    """
    name: str = "seaweed_rw_lock"
    timeout_s: Optional[float] = None

    _actor: Optional[ray.actor.ActorHandle] = None

    def __enter__(self):
        """Enter the worker lock context.

        Returns:
            worker_lock: The context manager instance.

        Raises:
            TimeoutError: If the lock cannot be acquired within timeout.

        Side Effects:
            Acquires a worker lock via Ray.
        """
        if os.getenv("NOBOOM_USE_SEAWEED") == "1":
            master = f"{os.environ.get('HEAD_LOCAL_IP', '127.0.0.1')}:9333"
            self._actor = get_or_create_maintenance_actor(
                master=master,
                garbage_threshold=float(os.environ.get("WEED_GC_THRESHOLD", "0.02")),
            )
            acquired = ray.get(self._actor.acquire_worker.remote(timeout_s=self.timeout_s))
            if not acquired:
                raise TimeoutError(f"Timed out while waiting to acquire worker lock (actor={self.name})")
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the worker lock context and release the lock.

        Args:
            exc_type (type | None): Exception type, if raised.
            exc (BaseException | None): Exception instance, if raised.
            tb (TracebackType | None): Traceback, if raised.

        Returns:
            bool: False to propagate exceptions.

        Side Effects:
            Releases the worker lock via Ray.
        """
        # Release even if the protected block raises.
        if self._actor is not None:
            ray.get(self._actor.release_worker.remote())
        return False  # do not suppress exceptions


@dataclass
class aworker_lock:
    """
    Async context manager for the shared "worker" lock.
    It resolves the Ray actor internally by (name, namespace), defaulting to
    RAY_NAMESPACE.

    Usage:
        async with aworker_lock(name="seaweed_rw_lock", namespace=RAY_NAMESPACE, timeout_s=600):
            ...
    """
    name: str = "seaweed_rw_lock"
    namespace: str = RAY_NAMESPACE
    timeout_s: Optional[float] = None

    _actor: Optional[ray.actor.ActorHandle] = None

    async def __aenter__(self):
        """Enter the async worker lock context.

        Returns:
            aworker_lock: The context manager instance.

        Raises:
            TimeoutError: If the lock cannot be acquired within timeout.

        Side Effects:
            Acquires a worker lock via Ray.
        """
        if os.getenv("NOBOOM_USE_SEAWEED") == "1":
            master = f"{os.environ.get('HEAD_LOCAL_IP', '127.0.0.1')}:9333"
            self._actor = get_or_create_maintenance_actor(
                name=self.name,
                namespace=self.namespace,
                master=master,
                garbage_threshold=float(os.environ.get("WEED_GC_THRESHOLD", "0.02")),
            )
            acquired = await self._actor.acquire_worker.remote(timeout_s=self.timeout_s)
            if not acquired:
                raise TimeoutError(f"Timed out while waiting to acquire worker lock (actor={self.name})")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the async worker lock context and release the lock.

        Args:
            exc_type (type | None): Exception type, if raised.
            exc (BaseException | None): Exception instance, if raised.
            tb (TracebackType | None): Traceback, if raised.

        Returns:
            bool: False to propagate exceptions.

        Side Effects:
            Releases the worker lock via Ray.
        """
        if self._actor is not None:
            await self._actor.release_worker.remote()
        return False
