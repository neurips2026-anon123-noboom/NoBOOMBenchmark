import asyncio
from contextlib import contextmanager

import ray

RAY_NAMESPACE = "noboom_benchmark"


@ray.remote(max_restarts=-1, max_task_retries=-1)
class AsyncLock:
    def __init__(self):
        """Initialize an async lock actor."""
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Block until the lock is acquired.

        Returns:
            None: Waits until the lock is acquired.
        """
        await self._lock.acquire()

    def release(self):
        """Release the lock.

        Returns:
            None: The lock is released.

        Raises:
            RuntimeError: If the lock is not held.
        """
        # asyncio.Lock.release() raises RuntimeError if not locked,
        # so you may want to handle that here if needed.
        self._lock.release()


@contextmanager
def ray_lock(lock_actor: AsyncLock):
    """Context manager to acquire and release a Ray async lock.

    Args:
        lock_actor (AsyncLock): Ray actor wrapping an asyncio lock.

    Yields:
        None: Context for protected execution.

    Side Effects:
        Blocks on Ray tasks to acquire and release the lock.
    """
    ray.get(lock_actor.acquire.remote())
    try:
        yield
    finally:
        ray.get(lock_actor.release.remote())
