from typing import Dict, Any, Union, Tuple, Optional, List
from multiprocessing import shared_memory
from pathlib import Path
import logging
import pickle
import sys
import asyncio
import time
from collections import defaultdict

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ..tune.tune_helpers import AsyncLock, RAY_NAMESPACE
from .shared_cache import SharedCacheMetadata, build_shared_cache_payload, write_shared_cache, open_shared_memory

logger = logging.getLogger(__name__)


@ray.remote
def _unlink_shared_memory(name: str) -> bool:
    try:
        kwargs = {}
        if sys.version_info >= (3, 13):
            kwargs["track"] = False
        shm = shared_memory.SharedMemory(name=name, create=False, **kwargs)
    except FileNotFoundError:
        return False
    try:
        shm.unlink()
    finally:
        shm.close()
    return True


@ray.remote
def _materialize_shared_cache(shm_name: str, payload: Any) -> bool:
    """Materialize a shared cache on the current node from a payload ref."""
    write_shared_cache(shm_name, payload)
    return True


@ray.remote
def _shared_memory_exists(shm_name: str) -> bool:
    """Check if a shared-memory segment exists on the current node."""
    try:
        shm = open_shared_memory(shm_name, create=False)
    except FileNotFoundError:
        return False
    shm.close()
    return True


@ray.remote
def clear_shm(shm_name: str) -> bool:
    """Check if a shared-memory segment exists on the current node."""
    try:
        shm = open_shared_memory(shm_name, create=False)
        shm.unlink()
    except FileNotFoundError:
        return False
    shm.close()
    return True


def lock_cache_registry(cache_name: str, modes: List[str]) -> bool:
    lock_start = time.perf_counter()
    lock = ray.get_actor(name='async_lock_cache', namespace=RAY_NAMESPACE)
    is_cached = ray.get(lock.lock_cache_request.remote(cache_name, modes))
    logger.info(
        "Cache lock check for '%s' completed in %.2fs (modes=%s, cached=%s).",
        cache_name,
        time.perf_counter() - lock_start,
        modes,
        is_cached,
    )
    return is_cached

def unlock_cache_registry(cache_name: str):
    lock = ray.get_actor(name='async_lock_cache', namespace=RAY_NAMESPACE)
    ray.get(lock.unlock_cache_request.remote(cache_name))


@ray.remote(max_restarts=-1,
    max_task_retries=-1,
    label_selector= {"role": "head"})
class _AsyncLockCacheRegistry:
    """Registry for shared cache payloads and metadata."""
    def __init__(self, ):
        self._cache_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def lock_cache_request(self, cache_name: str, modes: List[str]) -> bool:
        await self._cache_locks[cache_name].acquire()
        registry = get_preprocessed_cache_registry()
        try:
            is_cached = await registry.is_cached.remote(cache_name, modes)
            if is_cached:
                self._cache_locks[cache_name].release()
            return is_cached
        except Exception:
            self._cache_locks[cache_name].release()
            raise

    def unlock_cache_request(self, cache_name: str):
        try:
            self._cache_locks[cache_name].release()
        except RuntimeError:
            pass


@ray.remote(
    max_restarts=-1,
    max_task_retries=-1,
    label_selector= {"role": "head"}
)
class PreprocessedCacheRegistry:
    """Registry for shared preprocessed cache payloads and metadata."""
    def __init__(self):
        self._cache_payload: Dict[str, Dict[str, "ray.ObjectRef[Any]"]] = {}
        self._metadata: Dict[str, Dict[str, SharedCacheMetadata]] = {}
        self._shared_memory_names: set[str] = set()
        self._nodes: set[str] = set()
        self._num_features: Dict[str, int] = {}
        self._seq_len: Dict[str, Tuple[int, ...]] = {}

        self.lock = _AsyncLockCacheRegistry.options(name='async_lock_cache', namespace=RAY_NAMESPACE).remote()

        self._check_restore_state()

    def _check_restore_state(self):
        global_node = ray._private.worker._global_node
        if global_node is not None:
            session_dir = Path(global_node.get_session_dir_path())
            ckpt_file = session_dir / "actor_ckpts" / "cache_registry.pkl"
            if ckpt_file.exists():
                with ckpt_file.open('rb') as f:
                    state = pickle.load(f)
                    for s, val in state.items():
                        setattr(self, s, val)

    def _save_state(self):
        global_node = ray._private.worker._global_node
        if global_node is not None:
            session_dir = Path(global_node.get_session_dir_path())
            ckpt_file = session_dir / "actor_ckpts" / "cache_registry.pkl"
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                '_cache_payload': self._cache_payload,
                '_metadata': self._metadata,
                '_shared_memory_names': self._shared_memory_names,
                '_nodes': self._nodes,
                '_num_features': self._num_features,
            }
            with ckpt_file.open('wb') as f:
                pickle.dump(state, f)


    def get_payload(self, cache_name: str, mode: str) -> Any:
        mode_dict = self._cache_payload.get(cache_name)
        if mode_dict is not None:
            return mode_dict.get(mode)
        return None

    def has_payload(self, cache_name: str, mode: str) -> bool:
        mode_dict = self._cache_payload.get(cache_name)
        if mode_dict is not None:
            return mode in mode_dict
        return False

    def get_metadata(self, cache_name: str, mode: str) -> Optional[SharedCacheMetadata]:
        mode_dict = self._metadata.get(cache_name)
        if mode_dict is not None:
            return mode_dict.get(mode)
        return None

    def get_num_features(self, cache_name: str) -> Union[int, None]:
        return self._num_features.get(cache_name)

    def set_num_features(self, cache_name: str, num_features: int):
        if cache_name not in self._num_features:
            self._num_features[cache_name] = num_features
        return self._num_features[cache_name]

    def get_seq_len(self, cache_name: str, mode: str) -> Union[int, Tuple[int, ...], None]:
        metadata = self.get_metadata(cache_name, mode)
        return metadata.seq_len if metadata is not None else None

    def get_ndim(self, cache_name: str, mode: str) -> Union[int, Tuple[int, ...], None]:
        metadata = self.get_metadata(cache_name, mode)
        return metadata.ndim if metadata is not None else None

    def get_window_size(self, cache_name: str, mode: str) -> Union[int, Tuple[int, ...], None]:
        metadata = self.get_metadata(cache_name, mode)
        return metadata.window_size if metadata is not None else None

    def get_len(self, cache_name: str, mode: str) -> Union[int, None]:
        metadata = self.get_metadata(cache_name, mode)
        return metadata.length if metadata is not None else None

    def _register_shared_memory(self, node_id: str, shm_name: str) -> None:
        self._shared_memory_names.add(shm_name)
        self._nodes.add(f"{node_id}__{shm_name}")
        self._save_state()

    def _is_shm_set_on_node(self, node_id: str, shm_name: str) -> bool:
        return f"{node_id}__{shm_name}" in self._nodes

    def get_shm_nodes(self):
        return list(self._nodes)

    def _set_payload(
        self,
        cache_name: str,
        mode: str,
        payload_ref: "ray.ObjectRef[Any]",
        metadata: SharedCacheMetadata,
    ) -> bool:
        """Register a payload ref and metadata for the given cache/mode."""
        if cache_name not in self._cache_payload:
            self._cache_payload[cache_name] = {}
        if cache_name not in self._metadata:
            self._metadata[cache_name] = {}
        if mode in self._cache_payload[cache_name]:
            return False
        self._cache_payload[cache_name][mode] = payload_ref
        self._metadata[cache_name][mode] = metadata
        self._save_state()
        return True

    def _prepare_cache_for_dataset_impl(
        self,
        cache_name: str,
        mode: str,
        shm_name: str,
        node_id: str,
        parent: Optional[Any],
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        force_rebuild: bool = False,
        payload_ref_holder: Optional[Dict[str, "ray.ObjectRef[Any]"]] = None,
        metadata: Optional[SharedCacheMetadata] = None,
    ) -> SharedCacheMetadata:
        """Ensure the shared cache payload and node-local memory are available."""
        if not self.has_payload(cache_name, mode):
            if payload_ref_holder is not None and metadata is not None:
                payload_ref = payload_ref_holder["payload_ref"]
                self._set_payload(cache_name, mode, payload_ref, metadata)
                force_rebuild = True
            else:
                if parent is None:
                    raise ValueError("parent must be provided to build a new cache entry.")
                if cache_name not in self._num_features:
                    raise ValueError("Number of features '%s' is not set. Please set_num_features before "
                                     "calling this method" % cache_name)
                payload = build_shared_cache_payload(parent, transform, target_transform)
                payload_ref = ray.put(payload)
                metadata = SharedCacheMetadata(
                    num_features=self.get_num_features(cache_name),
                    seq_len=[parent.seq_len] * len(parent) if isinstance(parent.seq_len, int) else parent.seq_len,
                    length=len(parent),
                    ndim=parent.ndim,
                    window_size=parent.window_size,
                )
                logger.info(
                    "Built cache payload for '%s' (%s) with num_features=%s, window_size=%s, length=%s, ndim=%s.",
                    cache_name,
                    mode,
                    metadata.num_features,
                    metadata.window_size,
                    metadata.length,
                    metadata.ndim,
                )
                self._set_payload(cache_name, mode, payload_ref, metadata)
                force_rebuild = True
        else:
            payload_ref = self._cache_payload[cache_name][mode]
            metadata = self._metadata[cache_name][mode]

        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        if force_rebuild or not ray.get(_shared_memory_exists.options(scheduling_strategy=strategy).remote(shm_name)):
            logger.info("Materializing shared cache '%s' on node %s.", shm_name, node_id)
            materialize_start = time.perf_counter()
            ray.get(_materialize_shared_cache.options(scheduling_strategy=strategy).remote(shm_name, payload_ref))
            logger.info(
                "Materialized shared cache '%s' on node %s in %.2fs.",
                shm_name,
                node_id,
                time.perf_counter() - materialize_start,
            )
        if not self._is_shm_set_on_node(node_id, shm_name):
            self._register_shared_memory(node_id, shm_name)

        return metadata

    def prepare_cache_for_dataset(
            self,
            cache_name: str,
            mode: str,
            shm_name: str,
            node_id: str,
            parent: Optional[Any],
            transform: Optional[Any] = None,
            target_transform: Optional[Any] = None,
            force_rebuild: bool = False,
            payload_ref_holder: Optional[Dict[str, "ray.ObjectRef[Any]"]] = None,
            metadata: Optional[SharedCacheMetadata] = None,
    ) -> SharedCacheMetadata:
        return self._prepare_cache_for_dataset_impl(cache_name, mode, shm_name, node_id, parent, transform,
                                                    target_transform, force_rebuild, payload_ref_holder, metadata)


    def is_cached(self, cache_name: str, modes: List[str]) -> bool:
        is_cached = cache_name in self._cache_payload
        if is_cached:
            is_cached = all(mode in self._cache_payload[cache_name] for mode in modes)
        return is_cached

    def _cleanup_shared_memory(self) -> None:
        if not self._shared_memory_names:
            return
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        refs = []
        for name in self._shared_memory_names:
            for node in nodes:
                node_id = node.get("NodeID")
                if not node_id:
                    continue
                strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                refs.append(_unlink_shared_memory.options(scheduling_strategy=strategy).remote(name))
        if refs:
            ray.get(refs)
        self._shared_memory_names = set()

    def __ray_terminate__(self) -> None:
        self._cleanup_shared_memory()

    def __del__(self) -> None:
        try:
            self._cleanup_shared_memory()
        except Exception:
            pass


def create_preprocessed_cache_registry():
    name = "preprocessed_cache_registry"
    return PreprocessedCacheRegistry.options(name=name, namespace=RAY_NAMESPACE).remote()

def create_tune_lock():
    name = "tune_lock"
    return AsyncLock.options(name=name, namespace=RAY_NAMESPACE).remote()

def get_preprocessed_cache_registry():
    name = "preprocessed_cache_registry"
    return ray.get_actor(name, namespace=RAY_NAMESPACE)

def get_tune_lock():
    name = "tune_lock"
    return ray.get_actor(name, namespace=RAY_NAMESPACE)
