"""Compilation helpers for manifest-backed transform pipelines."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from jsonargparse import Namespace
from timesead.data.transforms import DatasetSource, Transform


StepNode = MutableMapping[str, Any]


class _DummyTransform(Transform):
    """Placeholder parent used while building transform chains."""

    def __init__(self) -> None:
        super().__init__(None)

    def _get_datapoint_impl(self, item: int):  # pragma: no cover - never called directly
        return None, None


class _ComposeTransform(Transform):
    """Attach a composed transform chain to a concrete dataset source."""

    def __init__(self, data: Transform, parent: Transform):
        super().__init__(parent)
        if data is None:
            raise ValueError("data must be provided.")
        parents = []
        current = self
        try:
            parents.append(type(current).__name__)
            while not isinstance(current.parent, (_DummyTransform, DatasetSource)):
                current = current.parent
                parents.append(type(current).__name__)
        except AttributeError as exc:
            raise ValueError(f"Could not find parent transform in chain: {parents}") from exc
        if isinstance(current.parent, _DummyTransform):
            current.parent = data

    def _get_datapoint_impl(self, item: int):
        return self.parent._get_datapoint_impl(item)


def _recursive_set_init_arg(
    node: StepNode,
    *,
    key: str,
    value: Any,
    only_if_present: bool = True,
    child_fields: Iterable[str] = ("parent",),
) -> StepNode:
    if not isinstance(node, MutableMapping):
        raise TypeError(f"node must be a mutable mapping, got {type(node)}")
    init_args = node.get("init_args")
    if init_args is None:
        init_args = {}
        node["init_args"] = init_args
    if not isinstance(init_args, MutableMapping):
        raise TypeError(f"node['init_args'] must be a mutable mapping, got {type(init_args)}")

    if (not only_if_present) or (key in init_args):
        init_args[key] = value

    for field in child_fields:
        child = init_args.get(field)
        if isinstance(child, MutableMapping) and "class_path" in child:
            _recursive_set_init_arg(
                child,
                key=key,
                value=value,
                only_if_present=only_if_present,
                child_fields=child_fields,
            )
    return node


def _dict_to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Namespace(**{key: _dict_to_namespace(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_dict_to_namespace(item) for item in value]
    return value


class DatasetTransformCompiler:
    """Compile YAML transform step lists into instantiated transform callables."""

    compose_class_path = (
        "noboom_benchmark.noboom_lib.core.benchmark_utils.transform_compiler._ComposeTransform"
    )

    @classmethod
    @lru_cache(maxsize=64)
    def _compile_cached(cls, cache_key: str) -> Dict[str, Any]:
        payload = json.loads(cache_key)
        normalized = payload["steps"]
        window_size = payload["window_size"]
        prediction_horizon = payload["prediction_horizon"]

        dummy_parent = {
            "class_path": "noboom_benchmark.noboom_lib.core.benchmark_utils.transform_compiler._DummyTransform",
            "init_args": {},
        }
        normalized[0].setdefault("init_args", {})
        normalized[0]["init_args"]["parent"] = dummy_parent

        chain: Optional[Dict[str, Any]] = None
        for step in normalized:
            if chain is None:
                chain = deepcopy(step)
            else:
                wrapped = deepcopy(step)
                wrapped.setdefault("init_args", {})
                wrapped["init_args"]["parent"] = chain
                chain = wrapped

        if chain is None:
            raise ValueError("steps must not be empty.")

        compiled: Dict[str, Any] = {
            "class_path": cls.compose_class_path,
            "init_args": {"parent": chain},
        }
        _recursive_set_init_arg(compiled, key="prediction_horizon", value=prediction_horizon)
        _recursive_set_init_arg(compiled, key="window_size", value=window_size)
        return compiled

    @classmethod
    def compile(
        cls,
        *,
        steps: Optional[List[StepNode]],
        window_size: Optional[int],
        prediction_horizon: Optional[int],
    ) -> Optional[Namespace]:
        if not steps:
            return None

        normalized: List[dict[str, Any]] = []
        for index, step in enumerate(steps):
            if not isinstance(step, MutableMapping):
                raise TypeError(f"steps[{index}] must be dict-like, got {type(step)}")
            if "class_path" not in step:
                raise ValueError(f"steps[{index}] is missing class_path: {dict(step)}")
            init_args = step.get("init_args", {})
            if init_args is None:
                init_args = {}
            if not isinstance(init_args, MutableMapping):
                raise TypeError(f"steps[{index}].init_args must be dict-like, got {type(init_args)}")
            normalized.append(
                {
                    "class_path": str(step["class_path"]),
                    "init_args": dict(init_args),
                }
            )

        cache_key = json.dumps(
            {
                "steps": normalized,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
            },
            sort_keys=True,
            default=str,
        )
        compiled = deepcopy(cls._compile_cached(cache_key))
        return _dict_to_namespace(compiled)
