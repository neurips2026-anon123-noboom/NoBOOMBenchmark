from __future__ import annotations

from copy import deepcopy
from typing import (Any, Dict, Iterable, List, Mapping, MutableMapping,
                    Optional, Sequence, TypeAlias)

from jsonargparse import Namespace

from timesead.data.transforms import DatasetSource, Transform
from .dataset_helpers import PreprocessDatasetWrapper

StepNode: TypeAlias = MutableMapping[str, Any]  # expects {"class_path": ..., "init_args": {...}}


class _DummyTransform(Transform):
    def __init__(self):
        """Initialize a dummy transform with no parent."""
        super().__init__(None)

    def _get_datapoint_impl(self, item: int):
        """Return empty datapoint placeholders.

        Args:
            item (int): Index of the datapoint.

        Returns:
            tuple: Tuple of (data, target) placeholders.
        """
        return None, None


class _ComposeTransform(Transform):
    def __init__(self, data: Transform, parent: Transform):
        """Compose transforms by wiring a parent chain.

        Args:
            data (Transform): Base data transform to attach.
            parent (Transform): Parent transform in the chain.

        Raises:
            ValueError: If required transform links are missing.
        """
        super().__init__(parent)
        if data is None:
            raise ValueError("data must be provided.")

        assert parent is not None

        parents = []
        cur = self
        try:
            parents.append(type(cur).__name__)
            while (not isinstance(cur.parent, _DummyTransform)
                   and not isinstance(cur.parent, DatasetSource)
                   and not isinstance(cur.parent, PreprocessDatasetWrapper)):
                cur = cur.parent
                parents.append(type(cur).__name__)
        except AttributeError:
            raise ValueError(f"Could not find parent transform in chain: {parents}")

        # It not assigned - assign
        if isinstance(cur.parent, _DummyTransform):
            cur.parent = data

    def _get_datapoint_impl(self, item: int):
        """Fetch datapoint from the parent transform.

        Args:
            item (int): Index of the datapoint.

        Returns:
            Any: Datapoint from the parent transform.
        """
        return self.parent._get_datapoint_impl(item)


def compose_transform(
    *,
    steps: List[StepNode],
    parent_key: str = "parent",
    compose_class_path: str = "noboom_benchmark.noboom_lib.core.benchmark_utils.cli_utils._ComposeTransform",
) -> Dict[str, Any]:
    """Compose a list of DI-style transform nodes into a nested config.

    Args:
        steps (List[StepNode]): List of DI-style transform nodes.
        parent_key (str): Key to use for parent linkage. Defaults to "parent".
        compose_class_path (str): Class path for the wrapper transform.
            Defaults to "benchmark_utils.cli_utils._ComposeTransform".

    Returns:
        Dict[str, Any]: Nested DI config for a composed transform chain.

    Raises:
        ValueError: If the steps list is empty or malformed.
        TypeError: If steps or init_args are not dict-like.
    """
    if not steps:
        raise ValueError("Transform list is empty.")

    # Normalize to plain dicts (avoid mutating caller objects)
    norm: List[Dict[str, Any]] = []
    for i, s in enumerate(steps):
        if not isinstance(s, MutableMapping):
            raise TypeError(f"steps[{i}] must be a dict-like mapping, got {type(s)}")
        if "class_path" not in s:
            raise ValueError(f"steps[{i}] missing 'class_path'. Got: {dict(s)}")

        init_args = s.get("init_args", {})
        if init_args is None:
            init_args = {}
        if not isinstance(init_args, MutableMapping):
            raise TypeError(f"steps[{i}]['init_args'] must be dict-like, got {type(init_args)}")

        norm.append({"class_path": s["class_path"], "init_args": dict(init_args)})

    # Ensure first step has parent=DummyTransform
    dummy_parent = {
        "class_path": "noboom_benchmark.noboom_lib.core.benchmark_utils.cli_utils._DummyTransform",
        "init_args": {},
    }
    norm[0].setdefault("init_args", {})
    norm[0]["init_args"][parent_key] = dummy_parent
    # Build inside-out: last step becomes outermost (DI nesting)
    chain: Optional[Dict[str, Any]] = None
    for step in norm:
        if chain is None:
            chain = deepcopy(step)
        else:
            wrapped = deepcopy(step)
            wrapped.setdefault("init_args", {})
            wrapped["init_args"][parent_key] = chain
            chain = wrapped

    assert chain is not None

    return {
        "class_path": compose_class_path,
        "init_args": {"parent": chain},
    }


def recursive_set_init_arg(
    node: StepNode,
    *,
    key: str,
    value: Any,
    only_if_present: bool = True,
    child_fields: Iterable[str] = ("parent",),
) -> StepNode:
    """Recursively set an init arg across a DI-style config tree.

    Args:
        node (StepNode): Root config node containing ``class_path`` and ``init_args``.
        key (str): init_args key to set.
        value (Any): Value to assign.
        only_if_present (bool): Only set when key exists. Defaults to True.
        child_fields (Iterable[str]): init_args fields containing nested nodes.
            Defaults to ("parent",).

    Returns:
        StepNode: The same node, modified in place.

    Raises:
        TypeError: If nodes or init_args are not dict-like.
    """
    if not isinstance(node, MutableMapping):
        raise TypeError(f"node must be a mutable mapping (dict-like), got {type(node)}")

    init_args = node.get("init_args")
    if init_args is None:
        init_args = {}
        node["init_args"] = init_args
    if not isinstance(init_args, MutableMapping):
        raise TypeError(f"node['init_args'] must be a dict-like mapping, got {type(init_args)}")

    # Set key at this node
    if (not only_if_present) or (key in init_args):
        init_args[key] = value

    # Recurse into children
    for fld in child_fields:
        child = init_args.get(fld)
        if isinstance(child, MutableMapping) and "class_path" in child:
            recursive_set_init_arg(
                child,
                key=key,
                value=value,
                only_if_present=only_if_present,
                child_fields=child_fields,
            )

    return node


def dict_to_namespace(x: Any) -> Any:
    """Recursively convert mappings to jsonargparse Namespace.

    Args:
        x (Any): Input object to convert.

    Returns:
        Any: Converted Namespace or list preserving structure.
    """
    if isinstance(x, Mapping):
        return Namespace(**{k: dict_to_namespace(v) for k, v in x.items()})
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return [dict_to_namespace(v) for v in x]
    return x
