"""YAML config loading + CLI override merging.

Configs are simple flat YAML files (see ``scripts/training/configs/*.yaml``).
CLI overrides use dotted-path keys, e.g. ``--set optim.lr=0.001``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML file into a plain dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        yaml.YAMLError: on parse failure.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}.")
    return data


def _coerce(value: str) -> Any:
    """Convert a CLI override string into bool / int / float / list / str."""
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    # ints / floats
    try:
        if value.startswith("0") and len(value) > 1 and not value.startswith(("0.", "0e", "0E")):
            return value  # preserve leading-zero strings like "001"
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    # JSON-ish lists: "[1,2,3]" or "1,2,3"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [_coerce(v.strip()) for v in inner.split(",") if v.strip()]
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",") if v.strip()]
    return value


def _set_dotted(d: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur: MutableMapping[str, Any] = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def merge_overrides(config: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Apply ``key=value`` (or ``a.b.c=value``) overrides to *config*.

    Returns a new dict; the input is not mutated.
    """
    if not overrides:
        return dict(config)
    out = _deepcopy_simple(config)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"--set override must be key=value, got '{ov}'")
        key, val = ov.split("=", 1)
        _set_dotted(out, key.strip(), _coerce(val.strip()))
    return out


def _deepcopy_simple(obj: Any) -> Any:
    """Cheap deepcopy that handles dict/list/scalar (no objects)."""
    if isinstance(obj, Mapping):
        return {k: _deepcopy_simple(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_simple(v) for v in obj]
    return obj


def resolve_paths(config: dict[str, Any], root: str | os.PathLike) -> dict[str, Any]:
    """Resolve all string keys ending in ``_dir`` / ``_path`` / ``_file`` relative to *root*.

    Only resolves relative paths; absolute paths are returned unchanged.
    """
    root = Path(root).resolve()
    out = _deepcopy_simple(config)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if isinstance(v, str) and (
                    k.endswith("_dir") or k.endswith("_path") or k.endswith("_file")
                ):
                    p = Path(v)
                    if not p.is_absolute() and v:
                        node[k] = str((root / p).resolve())
                else:
                    node[k] = _walk(v)
        elif isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    _walk(out)
    return out
