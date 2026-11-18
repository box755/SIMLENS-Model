"""YAML configuration loading with `_base_` inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Directory searched for a `_base_` file that is not found next to the configuration
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return `base` updated with `override`, merging nested dictionaries recursively."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_base(cfg_path: Path, visited: set[Path] | None = None) -> dict:
    """Load a configuration file and merge it over the file named by its `_base_` key, recursively."""
    visited = visited or set()
    cfg_path = cfg_path.resolve()
    if cfg_path in visited:
        raise ValueError(f"Config cycle through {cfg_path}")
    visited.add(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw

    # `_base_` is looked up next to the current file first, then in `CONFIG_DIR`
    base_path = (cfg_path.parent / base_name).resolve()
    if not base_path.exists():
        base_path = (CONFIG_DIR / base_name).resolve()
    base = _resolve_base(base_path, visited)
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    A file may name a parent file with the `_base_` key. Its own values are merged over those of the parent.

    Returns:
        `dict`: The resolved configuration.
    """
    return _resolve_base(Path(path))


def save_resolved_config(cfg: dict, out_path: str | Path) -> None:
    """Write the resolved configuration to `out_path` as YAML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def dict_to_ns(d: Any) -> Any:
    """Convert nested dictionaries to `SimpleNamespace` objects for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    return d
