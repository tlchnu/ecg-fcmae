"""YAML-backed argparse configuration helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _flatten_sections(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            nested = _flatten_sections(value)
            overlap = flattened.keys() & nested.keys()
            if overlap:
                raise ValueError(f"Duplicate YAML keys across sections: {sorted(overlap)}")
            flattened.update(nested)
        else:
            flattened[key] = value
    return flattened


def load_yaml_defaults(path: str | Path) -> dict[str, Any]:
    """Load and flatten a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML content must be a mapping")
    return _flatten_sections(config)


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Apply YAML values as argparse defaults, then parse CLI overrides."""
    known, _ = parser.parse_known_args()
    config_path = getattr(known, "config", None)
    if config_path:
        defaults = load_yaml_defaults(config_path)
        valid_destinations = {
            action.dest for action in parser._actions if action.dest != argparse.SUPPRESS
        }
        unknown = sorted(set(defaults) - valid_destinations)
        if unknown:
            raise ValueError(f"Unknown keys in {config_path}: {unknown}")
        parser.set_defaults(**defaults)
    return parser.parse_args()
