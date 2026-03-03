"""Structural analysis: JSON schema drift and line-level diff for text outputs."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class StructuralDiff:
    """Result of a structural comparison between two outputs."""

    has_schema_change: bool = False
    added_keys: List[str] = field(default_factory=list)
    removed_keys: List[str] = field(default_factory=list)
    type_changes: Dict[str, str] = field(default_factory=dict)
    format_change: Optional[str] = None  # e.g. "json->text"
    total_changes: int = 0
    # Line-level diff for plain-text outputs (git-style unified diff lines)
    text_diff: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_schema_change": self.has_schema_change,
            "added_keys": self.added_keys,
            "removed_keys": self.removed_keys,
            "type_changes": self.type_changes,
            "format_change": self.format_change,
            "total_changes": self.total_changes,
            "text_diff": self.text_diff,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
        }


class StructuralAnalyzer:
    """Deterministic structural comparison of LLM outputs."""

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _flatten_keys(obj: Any, prefix: str = "") -> Set[str]:
        """Recursively collect all dot-notated keys from a JSON object."""
        keys: Set[str] = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.add(full_key)
                keys |= StructuralAnalyzer._flatten_keys(v, full_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                keys |= StructuralAnalyzer._flatten_keys(item, f"{prefix}[{i}]")
        return keys

    @staticmethod
    def _get_type_map(obj: Any, prefix: str = "") -> Dict[str, str]:
        """Return a mapping of key → type name."""
        types: Dict[str, str] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                types[full_key] = type(v).__name__
                types.update(StructuralAnalyzer._get_type_map(v, full_key))
        return types

    @classmethod
    def analyze(cls, old_output: str, new_output: str) -> StructuralDiff:
        """Compare two outputs and return a StructuralDiff."""
        diff = StructuralDiff()

        old_json = cls._try_parse_json(old_output)
        new_json = cls._try_parse_json(new_output)

        old_is_json = old_json is not None
        new_is_json = new_json is not None

        if old_is_json != new_is_json:
            diff.format_change = (
                f"{'json' if old_is_json else 'text'}->{'json' if new_is_json else 'text'}"
            )
            diff.has_schema_change = True
            diff.total_changes = 1
            return diff

        if not old_is_json:
            # Both are plain text – produce a line-level diff
            old_lines = old_output.splitlines(keepends=True)
            new_lines = new_output.splitlines(keepends=True)
            unified = list(
                difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile="old", tofile="new",
                    lineterm="",
                )
            )
            added = sum(1 for l in unified if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in unified if l.startswith("-") and not l.startswith("---"))
            diff.text_diff = unified
            diff.lines_added = added
            diff.lines_removed = removed
            diff.total_changes = added + removed
            return diff

        old_keys = cls._flatten_keys(old_json)
        new_keys = cls._flatten_keys(new_json)

        diff.added_keys = sorted(new_keys - old_keys)
        diff.removed_keys = sorted(old_keys - new_keys)

        # Type changes for keys that exist in both
        old_types = cls._get_type_map(old_json)
        new_types = cls._get_type_map(new_json)
        for key in old_keys & new_keys:
            if key in old_types and key in new_types and old_types[key] != new_types[key]:
                diff.type_changes[key] = f"{old_types[key]}->{new_types[key]}"

        diff.has_schema_change = bool(
            diff.added_keys or diff.removed_keys or diff.type_changes
        )
        diff.total_changes = (
            len(diff.added_keys) + len(diff.removed_keys) + len(diff.type_changes)
        )
        return diff
