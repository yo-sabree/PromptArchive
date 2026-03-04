"""Core data models: Prompt, PromptSnapshot, Constraint."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    """Behavioral rule applied to a prompt's output."""

    name: str
    must_include: Optional[List[str]] = None
    must_not_include: Optional[List[str]] = None
    regex_patterns: Optional[List[str]] = None
    max_length: Optional[int] = None
    tone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "must_include": self.must_include,
            "must_not_include": self.must_not_include,
            "regex_patterns": self.regex_patterns,
            "max_length": self.max_length,
            "tone": self.tone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            name=data["name"],
            must_include=data.get("must_include"),
            must_not_include=data.get("must_not_include"),
            regex_patterns=data.get("regex_patterns"),
            max_length=data.get("max_length"),
            tone=data.get("tone"),
        )


@dataclass
class PromptSnapshot:
    """A versioned snapshot of a prompt's output at a specific point in time."""

    prompt_id: str
    version: str
    content: str
    output: str
    model: str
    temperature: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "output": self.output,
            "model": self.model,
            "temperature": self.temperature,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "constraints": [c.to_dict() for c in self.constraints],
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSnapshot":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        elif ts is None:
            ts = datetime.now(timezone.utc)
        return cls(
            prompt_id=data["prompt_id"],
            version=data["version"],
            content=data["content"],
            output=data["output"],
            model=data["model"],
            temperature=float(data.get("temperature", 0.0)),
            timestamp=ts,
            metadata=data.get("metadata", {}),
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            context=data.get("context"),
        )


@dataclass
class Prompt:
    """A versioned prompt with behavioral constraints and snapshot history."""

    id: str
    name: str
    content: str
    description: Optional[str] = None
    constraints: List[Constraint] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    snapshots: List[PromptSnapshot] = field(default_factory=list)

    def add_snapshot(
        self,
        output: str,
        model: str,
        temperature: float = 0.0,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptSnapshot:
        """Create and append a new versioned snapshot of this prompt's output."""
        version = f"v{len(self.snapshots) + 1}"
        snapshot = PromptSnapshot(
            prompt_id=self.id,
            version=version,
            content=self.content,
            output=output,
            model=model,
            temperature=temperature,
            context=context,
            metadata=metadata or {},
            constraints=list(self.constraints),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
            "tags": self.tags,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        return cls(
            id=data["id"],
            name=data["name"],
            content=data["content"],
            description=data.get("description"),
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            tags=data.get("tags", []),
            snapshots=[PromptSnapshot.from_dict(s) for s in data.get("snapshots", [])],
        )
