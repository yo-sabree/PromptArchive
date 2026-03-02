"""Tests for core data models: Prompt, PromptSnapshot, Constraint."""

import pytest
from datetime import datetime, timezone

from promptarchive.core.prompt import Constraint, Prompt, PromptSnapshot


class TestConstraint:
    def test_basic_creation(self):
        c = Constraint(name="test")
        assert c.name == "test"
        assert c.must_include is None
        assert c.tone is None

    def test_to_dict_roundtrip(self):
        c = Constraint(
            name="formal",
            must_include=["hello"],
            must_not_include=["bye"],
            max_length=100,
            tone="formal",
        )
        d = c.to_dict()
        c2 = Constraint.from_dict(d)
        assert c2.name == c.name
        assert c2.must_include == c.must_include
        assert c2.tone == c.tone


class TestPromptSnapshot:
    def test_basic_creation(self):
        s = PromptSnapshot(
            prompt_id="greet",
            version="v1",
            content="Hello {name}",
            output="Hello World",
            model="gpt-4",
            temperature=0.7,
        )
        assert s.version == "v1"
        assert s.model == "gpt-4"

    def test_to_dict_roundtrip(self):
        s = PromptSnapshot(
            prompt_id="greet",
            version="v1",
            content="Hello",
            output="Hi there",
            model="test-model",
            temperature=0.5,
            context="some context",
            metadata={"key": "val"},
        )
        d = s.to_dict()
        s2 = PromptSnapshot.from_dict(d)
        assert s2.prompt_id == "greet"
        assert s2.version == "v1"
        assert s2.context == "some context"
        assert s2.metadata == {"key": "val"}


class TestPrompt:
    def test_add_snapshot(self):
        p = Prompt(id="test", name="Test Prompt", content="Say hello")
        s = p.add_snapshot(output="Hello!", model="gpt-4", temperature=0.7)
        assert s.version == "v1"
        assert s.prompt_id == "test"
        assert len(p.snapshots) == 1

        s2 = p.add_snapshot(output="Hi!", model="gpt-4", temperature=0.7)
        assert s2.version == "v2"
        assert len(p.snapshots) == 2

    def test_to_dict_roundtrip(self):
        c = Constraint(name="formal", tone="formal")
        p = Prompt(
            id="my_prompt",
            name="My Prompt",
            content="Do something",
            description="Test prompt",
            constraints=[c],
            tags=["test"],
        )
        p.add_snapshot(output="Result", model="gpt-4", temperature=0.5)

        d = p.to_dict()
        p2 = Prompt.from_dict(d)
        assert p2.id == "my_prompt"
        assert len(p2.constraints) == 1
        assert len(p2.snapshots) == 1
        assert p2.snapshots[0].version == "v1"
