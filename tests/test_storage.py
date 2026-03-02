"""Tests for storage and CLI."""

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone

from promptarchive.core.prompt import Prompt, PromptSnapshot, Constraint
from promptarchive.core.registry import PromptRegistry
from promptarchive.storage.snapshots import SnapshotStore, init_archive


class TestSnapshotStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SnapshotStore(base_dir=self.tmpdir)

    def _make_snapshot(self, prompt_id="greet", version="v1", output="Hello"):
        return PromptSnapshot(
            prompt_id=prompt_id,
            version=version,
            content="Say hello",
            output=output,
            model="gpt-4",
            temperature=0.7,
        )

    def test_save_and_load(self):
        s = self._make_snapshot()
        self.store.save_snapshot(s)
        loaded = self.store.load_snapshot("greet", "v1")
        assert loaded is not None
        assert loaded.output == "Hello"
        assert loaded.model == "gpt-4"

    def test_list_snapshots(self):
        self.store.save_snapshot(self._make_snapshot(version="v1", output="Hi"))
        self.store.save_snapshot(self._make_snapshot(version="v2", output="Hello"))
        snaps = self.store.list_snapshots("greet")
        assert len(snaps) == 2

    def test_list_prompt_ids(self):
        self.store.save_snapshot(self._make_snapshot(prompt_id="a"))
        self.store.save_snapshot(self._make_snapshot(prompt_id="b"))
        ids = self.store.list_prompt_ids()
        assert "a" in ids
        assert "b" in ids

    def test_delete_snapshot(self):
        self.store.save_snapshot(self._make_snapshot())
        result = self.store.delete_snapshot("greet", "v1")
        assert result
        assert self.store.load_snapshot("greet", "v1") is None

    def test_load_nonexistent(self):
        assert self.store.load_snapshot("missing", "v1") is None


class TestPromptRegistry:
    def test_register_and_get(self):
        registry = PromptRegistry()
        p = Prompt(id="test", name="Test", content="Hello")
        registry.register(p)
        assert registry.get("test") is p

    def test_list_prompts(self):
        registry = PromptRegistry()
        registry.register(Prompt(id="b", name="B", content="b"))
        registry.register(Prompt(id="a", name="A", content="a"))
        prompts = registry.list_prompts()
        assert [p.id for p in prompts] == ["a", "b"]

    def test_remove(self):
        registry = PromptRegistry()
        registry.register(Prompt(id="x", name="X", content="x"))
        assert registry.remove("x")
        assert registry.get("x") is None

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "registry.json")
        registry = PromptRegistry(registry_path=path)
        c = Constraint(name="formal", tone="formal")
        p = Prompt(id="greet", name="Greet", content="Hello", constraints=[c])
        registry.register(p)
        registry.save()

        registry2 = PromptRegistry(registry_path=path)
        loaded = registry2.get("greet")
        assert loaded is not None
        assert loaded.name == "Greet"
        assert loaded.constraints[0].tone == "formal"


class TestInitArchive:
    def test_creates_structure(self):
        tmpdir = tempfile.mkdtemp()
        base = os.path.join(tmpdir, ".promptarchive")
        result = init_archive(base)
        assert os.path.isdir(result)
        assert os.path.isdir(os.path.join(result, "snapshots"))
        assert os.path.isfile(os.path.join(result, ".gitignore"))
        assert os.path.isfile(os.path.join(result, "README.md"))
