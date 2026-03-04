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

    def _make_snapshot(self, prompt_id="greet", version="v1", output="Hello", model="gpt-4"):
        return PromptSnapshot(
            prompt_id=prompt_id,
            version=version,
            content="Say hello",
            output=output,
            model=model,
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

    def test_delete_prompt(self):
        self.store.save_snapshot(self._make_snapshot(version="v1"))
        self.store.save_snapshot(self._make_snapshot(version="v2"))
        count = self.store.delete_prompt("greet")
        assert count == 2
        assert self.store.list_snapshots("greet") == []

    def test_delete_prompt_not_found(self):
        count = self.store.delete_prompt("nonexistent")
        assert count == 0

    def test_list_snapshots_numeric_order(self):
        """Snapshots should be returned in numeric version order (v1 < v2 < v10)."""
        for v in ("v1", "v10", "v2", "v3"):
            self.store.save_snapshot(self._make_snapshot(version=v))
        versions = [s.version for s in self.store.list_snapshots("greet")]
        assert versions == ["v1", "v2", "v3", "v10"]


class TestSnapshotSearch:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SnapshotStore(base_dir=self.tmpdir)
        # Add a variety of snapshots
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="alpha", version="v1", content="alpha content",
            output="alpha output gpt", model="gpt-4", temperature=0.5,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="beta", version="v1", content="beta content",
            output="beta output claude", model="claude-3", temperature=0.7,
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        ))
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="alpha", version="v2", content="alpha content updated",
            output="new alpha output", model="gpt-4", temperature=0.5,
            timestamp=datetime(2024, 12, 1, tzinfo=timezone.utc),
        ))

    def test_search_all(self):
        results = self.store.search_snapshots()
        assert len(results) == 3

    def test_search_by_prompt_id(self):
        results = self.store.search_snapshots(prompt_id="alpha")
        assert len(results) == 2
        assert all(s.prompt_id == "alpha" for s in results)

    def test_search_by_model(self):
        results = self.store.search_snapshots(model="claude")
        assert len(results) == 1
        assert results[0].prompt_id == "beta"

    def test_search_by_keyword_in_output(self):
        results = self.store.search_snapshots(keyword="claude")
        assert len(results) == 1
        assert results[0].prompt_id == "beta"

    def test_search_by_keyword_in_content(self):
        results = self.store.search_snapshots(keyword="updated")
        assert len(results) == 1
        assert results[0].version == "v2"

    def test_search_since(self):
        since = datetime(2024, 3, 1, tzinfo=timezone.utc)
        results = self.store.search_snapshots(since=since)
        # Should exclude alpha/v1 from Jan 2024
        assert all(s.timestamp >= since for s in results)

    def test_search_until(self):
        until = datetime(2024, 3, 1, tzinfo=timezone.utc)
        results = self.store.search_snapshots(until=until)
        assert all(s.timestamp <= until for s in results)
        assert len(results) == 1

    def test_search_no_match(self):
        results = self.store.search_snapshots(model="nonexistent-model")
        assert results == []


class TestSnapshotStats:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SnapshotStore(base_dir=self.tmpdir)

    def test_stats_empty(self):
        stats = self.store.get_stats()
        assert stats["total_prompts"] == 0
        assert stats["total_snapshots"] == 0

    def test_stats_with_data(self):
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="a", version="v1", content="c", output="hello world",
            model="gpt-4", temperature=0.5,
        ))
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="b", version="v1", content="c", output="hi",
            model="claude-3", temperature=0.0,
        ))
        stats = self.store.get_stats()
        assert stats["total_prompts"] == 2
        assert stats["total_snapshots"] == 2
        assert "gpt-4" in stats["models"]
        assert "claude-3" in stats["models"]
        assert stats["models"]["gpt-4"] == 1
        assert isinstance(stats["avg_output_length"], float)


class TestExportImport:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SnapshotStore(base_dir=self.tmpdir)
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="greet", version="v1", content="hi", output="Hello!",
            model="gpt-4", temperature=0.0,
        ))
        self.store.save_snapshot(PromptSnapshot(
            prompt_id="greet", version="v2", content="hi", output="Hey there!",
            model="gpt-4", temperature=0.0,
        ))

    def test_export_creates_zip(self):
        import zipfile as zf
        dest = os.path.join(self.tmpdir, "archive.zip")
        count = self.store.export_archive(dest)
        assert count == 2
        assert os.path.isfile(dest)
        with zf.ZipFile(dest) as z:
            names = z.namelist()
        assert len(names) == 2

    def test_import_restores_snapshots(self):
        dest = os.path.join(self.tmpdir, "archive.zip")
        self.store.export_archive(dest)

        # Import into a fresh store
        new_dir = tempfile.mkdtemp()
        new_store = SnapshotStore(base_dir=new_dir)
        count = new_store.import_archive(dest)
        assert count == 2
        snaps = new_store.list_snapshots("greet")
        assert len(snaps) == 2

    def test_import_skip_existing(self):
        dest = os.path.join(self.tmpdir, "archive.zip")
        self.store.export_archive(dest)
        # Import into same store → should skip (overwrite=False)
        count = self.store.import_archive(dest, overwrite=False)
        assert count == 0

    def test_import_overwrite(self):
        dest = os.path.join(self.tmpdir, "archive.zip")
        self.store.export_archive(dest)
        count = self.store.import_archive(dest, overwrite=True)
        assert count == 2


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

