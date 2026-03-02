"""Local storage for PromptSnapshots with optional Git integration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List, Optional

from promptarchive.core.prompt import PromptSnapshot

_DEFAULT_BASE = ".promptarchive"
_SNAPSHOTS_DIR = "snapshots"
_GIT_GITIGNORE = ".gitignore"


class SnapshotStore:
    """Persist and retrieve PromptSnapshots in a local directory."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = os.path.abspath(base_dir or _DEFAULT_BASE)
        self.snapshots_dir = os.path.join(self.base_dir, _SNAPSHOTS_DIR)
        os.makedirs(self.snapshots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _prompt_dir(self, prompt_id: str) -> str:
        safe_id = _safe_name(prompt_id)
        path = os.path.join(self.snapshots_dir, safe_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _snapshot_path(self, snapshot: PromptSnapshot) -> str:
        safe_version = _safe_name(snapshot.version)
        return os.path.join(
            self._prompt_dir(snapshot.prompt_id),
            f"{safe_version}.json",
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: PromptSnapshot) -> str:
        """Persist a snapshot to disk. Returns the file path."""
        path = self._snapshot_path(snapshot)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    def load_snapshot(self, prompt_id: str, version: str) -> Optional[PromptSnapshot]:
        """Load a single snapshot by prompt ID and version. Returns None if not found."""
        safe_id = _safe_name(prompt_id)
        safe_ver = _safe_name(version)
        path = os.path.join(self.snapshots_dir, safe_id, f"{safe_ver}.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return PromptSnapshot.from_dict(json.load(fh))

    def list_snapshots(self, prompt_id: str) -> List[PromptSnapshot]:
        """Return all snapshots for a prompt, ordered by version string."""
        safe_id = _safe_name(prompt_id)
        prompt_dir = os.path.join(self.snapshots_dir, safe_id)
        if not os.path.isdir(prompt_dir):
            return []
        snapshots = []
        for fname in sorted(os.listdir(prompt_dir)):
            if fname.endswith(".json"):
                path = os.path.join(prompt_dir, fname)
                with open(path, encoding="utf-8") as fh:
                    snapshots.append(PromptSnapshot.from_dict(json.load(fh)))
        return snapshots

    def list_prompt_ids(self) -> List[str]:
        """Return all prompt IDs that have at least one snapshot."""
        if not os.path.isdir(self.snapshots_dir):
            return []
        return sorted(
            entry
            for entry in os.listdir(self.snapshots_dir)
            if os.path.isdir(os.path.join(self.snapshots_dir, entry))
        )

    def delete_snapshot(self, prompt_id: str, version: str) -> bool:
        """Delete a snapshot file. Returns True if deleted, False if not found."""
        safe_id = _safe_name(prompt_id)
        safe_ver = _safe_name(version)
        path = os.path.join(self.snapshots_dir, safe_id, f"{safe_ver}.json")
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def git_add(self, snapshot: PromptSnapshot) -> bool:
        """Stage the snapshot file in Git. Returns True on success."""
        path = self._snapshot_path(snapshot)
        return _git_add(path)

    def git_commit(self, message: str) -> bool:
        """Commit staged changes. Returns True on success."""
        return _git_commit(message)


# ---------------------------------------------------------------------------
# Init helper
# ---------------------------------------------------------------------------

def init_archive(base_dir: Optional[str] = None) -> str:
    """Initialize a .promptarchive directory with Git support."""
    base = os.path.abspath(base_dir or _DEFAULT_BASE)
    os.makedirs(os.path.join(base, _SNAPSHOTS_DIR), exist_ok=True)

    # Write a .gitignore so temporary files are not tracked
    gitignore_path = os.path.join(base, _GIT_GITIGNORE)
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as fh:
            fh.write("# PromptArchive internal files\n*.tmp\n")

    # Write README
    readme_path = os.path.join(base, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as fh:
            fh.write(
                "# .promptarchive\n\nGenerated by PromptArchive. "
                "Track this directory with Git for version control.\n"
            )

    return base


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert a prompt ID or version to a filesystem-safe string."""
    import re
    return re.sub(r"[^\w.\-]", "_", name)


def _run_git(args: List[str], cwd: Optional[str] = None) -> bool:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _git_add(path: str) -> bool:
    return _run_git(["add", path])


def _git_commit(message: str) -> bool:
    return _run_git(["commit", "-m", message])
