"""Local storage for PromptSnapshots with optional Git integration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from promptarchive.core.prompt import PromptSnapshot

_DEFAULT_BASE = ".promptarchive"
_SNAPSHOTS_DIR = "snapshots"
_GIT_GITIGNORE = ".gitignore"


class SnapshotStore:
    """Persist and retrieve PromptSnapshots in a local directory."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir or _DEFAULT_BASE)
        self.snapshots_dir = os.path.join(self.base_dir, _SNAPSHOTS_DIR)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self._cipher = _build_cipher(passphrase) if passphrase else None

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
        data = snapshot.to_dict()
        raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if self._cipher:
            enc_path = path + ".enc"
            with open(enc_path, "wb") as fh:
                fh.write(self._cipher.encrypt(raw))
            return enc_path
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw.decode("utf-8"))
        return path

    def load_snapshot(self, prompt_id: str, version: str) -> Optional[PromptSnapshot]:
        """Load a single snapshot by prompt ID and version. Returns None if not found."""
        safe_id = _safe_name(prompt_id)
        safe_ver = _safe_name(version)
        base = os.path.join(self.snapshots_dir, safe_id, f"{safe_ver}.json")
        if self._cipher and os.path.isfile(base + ".enc"):
            with open(base + ".enc", "rb") as fh:
                raw = self._cipher.decrypt(fh.read())
            return PromptSnapshot.from_dict(json.loads(raw.decode("utf-8")))
        if os.path.isfile(base):
            with open(base, encoding="utf-8") as fh:
                return PromptSnapshot.from_dict(json.load(fh))
        return None

    def list_snapshots(self, prompt_id: str) -> List[PromptSnapshot]:
        """Return all snapshots for a prompt, ordered by version string."""
        safe_id = _safe_name(prompt_id)
        prompt_dir = os.path.join(self.snapshots_dir, safe_id)
        if not os.path.isdir(prompt_dir):
            return []
        snapshots = []
        for fname in sorted(os.listdir(prompt_dir)):
            snapshot = self._load_file(os.path.join(prompt_dir, fname))
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def _load_file(self, path: str) -> Optional[PromptSnapshot]:
        """Load a snapshot from *path* (plain JSON or encrypted .enc)."""
        if path.endswith(".json.enc") and self._cipher:
            with open(path, "rb") as fh:
                raw = self._cipher.decrypt(fh.read())
            return PromptSnapshot.from_dict(json.loads(raw.decode("utf-8")))
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                return PromptSnapshot.from_dict(json.load(fh))
        return None

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
        base = os.path.join(self.snapshots_dir, safe_id, f"{safe_ver}.json")
        for candidate in (base + ".enc", base):
            if os.path.isfile(candidate):
                _secure_delete(candidate)
                return True
        return False

    def delete_prompt(self, prompt_id: str) -> int:
        """Delete all snapshots for a prompt. Returns the number of files removed."""
        safe_id = _safe_name(prompt_id)
        prompt_dir = os.path.join(self.snapshots_dir, safe_id)
        if not os.path.isdir(prompt_dir):
            return 0
        count = 0
        for fname in os.listdir(prompt_dir):
            fpath = os.path.join(prompt_dir, fname)
            if os.path.isfile(fpath):
                _secure_delete(fpath)
                count += 1
        shutil.rmtree(prompt_dir, ignore_errors=True)
        return count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_snapshots(
        self,
        prompt_id: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        keyword: Optional[str] = None,
    ) -> List[PromptSnapshot]:
        """Search snapshots across all prompts with optional filters.

        Parameters
        ----------
        prompt_id:
            Restrict to a specific prompt ID (exact match).
        model:
            Case-insensitive substring match against the model field.
        since / until:
            Filter by timestamp (inclusive).  Both are timezone-aware if
            the stored timestamps are timezone-aware.
        keyword:
            Case-insensitive substring searched in both *content* and *output*.
        """
        ids = [prompt_id] if prompt_id else self.list_prompt_ids()
        results: List[PromptSnapshot] = []
        kw = keyword.lower() if keyword else None
        mdl = model.lower() if model else None
        for pid in ids:
            for snap in self.list_snapshots(pid):
                if mdl and mdl not in snap.model.lower():
                    continue
                ts = snap.timestamp
                if since and ts < since:
                    continue
                if until and ts > until:
                    continue
                if kw and kw not in snap.content.lower() and kw not in snap.output.lower():
                    continue
                results.append(snap)
        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all stored snapshots."""
        all_snapshots = self.search_snapshots()
        if not all_snapshots:
            return {
                "total_prompts": 0,
                "total_snapshots": 0,
                "models": {},
                "oldest": None,
                "newest": None,
                "avg_output_length": 0.0,
            }
        prompt_ids = set(s.prompt_id for s in all_snapshots)
        models: Dict[str, int] = {}
        for s in all_snapshots:
            models[s.model] = models.get(s.model, 0) + 1
        timestamps = sorted(s.timestamp for s in all_snapshots)
        avg_len = sum(len(s.output) for s in all_snapshots) / len(all_snapshots)
        return {
            "total_prompts": len(prompt_ids),
            "total_snapshots": len(all_snapshots),
            "models": models,
            "oldest": timestamps[0].isoformat(),
            "newest": timestamps[-1].isoformat(),
            "avg_output_length": round(avg_len, 1),
        }

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_archive(self, path: str) -> int:
        """Export all snapshots to a ZIP archive at *path*.

        Returns the number of snapshot files written.
        """
        count = 0
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pid in self.list_prompt_ids():
                safe_id = _safe_name(pid)
                prompt_dir = os.path.join(self.snapshots_dir, safe_id)
                for fname in sorted(os.listdir(prompt_dir)):
                    fpath = os.path.join(prompt_dir, fname)
                    if os.path.isfile(fpath):
                        arcname = os.path.join("snapshots", safe_id, fname)
                        zf.write(fpath, arcname)
                        count += 1
        return count

    def import_archive(self, path: str, overwrite: bool = False) -> int:
        """Import snapshots from a ZIP archive created by :meth:`export_archive`.

        Returns the number of snapshots imported.  Existing files are skipped
        unless *overwrite* is ``True``.
        """
        count = 0
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                dest = os.path.join(self.base_dir, member)
                if not overwrite and os.path.exists(dest):
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                count += 1
        return count

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


def _secure_delete(path: str) -> None:
    """Overwrite a file with zeros before deleting it.

    This is a best-effort measure; it does not guarantee secure erasure on
    SSDs or copy-on-write filesystems, but it reduces accidental exposure.
    Writes are done in 64 KB chunks to keep memory usage constant for large files.
    """
    _CHUNK = 64 * 1024  # 64 KB
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            remaining = size
            while remaining > 0:
                fh.write(b"\x00" * min(_CHUNK, remaining))
                remaining -= _CHUNK
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    os.remove(path)


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


# ---------------------------------------------------------------------------
# Optional AES-256-GCM encryption
# ---------------------------------------------------------------------------


class _AESCipher:
    """AES-256-GCM authenticated encryption backed by the ``cryptography`` package."""

    _SALT_LEN = 16
    _NONCE_LEN = 12
    # 200 000 iterations follows OWASP PBKDF2 guidelines (2023) for SHA-256.
    # It balances security (brute-force resistance) with acceptable startup cost
    # (~0.1 s on modern hardware), appropriate for an interactive CLI tool.
    _ITERATIONS = 200_000

    def __init__(self, passphrase: str) -> None:
        self._passphrase = passphrase.encode("utf-8")

    def _derive_key(self, salt: bytes) -> bytes:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
        from cryptography.hazmat.primitives import hashes  # type: ignore

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._ITERATIONS,
        )
        return kdf.derive(self._passphrase)

    def encrypt(self, plaintext: bytes) -> bytes:
        import os as _os
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        salt = _os.urandom(self._SALT_LEN)
        nonce = _os.urandom(self._NONCE_LEN)
        key = self._derive_key(salt)
        ct = AESGCM(key).encrypt(nonce, plaintext, None)
        return salt + nonce + ct

    def decrypt(self, ciphertext: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        salt = ciphertext[: self._SALT_LEN]
        nonce = ciphertext[self._SALT_LEN : self._SALT_LEN + self._NONCE_LEN]
        ct = ciphertext[self._SALT_LEN + self._NONCE_LEN :]
        key = self._derive_key(salt)
        return AESGCM(key).decrypt(nonce, ct, None)


def _build_cipher(passphrase: str) -> Optional["_AESCipher"]:
    """Return an :class:`_AESCipher` if ``cryptography`` is available."""
    try:
        import cryptography  # noqa: F401  # type: ignore
        return _AESCipher(passphrase)
    except ImportError:
        import warnings
        warnings.warn(
            "The 'cryptography' package is not installed. "
            "Install promptarchive[secure] to enable encryption.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
