"""PromptRegistry – manages a collection of prompts."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from promptarchive.core.prompt import Prompt


class PromptRegistry:
    """In-memory registry of prompts, with optional persistence to disk."""

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._prompts: Dict[str, Prompt] = {}
        self.registry_path = registry_path

        if registry_path and os.path.isfile(registry_path):
            self._load(registry_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, prompt: Prompt) -> None:
        """Add or replace a prompt in the registry."""
        self._prompts[prompt.id] = prompt

    def get(self, prompt_id: str) -> Optional[Prompt]:
        """Return a prompt by ID, or None if not found."""
        return self._prompts.get(prompt_id)

    def list_prompts(self) -> List[Prompt]:
        """Return all registered prompts sorted by ID."""
        return sorted(self._prompts.values(), key=lambda p: p.id)

    def remove(self, prompt_id: str) -> bool:
        """Remove a prompt by ID. Returns True if removed, False if not found."""
        if prompt_id in self._prompts:
            del self._prompts[prompt_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the registry to a JSON file."""
        target = path or self.registry_path
        if not target:
            raise ValueError("No registry_path specified.")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(
                {pid: p.to_dict() for pid, p in self._prompts.items()},
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for pid, pdata in data.items():
            self._prompts[pid] = Prompt.from_dict(pdata)

    def __len__(self) -> int:
        return len(self._prompts)
