"""Factual drift: compare output against a reference answer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _word_set(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", _normalize(text)))


@dataclass
class FactualDrift:
    """Result of factual comparison against a reference answer."""

    has_reference: bool
    overlap_score: float        # 0.0 – 1.0 (word-level F1)
    missing_facts: List[str]    # important words in reference missing from output
    extra_facts: List[str]      # words in output not in reference
    drift_level: str            # "none" | "minor" | "moderate" | "major"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_reference": self.has_reference,
            "overlap_score": round(self.overlap_score, 4),
            "missing_facts": self.missing_facts,
            "extra_facts": self.extra_facts,
            "drift_level": self.drift_level,
        }


# Stop words to filter from "important" words
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "about", "as", "into",
    "and", "or", "but", "not", "this", "that", "it", "its", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "us", "them",
}


def _important_words(text: str) -> Set[str]:
    return _word_set(text) - _STOP_WORDS


class FactualAnalyzer:
    """Word-overlap factual comparison against a reference answer."""

    @staticmethod
    def analyze(
        output: str, reference: Optional[str]
    ) -> FactualDrift:
        if reference is None:
            return FactualDrift(
                has_reference=False,
                overlap_score=0.0,
                missing_facts=[],
                extra_facts=[],
                drift_level="none",
            )

        ref_words = _important_words(reference)
        out_words = _important_words(output)

        if not ref_words and not out_words:
            return FactualDrift(
                has_reference=True,
                overlap_score=1.0,
                missing_facts=[],
                extra_facts=[],
                drift_level="none",
            )

        intersection = ref_words & out_words
        precision = len(intersection) / len(out_words) if out_words else 0.0
        recall = len(intersection) / len(ref_words) if ref_words else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        missing = sorted(ref_words - out_words)
        extra = sorted(out_words - ref_words)

        if f1 >= 0.90:
            drift_level = "none"
        elif f1 >= 0.70:
            drift_level = "minor"
        elif f1 >= 0.50:
            drift_level = "moderate"
        else:
            drift_level = "major"

        return FactualDrift(
            has_reference=True,
            overlap_score=f1,
            missing_facts=missing[:10],  # cap for readability
            extra_facts=extra[:10],
            drift_level=drift_level,
        )
