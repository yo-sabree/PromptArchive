"""Tone analysis: formality, sentiment, assertiveness, reading grade."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Word lists (heuristic)
# ---------------------------------------------------------------------------

_FORMAL_MARKERS = {
    "furthermore", "therefore", "consequently", "nevertheless", "hereby",
    "henceforth", "wherein", "pursuant", "aforementioned", "subsequent",
    "notwithstanding", "hitherto", "therein", "heretofore", "accordingly",
    "thus", "hence", "moreover", "additionally", "however",
}

_CASUAL_MARKERS = {
    "hey", "hi", "yeah", "yep", "nope", "gonna", "wanna", "gotta",
    "kinda", "sorta", "basically", "literally", "awesome", "cool", "lol",
    "omg", "btw", "tbh", "idk", "fyi", "stuff", "things", "like",
}

_POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "fantastic", "amazing",
    "outstanding", "superb", "positive", "beneficial", "effective",
    "successful", "helpful", "useful", "valuable", "perfect",
}

_NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "negative",
    "ineffective", "harmful", "useless", "failure", "wrong", "error",
    "problem", "issue", "concern", "risk", "danger",
}

_ASSERTIVE_MARKERS = {
    "must", "will", "shall", "require", "demand", "insist", "ensure",
    "guarantee", "absolutely", "certainly", "definitely", "clearly",
    "obviously", "undoubtedly", "always", "never",
}

_HEDGING_MARKERS = {
    "maybe", "perhaps", "might", "could", "possibly", "probably",
    "seems", "appears", "suggest", "consider", "think", "believe",
    "assume", "estimate", "approximately", "roughly",
}


@dataclass
class ToneProfile:
    """Heuristic tone scores for a single text."""

    formality_score: float    # 0 (very casual) – 100 (very formal)
    sentiment_score: float    # -100 (negative) – 100 (positive)
    assertiveness_score: float  # 0 – 100
    reading_grade: float      # Flesch-Kincaid grade

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formality_score": round(self.formality_score, 1),
            "sentiment_score": round(self.sentiment_score, 1),
            "assertiveness_score": round(self.assertiveness_score, 1),
            "reading_grade": round(self.reading_grade, 1),
        }


@dataclass
class ToneShift:
    """Delta between two ToneProfiles."""

    formality_delta: float
    sentiment_delta: float
    assertiveness_delta: float
    reading_grade_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formality_delta": round(self.formality_delta, 1),
            "sentiment_delta": round(self.sentiment_delta, 1),
            "assertiveness_delta": round(self.assertiveness_delta, 1),
            "reading_grade_delta": round(self.reading_grade_delta, 1),
        }


def _count_syllables(word: str) -> int:
    """Very rough syllable count."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_kincaid_grade(text: str) -> float:
    """Estimate Flesch-Kincaid grade level."""
    sentences = max(1, len(re.split(r"[.!?]+", text)))
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / sentences  # average sentence length
    asw = syllables / len(words)  # average syllables per word
    grade = 0.39 * asl + 11.8 * asw - 15.59
    return max(0.0, grade)


class ToneAnalyzer:
    """Heuristic tone classification for LLM outputs."""

    @classmethod
    def analyze(cls, text: str) -> ToneProfile:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return ToneProfile(50.0, 0.0, 50.0, 0.0)
        word_set = set(words)
        total = len(words)

        # Formality (0–100)
        formal_count = len(word_set & _FORMAL_MARKERS)
        casual_count = len(word_set & _CASUAL_MARKERS)
        formality = 50.0 + 50.0 * (formal_count - casual_count) / max(1, total / 10)
        formality = max(0.0, min(100.0, formality))

        # Sentiment (-100 to 100)
        pos_count = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
        sentiment = 100.0 * (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
        sentiment = max(-100.0, min(100.0, sentiment))

        # Assertiveness (0–100)
        assert_count = sum(1 for w in words if w in _ASSERTIVE_MARKERS)
        hedge_count = sum(1 for w in words if w in _HEDGING_MARKERS)
        assertiveness = 50.0 + 50.0 * (assert_count - hedge_count) / max(1, total / 10)
        assertiveness = max(0.0, min(100.0, assertiveness))

        reading_grade = _flesch_kincaid_grade(text)

        return ToneProfile(
            formality_score=formality,
            sentiment_score=sentiment,
            assertiveness_score=assertiveness,
            reading_grade=reading_grade,
        )

    @classmethod
    def compare(cls, old_text: str, new_text: str) -> ToneShift:
        """Return the delta between two tone profiles."""
        old_profile = cls.analyze(old_text)
        new_profile = cls.analyze(new_text)
        return ToneShift(
            formality_delta=new_profile.formality_score - old_profile.formality_score,
            sentiment_delta=new_profile.sentiment_score - old_profile.sentiment_score,
            assertiveness_delta=new_profile.assertiveness_score - old_profile.assertiveness_score,
            reading_grade_delta=new_profile.reading_grade - old_profile.reading_grade,
        )
