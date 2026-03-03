"""Hallucination detection: entity validation against context and old output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Common English words that are often capitalised (sentence-start, titles,
# UI strings, etc.) but should NOT be treated as named entities.
# ---------------------------------------------------------------------------

_COMMON_CAPITALIZED = {
    "The", "A", "An", "In", "On", "At", "To", "For", "With", "By",
    "From", "Of", "And", "Or", "But", "Not", "This", "That", "It",
    "Its", "He", "She", "They", "We", "You", "I", "Me", "Him", "Her",
    "Us", "Them", "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would", "Shall",
    "Should", "May", "Might", "Must", "Can", "Could", "Also", "As",
    "All", "Any", "Each", "Every", "Some", "When", "While", "Where",
    "Who", "What", "Which", "How", "If", "Then", "There", "Here",
    "So", "Than", "Into", "About", "Both", "After", "Before", "During",
    "Between", "Through", "Over", "Under", "Above", "Below",
    "Given", "Since", "Unless", "Until", "Within", "Without",
    "Following", "Using", "Based", "According", "Per", "Via",
    "Such", "Other", "These", "Those", "More", "Most", "Less",
    "First", "Last", "Next", "New", "Old", "Same", "Another",
    "Each", "No", "Our", "Their", "Your", "My", "His",
}

# Sentence boundary: period/question/exclamation followed by whitespace.
_SENTENCE_END = re.compile(r'[.!?]\s+')


def _sentence_start_positions(text: str) -> Set[int]:
    """Return character positions of the first word in each sentence."""
    positions: Set[int] = {0}
    for m in _SENTENCE_END.finditer(text):
        pos = m.end()
        if pos < len(text):
            positions.add(pos)
    return positions


def _extract_entities(text: str) -> Set[str]:
    """Extract potential named entities, filtering sentence-start capitalization.

    Rules applied (in order):
    1. Multi-word capitalised phrases (e.g. "John Smith", "United Nations") –
       always treated as entities.
    2. Single ALL-CAPS acronyms of 2+ chars (e.g. "AI", "NASA") –
       always treated as entities.
    3. Single capitalised words that do NOT appear at a sentence start and
       are NOT in the common-word exclusion list – treated as entities.
    """
    sentence_starts = _sentence_start_positions(text)
    entities: Set[str] = set()

    # Multi-word capitalised phrases
    for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", text):
        entities.add(m.group(1))

    # ALL-CAPS acronyms (2+ chars)
    for m in re.finditer(r"\b([A-Z]{2,})\b", text):
        entities.add(m.group(1))

    # Single capitalised words (not sentence-start, not common)
    for m in re.finditer(r"\b([A-Z][a-z]+)\b", text):
        word = m.group(1)
        if word in _COMMON_CAPITALIZED:
            continue
        if m.start() in sentence_starts:
            continue
        entities.add(word)

    return entities


def _extract_numeric_claims(text: str) -> List[str]:
    """Extract numeric claims (dates, percentages, dollar amounts, quantities).

    These are high-risk for hallucination because they are specific and
    verifiable, yet easy for an LLM to fabricate.
    """
    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",           # dates: 01/15/2024
        r"\b\d{4}-\d{2}-\d{2}\b",                   # dates: 2024-01-15
        (
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
        ),
        r"\b\d+(?:\.\d+)?%",                         # percentages
        r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b",  # money
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion)\b",
        r"\b\d+(?:\.\d+)?\s*(?:km|miles?|kg|lbs?|mph|years?|months?|days?|hours?)\b",
    ]
    claims: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            claims.append(m.group(0).strip())
    return claims


@dataclass
class HallucinationRisk:
    """Result of hallucination risk analysis."""

    risk_level: str          # "low" | "medium" | "high"
    confidence: float        # 0.0 – 1.0
    new_entities: List[str]  # entities appearing in new output not in old
    unsupported_entities: List[str]  # new entities not in context
    all_entities_new: List[str] = field(default_factory=list)
    new_numeric_claims: List[str] = field(default_factory=list)  # new numeric claims
    unsupported_numeric_claims: List[str] = field(default_factory=list)  # not in context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 2),
            "new_entities": self.new_entities,
            "unsupported_entities": self.unsupported_entities,
            "new_numeric_claims": self.new_numeric_claims,
            "unsupported_numeric_claims": self.unsupported_numeric_claims,
        }


class HallucinationDetector:
    """Entity- and numeric-claim-based hallucination risk detection."""

    @staticmethod
    def detect(
        new_output: str,
        old_output: str,
        context: Optional[str] = None,
    ) -> HallucinationRisk:
        """
        Detect hallucination risk by comparing named entities and numeric claims.

        new_entities  = entities in new_output not present in old_output
        unsupported   = new_entities not mentioned in context (if provided)

        Numeric claims (dates, percentages, amounts) are tracked separately
        because they are high-risk for hallucination even when the entity set
        looks stable.
        """
        new_ents = _extract_entities(new_output)
        old_ents = _extract_entities(old_output)
        context_ents = _extract_entities(context) if context else set()

        introduced = sorted(new_ents - old_ents)

        if context is not None:
            unsupported = sorted(e for e in introduced if e not in context_ents)
        else:
            unsupported = []

        # Numeric claims
        new_num = set(_extract_numeric_claims(new_output))
        old_num = set(_extract_numeric_claims(old_output))
        context_num = set(_extract_numeric_claims(context)) if context else set()

        new_numeric = sorted(new_num - old_num)
        unsupported_numeric: List[str] = []
        if context is not None:
            unsupported_numeric = sorted(n for n in new_numeric if n not in context_num)
        else:
            unsupported_numeric = []

        # Risk scoring – numeric claims carry higher weight than named entities
        # because fabricated numbers are more dangerous than novel entity names.
        high_risk_count = len(unsupported) + len(unsupported_numeric) * 2
        medium_risk_count = len(introduced) + len(new_numeric)

        if high_risk_count >= 2:
            risk_level = "high"
            confidence = 0.80
        elif high_risk_count == 1 or medium_risk_count >= 3:
            risk_level = "medium"
            confidence = 0.65
        elif medium_risk_count >= 1:
            risk_level = "medium"
            confidence = 0.55
        else:
            risk_level = "low"
            confidence = 0.90

        return HallucinationRisk(
            risk_level=risk_level,
            confidence=confidence,
            new_entities=introduced,
            unsupported_entities=unsupported,
            all_entities_new=sorted(new_ents),
            new_numeric_claims=new_numeric,
            unsupported_numeric_claims=unsupported_numeric,
        )
