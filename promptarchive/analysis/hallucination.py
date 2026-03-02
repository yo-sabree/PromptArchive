"""Hallucination detection: entity validation against context and old output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Simple NER: extract capitalized phrases as named entities
# ---------------------------------------------------------------------------

def _extract_entities(text: str) -> Set[str]:
    """Extract potential named entities (capitalized multi-word phrases or acronyms)."""
    # Match capitalized words / sequences (excluding sentence-start heuristic)
    tokens = re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", text)
    # Also match acronyms
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    entities: Set[str] = set()
    for t in tokens:
        if len(t) > 1:  # skip single letters
            entities.add(t.strip())
    for a in acronyms:
        entities.add(a)
    return entities


@dataclass
class HallucinationRisk:
    """Result of hallucination risk analysis."""

    risk_level: str          # "low" | "medium" | "high"
    confidence: float        # 0.0 – 1.0
    new_entities: List[str]  # entities appearing in new output not in old
    unsupported_entities: List[str]  # new entities not in context
    all_entities_new: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 2),
            "new_entities": self.new_entities,
            "unsupported_entities": self.unsupported_entities,
        }


class HallucinationDetector:
    """Simple entity-based hallucination risk detection."""

    @staticmethod
    def detect(
        new_output: str,
        old_output: str,
        context: Optional[str] = None,
    ) -> HallucinationRisk:
        """
        Detect hallucination risk by comparing named entities.

        new_entities  = entities in new_output not present in old_output
        unsupported   = new_entities not mentioned in context (if provided)
        """
        new_ents = _extract_entities(new_output)
        old_ents = _extract_entities(old_output)
        context_ents = _extract_entities(context) if context else set()

        introduced = sorted(new_ents - old_ents)

        if context is not None:
            unsupported = sorted(e for e in introduced if e not in context_ents)
        else:
            unsupported = []

        # Risk scoring
        if unsupported:
            risk_level = "high" if len(unsupported) >= 2 else "medium"
            confidence = 0.75
        elif introduced:
            risk_level = "medium"
            confidence = 0.60
        else:
            risk_level = "low"
            confidence = 0.85

        return HallucinationRisk(
            risk_level=risk_level,
            confidence=confidence,
            new_entities=introduced,
            unsupported_entities=unsupported,
            all_entities_new=sorted(new_ents),
        )
