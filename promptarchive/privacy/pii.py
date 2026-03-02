"""PII (Personally Identifiable Information) detection and redaction.

All processing is performed locally using regex patterns — no external calls,
no network access, no telemetry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PII patterns  (label, compiled regex, redaction placeholder)
# ---------------------------------------------------------------------------

_PATTERN_SPECS: List[Tuple[str, str, str]] = [
    (
        "email",
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
    ),
    (
        "phone",
        # US-style: optional country code, optional area code, 7-digit core
        r"\b(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}\b",
        "[PHONE]",
    ),
    (
        "ssn",
        # US Social Security Number  NNN-NN-NNNN or NNN NN NNNN
        # Note: this pattern may produce false positives for date-like sequences;
        # treat findings as candidates for human review rather than definitive matches.
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "[SSN]",
    ),
    (
        "credit_card",
        # Visa / MC / Amex / Discover (no spaces)
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}"
        r"|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "[CREDIT_CARD]",
    ),
    (
        "ip_address",
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "[IP_ADDRESS]",
    ),
    (
        "api_key",
        # Common API-key-like patterns: prefix + long alphanumeric token
        r"\b(?:sk|pk|api|secret|token)[-_][A-Za-z0-9]{16,}\b",
        "[API_KEY]",
    ),
]

_COMPILED: List[Tuple[str, re.Pattern[str], str]] = [
    (label, re.compile(pattern), placeholder)
    for label, pattern, placeholder in _PATTERN_SPECS
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PIIFinding:
    """A single PII occurrence found in the scanned text."""

    label: str   # e.g. "email", "phone", "ssn"
    value: str   # the matched text
    start: int   # character start index
    end: int     # character end index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class PIIReport:
    """Result of scanning a piece of text for PII."""

    has_pii: bool
    findings: List[PIIFinding] = field(default_factory=list)
    redacted_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_pii": self.has_pii,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PIIDetector:
    """Locally scan and optionally redact PII from text.

    Uses only Python standard library (``re``); no external API calls.
    """

    @staticmethod
    def scan(text: str) -> PIIReport:
        """Return a :class:`PIIReport` listing every PII occurrence found."""
        findings: List[PIIFinding] = []
        for label, pattern, _ in _COMPILED:
            for match in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        label=label,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return PIIReport(has_pii=bool(findings), findings=findings)

    @staticmethod
    def redact(text: str) -> str:
        """Return a copy of *text* with all detected PII replaced by placeholders."""
        for _label, pattern, placeholder in _COMPILED:
            text = pattern.sub(placeholder, text)
        return text
