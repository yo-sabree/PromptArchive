"""Constraint validation: check LLM outputs against user-defined rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from promptarchive.core.prompt import Constraint


_CASUAL_WORDS = {
    "hey", "hi", "yeah", "yep", "nope", "gonna", "wanna", "gotta",
    "kinda", "sorta", "basically", "literally", "awesome", "cool", "lol",
    "omg", "btw", "tbh", "idk", "fyi",
}

_FORMAL_WORDS = {
    "furthermore", "therefore", "consequently", "nevertheless", "hereby",
    "henceforth", "wherein", "pursuant", "aforementioned", "subsequent",
}


@dataclass
class ConstraintViolation:
    constraint_name: str
    constraint_type: str  # "must_include" | "must_not_include" | "regex" | "length" | "tone"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "constraint_type": self.constraint_type,
            "message": self.message,
        }


@dataclass
class ConstraintResult:
    """Result of validating an output against all constraints."""

    passed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "violations": [v.to_dict() for v in self.violations],
        }


class ConstraintValidator:
    """Validate an LLM output against a list of Constraint objects."""

    @staticmethod
    def _check_tone(text: str, expected_tone: str) -> bool:
        lower = text.lower()
        words = set(re.findall(r"\b\w+\b", lower))
        if expected_tone == "formal":
            casual_hits = words & _CASUAL_WORDS
            return len(casual_hits) == 0
        if expected_tone == "casual":
            formal_hits = words & _FORMAL_WORDS
            return len(formal_hits) == 0
        return True  # unknown tone – pass

    @classmethod
    def validate(
        cls, output: str, constraints: List[Constraint]
    ) -> ConstraintResult:
        violations: List[ConstraintViolation] = []

        for constraint in constraints:
            # must_include
            if constraint.must_include:
                for phrase in constraint.must_include:
                    if phrase.lower() not in output.lower():
                        violations.append(
                            ConstraintViolation(
                                constraint_name=constraint.name,
                                constraint_type="must_include",
                                message=f"Required phrase not found: '{phrase}'",
                            )
                        )

            # must_not_include
            if constraint.must_not_include:
                for phrase in constraint.must_not_include:
                    if phrase.lower() in output.lower():
                        violations.append(
                            ConstraintViolation(
                                constraint_name=constraint.name,
                                constraint_type="must_not_include",
                                message=f"Forbidden phrase found: '{phrase}'",
                            )
                        )

            # regex_patterns
            if constraint.regex_patterns:
                for pattern in constraint.regex_patterns:
                    try:
                        if not re.search(pattern, output, re.DOTALL):
                            violations.append(
                                ConstraintViolation(
                                    constraint_name=constraint.name,
                                    constraint_type="regex",
                                    message=f"Output does not match pattern: {pattern}",
                                )
                            )
                    except re.error as exc:
                        violations.append(
                            ConstraintViolation(
                                constraint_name=constraint.name,
                                constraint_type="regex",
                                message=f"Invalid regex pattern '{pattern}': {exc}",
                            )
                        )

            # max_length
            if constraint.max_length is not None and len(output) > constraint.max_length:
                violations.append(
                    ConstraintViolation(
                        constraint_name=constraint.name,
                        constraint_type="length",
                        message=(
                            f"Output length {len(output)} exceeds limit {constraint.max_length}"
                        ),
                    )
                )

            # tone
            if constraint.tone is not None:
                if not cls._check_tone(output, constraint.tone):
                    violations.append(
                        ConstraintViolation(
                            constraint_name=constraint.name,
                            constraint_type="tone",
                            message=f"Detected {'casual' if constraint.tone == 'formal' else 'formal'} language",
                        )
                    )

        return ConstraintResult(passed=len(violations) == 0, violations=violations)
