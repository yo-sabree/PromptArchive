"""Report generation: human-readable text and JSON output."""

from __future__ import annotations

import json
from typing import Any, Dict

from promptarchive.analysis.engine import RegressionResult


class RegressionReport:
    """Generate human-readable and machine-readable reports from a RegressionResult."""

    @staticmethod
    def generate_text(result: RegressionResult) -> str:
        """Return a formatted text regression report."""
        lines = []
        lines.append(f"REGRESSION REPORT: {result.prompt_id}")
        lines.append("")
        lines.append(f"Comparing: {result.old_version} \u2192 {result.new_version}")
        lines.append(f"Risk Level: {result.risk_level.upper()}")
        lines.append(f"Has Changes: {result.has_changes}")
        lines.append("")

        # Layer 1: Semantic
        sem = result.semantic
        lines.append("LAYER 1: SEMANTIC SIMILARITY")
        lines.append(f"  Similarity Score: {sem.similarity_score * 100:.2f}%")
        lines.append(f"  Drift Level: {sem.drift_level}")
        lines.append(f"  Direction: {sem.direction}")
        lines.append(f"  Method: {sem.method}")
        lines.append("")

        # Layer 2: Structural
        struct = result.structural
        lines.append("LAYER 2: STRUCTURAL ANALYSIS")
        lines.append(f"  Schema Changes: {struct.has_schema_change}")
        lines.append(f"  Total Changes: {struct.total_changes}")
        if struct.added_keys:
            lines.append(f"  Added Keys: {', '.join(struct.added_keys)}")
        if struct.removed_keys:
            lines.append(f"  Removed Keys: {', '.join(struct.removed_keys)}")
        if struct.format_change:
            lines.append(f"  Format Change: {struct.format_change}")
        lines.append("")

        # Layer 3: Tone
        tone = result.tone
        lines.append("LAYER 3: TONE SHIFT")
        lines.append(f"  Formality: {_fmt_delta(tone.formality_delta)}%")
        lines.append(f"  Sentiment: {_fmt_delta(tone.sentiment_delta)}%")
        lines.append(f"  Assertiveness: {_fmt_delta(tone.assertiveness_delta)}%")
        lines.append(f"  Reading Grade: {_fmt_delta(tone.reading_grade_delta)}")
        lines.append("")

        # Layer 4: Constraints
        cons = result.constraints
        v_count = cons.violation_count
        status = f"\u2713 All passed" if cons.passed else f"\u2717 {v_count} violation{'s' if v_count != 1 else ''}"
        lines.append("LAYER 4: CONSTRAINTS")
        lines.append(f"  Status: {status}")
        for v in cons.violations:
            lines.append(f"    - {v.constraint_name} ({v.constraint_type}): {v.message}")
        lines.append("")

        # Layer 5: Hallucination
        hal = result.hallucination
        lines.append("LAYER 5: HALLUCINATION RISK")
        lines.append(f"  Risk Level: {hal.risk_level}")
        lines.append(f"  Confidence: {hal.confidence * 100:.0f}%")
        lines.append(f"  New Entities: {len(hal.new_entities)}")
        lines.append(f"  Unsupported: {len(hal.unsupported_entities)}")
        if hal.unsupported_entities:
            lines.append(f"  Unsupported Entities: {', '.join(hal.unsupported_entities)}")
        lines.append("")

        # Layer 6: Factual
        fact = result.factual
        lines.append("LAYER 6: FACTUAL DRIFT")
        if not fact.has_reference:
            lines.append("  Status: No reference answer provided")
        else:
            lines.append(f"  Overlap Score: {fact.overlap_score * 100:.2f}%")
            lines.append(f"  Drift Level: {fact.drift_level}")
            if fact.missing_facts:
                lines.append(f"  Missing: {', '.join(fact.missing_facts[:5])}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_json(result: RegressionResult) -> str:
        """Return a JSON string of the full regression result."""
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def _fmt_delta(value: float) -> str:
    """Format a numeric delta with sign."""
    if value >= 0:
        return f"+{value:.1f}"
    return f"{value:.1f}"
