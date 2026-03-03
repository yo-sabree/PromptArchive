"""Report generation: human-readable text and JSON output."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from promptarchive.analysis.engine import RegressionResult

# ANSI color codes – only applied when writing to a real terminal.
_USE_COLOR = sys.stdout.isatty()

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_RED     = "\033[31m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_CYAN    = "\033[36m"
_WHITE   = "\033[37m"
_DIM     = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes when colour is enabled."""
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + _RESET


# Risk level colours
_RISK_COLORS = {
    "low":      _GREEN,
    "medium":   _YELLOW,
    "high":     _RED,
    "critical": _RED + _BOLD,
}


def _risk_str(level: str) -> str:
    color = _RISK_COLORS.get(level, _WHITE)
    return _c(level.upper(), color)


class RegressionReport:
    """Generate human-readable and machine-readable reports from a RegressionResult."""

    @staticmethod
    def generate_text(result: RegressionResult) -> str:
        """Return a formatted text regression report with git-style diff."""
        lines = []

        # ── Header ──────────────────────────────────────────────────────────
        lines.append(_c(f"REGRESSION REPORT: {result.prompt_id}", _BOLD))
        lines.append("")
        lines.append(
            f"Comparing: {_c(result.old_version, _DIM)} → {_c(result.new_version, _BOLD)}"
        )
        lines.append(f"Risk Level: {_risk_str(result.risk_level)}")
        lines.append(f"Has Changes: {result.has_changes}")
        lines.append("")

        # ── Gating verdict ──────────────────────────────────────────────────
        if result.gating is not None:
            g = result.gating
            verdict = (
                _c("✓ GATE PASSED", _GREEN, _BOLD)
                if g.passed
                else _c("✗ GATE FAILED", _RED, _BOLD)
            )
            lines.append(f"GATING VERDICT (config v{g.config_version}): {verdict}")
            for lr in g.layer_results:
                if lr.passed:
                    mark = _c("  ✓", _GREEN)
                    detail = ""
                else:
                    mark = _c("  ✗", _RED)
                    detail = f" — {lr.reason}"
                lines.append(f"{mark} {lr.layer}{detail}")
            lines.append("")

        # ── Layer 1: Semantic ────────────────────────────────────────────────
        sem = result.semantic
        lines.append(_c("LAYER 1: SEMANTIC SIMILARITY", _BOLD))
        lines.append(f"  Similarity Score: {sem.similarity_score * 100:.2f}%")
        lines.append(
            f"  Lexical Precision: {sem.precision_score * 100:.2f}%"
            + ("  ⚠  precision collapse" if sem.precision_score < 0.70 else "")
        )
        lines.append(f"  Lexical Recall:    {sem.recall_score * 100:.2f}%")
        lines.append(f"  Drift Level: {sem.drift_level}")
        lines.append(f"  Direction: {sem.direction}")
        lines.append(f"  Method: {sem.method}")
        lines.append("")

        # ── Layer 2: Structural ──────────────────────────────────────────────
        struct = result.structural
        lines.append(_c("LAYER 2: STRUCTURAL DIFF", _BOLD))
        lines.append(f"  Schema Changes: {struct.has_schema_change}")
        if struct.format_change:
            lines.append(f"  Format Change: {struct.format_change}")
        if struct.added_keys:
            lines.append(f"  Added Keys: {', '.join(struct.added_keys)}")
        if struct.removed_keys:
            lines.append(f"  Removed Keys: {', '.join(struct.removed_keys)}")
        if struct.type_changes:
            for k, v in struct.type_changes.items():
                lines.append(f"  Type Change [{k}]: {v}")
        if struct.text_diff:
            # Git-like diff block
            lines.append(
                f"  Lines: {_c(f'+{struct.lines_added}', _GREEN)} "
                f"{_c(f'-{struct.lines_removed}', _RED)}"
            )
            lines.append("  " + _c("─" * 58, _DIM))
            for dl in struct.text_diff[:80]:  # cap for readability
                dl_stripped = dl.rstrip("\n")
                if dl_stripped.startswith("+") and not dl_stripped.startswith("+++"):
                    lines.append("  " + _c(dl_stripped, _GREEN))
                elif dl_stripped.startswith("-") and not dl_stripped.startswith("---"):
                    lines.append("  " + _c(dl_stripped, _RED))
                elif dl_stripped.startswith("@@"):
                    lines.append("  " + _c(dl_stripped, _CYAN))
                else:
                    lines.append("  " + dl_stripped)
            if len(struct.text_diff) > 80:
                lines.append(
                    _c(f"  … {len(struct.text_diff) - 80} more diff lines omitted", _DIM)
                )
        else:
            lines.append(f"  Total Changes: {struct.total_changes}")
        lines.append("")

        # ── Layer 3: Tone ────────────────────────────────────────────────────
        tone = result.tone
        lines.append(_c("LAYER 3: TONE SHIFT", _BOLD))
        lines.append(f"  Formality:      {_fmt_delta(tone.formality_delta)}%")
        lines.append(f"  Sentiment:      {_fmt_delta(tone.sentiment_delta)}%")
        lines.append(f"  Assertiveness:  {_fmt_delta(tone.assertiveness_delta)}%")
        lines.append(f"  Reading Grade:  {_fmt_delta(tone.reading_grade_delta)}")
        lines.append("")

        # ── Layer 4: Constraints ─────────────────────────────────────────────
        cons = result.constraints
        v_count = cons.violation_count
        status = (
            _c("✓ All passed", _GREEN)
            if cons.passed
            else _c(f"✗ {v_count} violation{'s' if v_count != 1 else ''}", _RED)
        )
        lines.append(_c("LAYER 4: CONSTRAINTS", _BOLD))
        lines.append(f"  Status: {status}")
        for v in cons.violations:
            lines.append(
                f"    {_c('✗', _RED)} {v.constraint_name} ({v.constraint_type}): {v.message}"
            )
        lines.append("")

        # ── Layer 5: Hallucination ───────────────────────────────────────────
        hal = result.hallucination
        hal_color = _GREEN if hal.risk_level == "low" else (
            _YELLOW if hal.risk_level == "medium" else _RED
        )
        lines.append(_c("LAYER 5: HALLUCINATION RISK", _BOLD))
        lines.append(f"  Risk Level: {_c(hal.risk_level, hal_color)}")
        lines.append(f"  Confidence: {hal.confidence * 100:.0f}%")
        lines.append(f"  New Named Entities:     {len(hal.new_entities)}")
        lines.append(f"  Unsupported Entities:   {len(hal.unsupported_entities)}")
        if hal.unsupported_entities:
            lines.append(
                f"    {_c('⚠', _YELLOW)} {', '.join(hal.unsupported_entities)}"
            )
        lines.append(f"  New Numeric Claims:     {len(hal.new_numeric_claims)}")
        lines.append(
            f"  Unsupported Num Claims: {len(hal.unsupported_numeric_claims)}"
        )
        if hal.unsupported_numeric_claims:
            lines.append(
                f"    {_c('⚠', _YELLOW)} {', '.join(hal.unsupported_numeric_claims)}"
            )
        lines.append("")

        # ── Layer 6: Factual ─────────────────────────────────────────────────
        fact = result.factual
        lines.append(_c("LAYER 6: FACTUAL DRIFT", _BOLD))
        if not fact.has_reference:
            lines.append("  Status: No reference answer provided")
        else:
            lines.append(f"  Overlap Score: {fact.overlap_score * 100:.2f}%")
            lines.append(f"  Drift Level: {fact.drift_level}")
            if fact.missing_facts:
                lines.append(
                    f"  {_c('Missing facts:', _RED)} {', '.join(fact.missing_facts[:5])}"
                )
            if fact.extra_facts:
                lines.append(
                    f"  {_c('Extra content:', _YELLOW)} {', '.join(fact.extra_facts[:5])}"
                )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_json(result: RegressionResult) -> str:
        """Return a JSON string of the full regression result."""
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def _fmt_delta(value: float) -> str:
    """Format a numeric delta with sign and colour."""
    s = f"+{value:.1f}" if value >= 0 else f"{value:.1f}"
    if _USE_COLOR:
        color = _GREEN if value >= 0 else _RED
        return _c(s, color)
    return s
