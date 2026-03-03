"""Gating logic: deterministic pass/fail evaluation with configurable thresholds.

Production regression pipelines need more than diagnostics – they need a
clear, versioned pass/fail verdict so that CI/CD gates can block bad prompt
changes automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from promptarchive.analysis.engine import RegressionResult

# Default lists used by GatingThresholds – defined at module level so that
# dataclass field(default_factory=...) can reference them without lambdas.
_DEFAULT_HAL_FAIL_LEVELS = ["high"]
_DEFAULT_FACTUAL_FAIL_LEVELS = ["major"]
_DEFAULT_OVERALL_FAIL_LEVELS = ["high", "critical"]


@dataclass
class GatingThresholds:
    """Configurable thresholds for each analysis layer.

    All fields have sensible defaults that mirror the existing qualitative
    labels, so existing workflows continue to work without any configuration.
    A different profile (e.g. ``strict``) can be loaded from a JSON file.
    """

    # Semantic: minimum cosine/embedding similarity to pass
    min_semantic_similarity: float = 0.70
    # Semantic: maximum allowed lexical-precision drop
    # (precision_score below 1 - max_precision_drop triggers failure)
    max_precision_drop: float = 0.30

    # Hallucination: maximum allowed unsupported named entities
    max_unsupported_entities: int = 0
    # Hallucination: risk levels that trigger hard failure
    hallucination_fail_levels: List[str] = field(
        default_factory=lambda: list(_DEFAULT_HAL_FAIL_LEVELS)
    )
    # Hallucination: maximum allowed unsupported numeric claims
    max_unsupported_numeric_claims: int = 0

    # Factual: minimum word-overlap F1 score (when reference is provided)
    min_factual_overlap: float = 0.70
    # Factual: drift levels that trigger failure
    factual_fail_levels: List[str] = field(
        default_factory=lambda: list(_DEFAULT_FACTUAL_FAIL_LEVELS)
    )

    # Constraints: any violation triggers failure
    fail_on_constraint_violation: bool = True

    # Structural: schema changes trigger failure
    fail_on_schema_change: bool = False

    # Overall risk levels that cause the gate to fail regardless of layers
    overall_fail_levels: List[str] = field(
        default_factory=lambda: list(_DEFAULT_OVERALL_FAIL_LEVELS)
    )

    # Version identifier for audit / reproducibility
    config_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "min_semantic_similarity": self.min_semantic_similarity,
            "max_precision_drop": self.max_precision_drop,
            "max_unsupported_entities": self.max_unsupported_entities,
            "hallucination_fail_levels": self.hallucination_fail_levels,
            "max_unsupported_numeric_claims": self.max_unsupported_numeric_claims,
            "min_factual_overlap": self.min_factual_overlap,
            "factual_fail_levels": self.factual_fail_levels,
            "fail_on_constraint_violation": self.fail_on_constraint_violation,
            "fail_on_schema_change": self.fail_on_schema_change,
            "overall_fail_levels": self.overall_fail_levels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatingThresholds":
        return cls(
            config_version=data.get("config_version", "1.0"),
            min_semantic_similarity=data.get("min_semantic_similarity", 0.70),
            max_precision_drop=data.get("max_precision_drop", 0.30),
            max_unsupported_entities=data.get("max_unsupported_entities", 0),
            hallucination_fail_levels=data.get(
                "hallucination_fail_levels", ["high"]
            ),
            max_unsupported_numeric_claims=data.get(
                "max_unsupported_numeric_claims", 0
            ),
            min_factual_overlap=data.get("min_factual_overlap", 0.70),
            factual_fail_levels=data.get("factual_fail_levels", ["major"]),
            fail_on_constraint_violation=data.get(
                "fail_on_constraint_violation", True
            ),
            fail_on_schema_change=data.get("fail_on_schema_change", False),
            overall_fail_levels=data.get(
                "overall_fail_levels", ["high", "critical"]
            ),
        )


@dataclass
class LayerGateResult:
    """Pass/fail verdict for a single analysis layer."""

    layer: str
    passed: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class GatingResult:
    """Aggregated pass/fail verdict from all analysis layers."""

    passed: bool
    layer_results: List[LayerGateResult] = field(default_factory=list)
    config_version: str = "1.0"

    @property
    def failed_layers(self) -> List[LayerGateResult]:
        return [r for r in self.layer_results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "config_version": self.config_version,
            "layer_results": [r.to_dict() for r in self.layer_results],
        }


class GateEvaluator:
    """Apply ``GatingThresholds`` to a ``RegressionResult`` for a verdict."""

    def __init__(self, thresholds: Optional[GatingThresholds] = None) -> None:
        self.thresholds = thresholds or GatingThresholds()

    def evaluate(self, result: "RegressionResult") -> GatingResult:
        """Evaluate all layers and return a ``GatingResult``."""
        layer_results: List[LayerGateResult] = []

        t = self.thresholds

        # ── Semantic ────────────────────────────────────────────────────────
        sem = result.semantic
        sem_reasons: List[str] = []
        if sem.similarity_score < t.min_semantic_similarity:
            sem_reasons.append(
                f"similarity {sem.similarity_score:.3f} < {t.min_semantic_similarity:.3f}"
            )
        precision_drop = 1.0 - sem.precision_score
        if precision_drop > t.max_precision_drop:
            sem_reasons.append(
                f"precision drop {precision_drop:.3f} > {t.max_precision_drop:.3f}"
            )
        layer_results.append(
            LayerGateResult(
                layer="semantic",
                passed=not sem_reasons,
                reason="; ".join(sem_reasons) if sem_reasons else None,
            )
        )

        # ── Structural ──────────────────────────────────────────────────────
        struct = result.structural
        struct_fail = t.fail_on_schema_change and struct.has_schema_change
        layer_results.append(
            LayerGateResult(
                layer="structural",
                passed=not struct_fail,
                reason=(
                    f"schema change: {struct.format_change or 'key/type changes'}"
                    if struct_fail
                    else None
                ),
            )
        )

        # ── Constraints ─────────────────────────────────────────────────────
        cons = result.constraints
        cons_fail = t.fail_on_constraint_violation and not cons.passed
        layer_results.append(
            LayerGateResult(
                layer="constraints",
                passed=not cons_fail,
                reason=(
                    f"{cons.violation_count} constraint violation(s)"
                    if cons_fail
                    else None
                ),
            )
        )

        # ── Hallucination ───────────────────────────────────────────────────
        hal = result.hallucination
        hal_reasons: List[str] = []
        if hal.risk_level in t.hallucination_fail_levels:
            hal_reasons.append(f"risk level: {hal.risk_level}")
        if len(hal.unsupported_entities) > t.max_unsupported_entities:
            hal_reasons.append(
                f"{len(hal.unsupported_entities)} unsupported entities "
                f"(max {t.max_unsupported_entities})"
            )
        if len(hal.unsupported_numeric_claims) > t.max_unsupported_numeric_claims:
            hal_reasons.append(
                f"{len(hal.unsupported_numeric_claims)} unsupported numeric claim(s) "
                f"(max {t.max_unsupported_numeric_claims})"
            )
        layer_results.append(
            LayerGateResult(
                layer="hallucination",
                passed=not hal_reasons,
                reason="; ".join(hal_reasons) if hal_reasons else None,
            )
        )

        # ── Factual ─────────────────────────────────────────────────────────
        fact = result.factual
        if fact.has_reference:
            fact_reasons: List[str] = []
            if fact.overlap_score < t.min_factual_overlap:
                fact_reasons.append(
                    f"overlap {fact.overlap_score:.3f} < {t.min_factual_overlap:.3f}"
                )
            if fact.drift_level in t.factual_fail_levels:
                fact_reasons.append(f"drift level: {fact.drift_level}")
            layer_results.append(
                LayerGateResult(
                    layer="factual",
                    passed=not fact_reasons,
                    reason="; ".join(fact_reasons) if fact_reasons else None,
                )
            )
        else:
            layer_results.append(
                LayerGateResult(
                    layer="factual",
                    passed=True,
                    reason="no reference provided",
                )
            )

        # ── Overall risk ────────────────────────────────────────────────────
        overall_fail = result.risk_level in t.overall_fail_levels
        layer_results.append(
            LayerGateResult(
                layer="overall",
                passed=not overall_fail,
                reason=(
                    f"overall risk level: {result.risk_level}" if overall_fail else None
                ),
            )
        )

        return GatingResult(
            passed=all(r.passed for r in layer_results),
            layer_results=layer_results,
            config_version=t.config_version,
        )
