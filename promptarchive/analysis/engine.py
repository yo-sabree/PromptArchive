"""Analysis engine: orchestrates all 6 analysis layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptarchive.analysis.constraints import ConstraintResult, ConstraintValidator
from promptarchive.analysis.factual import FactualDrift, FactualAnalyzer
from promptarchive.analysis.hallucination import HallucinationRisk, HallucinationDetector
from promptarchive.analysis.semantic import SemanticResult, SemanticAnalyzer
from promptarchive.analysis.structural import StructuralDiff, StructuralAnalyzer
from promptarchive.analysis.tone import ToneShift, ToneAnalyzer
from promptarchive.core.prompt import PromptSnapshot


@dataclass
class RegressionResult:
    """Aggregated result of all 6 analysis layers."""

    prompt_id: str
    old_version: str
    new_version: str
    risk_level: str  # "low" | "medium" | "high" | "critical"
    has_changes: bool

    # Layer results
    semantic: SemanticResult
    structural: StructuralDiff
    tone: ToneShift
    constraints: ConstraintResult
    hallucination: HallucinationRisk
    factual: FactualDrift

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "risk_level": self.risk_level,
            "has_changes": self.has_changes,
            "semantic": self.semantic.to_dict(),
            "structural": self.structural.to_dict(),
            "tone": self.tone.to_dict(),
            "constraints": self.constraints.to_dict(),
            "hallucination": self.hallucination.to_dict(),
            "factual": self.factual.to_dict(),
        }


def _compute_risk(
    semantic: SemanticResult,
    structural: StructuralDiff,
    constraints: ConstraintResult,
    hallucination: HallucinationRisk,
    factual: FactualDrift,
) -> str:
    """Determine overall risk level from layer results."""
    score = 0

    # Semantic drift
    drift_scores = {"none": 0, "minimal": 1, "moderate": 2, "significant": 3, "extreme": 4}
    score += drift_scores.get(semantic.drift_level, 0)

    # Structural
    if structural.has_schema_change:
        score += 2

    # Constraint violations
    score += constraints.violation_count * 2

    # Hallucination
    hal_scores = {"low": 0, "medium": 1, "high": 3}
    score += hal_scores.get(hallucination.risk_level, 0)

    # Factual
    fact_scores = {"none": 0, "minor": 1, "moderate": 2, "major": 4}
    if factual.has_reference:
        score += fact_scores.get(factual.drift_level, 0)

    if score == 0:
        return "low"
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    if score <= 9:
        return "high"
    return "critical"


class AnalysisEngine:
    """Orchestrates all 6 analysis layers to produce a RegressionResult."""

    def analyze(
        self,
        old_snapshot: PromptSnapshot,
        new_snapshot: PromptSnapshot,
        reference_answer: Optional[str] = None,
    ) -> RegressionResult:
        """Run all analysis layers and return a consolidated RegressionResult."""

        semantic = SemanticAnalyzer.analyze(old_snapshot.output, new_snapshot.output)
        structural = StructuralAnalyzer.analyze(old_snapshot.output, new_snapshot.output)
        tone = ToneAnalyzer.compare(old_snapshot.output, new_snapshot.output)
        constraints = ConstraintValidator.validate(
            new_snapshot.output, new_snapshot.constraints
        )
        hallucination = HallucinationDetector.detect(
            new_output=new_snapshot.output,
            old_output=old_snapshot.output,
            context=new_snapshot.context,
        )
        factual = FactualAnalyzer.analyze(new_snapshot.output, reference_answer)

        risk_level = _compute_risk(semantic, structural, constraints, hallucination, factual)

        has_changes = (
            semantic.drift_level not in ("none", "minimal")
            or structural.has_schema_change
            or not constraints.passed
            or hallucination.risk_level != "low"
            or (factual.has_reference and factual.drift_level != "none")
        )

        return RegressionResult(
            prompt_id=new_snapshot.prompt_id,
            old_version=old_snapshot.version,
            new_version=new_snapshot.version,
            risk_level=risk_level,
            has_changes=has_changes,
            semantic=semantic,
            structural=structural,
            tone=tone,
            constraints=constraints,
            hallucination=hallucination,
            factual=factual,
        )
