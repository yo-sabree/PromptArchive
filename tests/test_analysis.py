"""Tests for analysis modules."""

import pytest

from promptarchive.analysis.structural import StructuralAnalyzer
from promptarchive.analysis.semantic import SemanticAnalyzer
from promptarchive.analysis.constraints import ConstraintValidator
from promptarchive.analysis.tone import ToneAnalyzer
from promptarchive.analysis.hallucination import HallucinationDetector
from promptarchive.analysis.factual import FactualAnalyzer
from promptarchive.core.prompt import Constraint


class TestStructuralAnalyzer:
    def test_identical_json(self):
        js = '{"key": "value", "num": 1}'
        diff = StructuralAnalyzer.analyze(js, js)
        assert not diff.has_schema_change
        assert diff.total_changes == 0

    def test_added_key(self):
        old = '{"a": 1}'
        new = '{"a": 1, "b": 2}'
        diff = StructuralAnalyzer.analyze(old, new)
        assert diff.has_schema_change
        assert "b" in diff.added_keys

    def test_removed_key(self):
        old = '{"a": 1, "b": 2}'
        new = '{"a": 1}'
        diff = StructuralAnalyzer.analyze(old, new)
        assert diff.has_schema_change
        assert "b" in diff.removed_keys

    def test_json_to_text_format_change(self):
        old = '{"key": "value"}'
        new = "plain text output"
        diff = StructuralAnalyzer.analyze(old, new)
        assert diff.has_schema_change
        assert "json->text" in (diff.format_change or "")

    def test_plain_text_no_diff(self):
        # Identical plain text → zero changes (and no schema change)
        diff = StructuralAnalyzer.analyze("hello world", "hello world")
        assert not diff.has_schema_change
        assert diff.total_changes == 0

    def test_plain_text_with_diff(self):
        # Different plain text → line-level diff with meaningful total_changes
        diff = StructuralAnalyzer.analyze("hello world", "goodbye world")
        assert not diff.has_schema_change  # no schema change for plain text
        assert diff.total_changes > 0      # but changes are now captured
        assert diff.lines_added >= 1
        assert diff.lines_removed >= 1
        assert any(l.startswith("+") and not l.startswith("+++") for l in diff.text_diff)
        assert any(l.startswith("-") and not l.startswith("---") for l in diff.text_diff)


class TestSemanticAnalyzer:
    def test_identical_texts(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = SemanticAnalyzer.analyze(text, text)
        assert result.similarity_score > 0.95
        assert result.drift_level == "none"

    def test_completely_different(self):
        result = SemanticAnalyzer.analyze(
            "The cat sat on the mat.",
            "Quantum mechanics describes the behavior of subatomic particles.",
        )
        assert result.similarity_score < 0.8

    def test_result_fields(self):
        result = SemanticAnalyzer.analyze("Hello world", "Hello earth")
        assert result.method in ("embeddings", "tfidf")
        assert result.direction in ("increased", "decreased", "stable")
        assert 0.0 <= result.similarity_score <= 1.0


class TestConstraintValidator:
    def test_must_include_pass(self):
        c = Constraint(name="req", must_include=["termination clause"])
        result = ConstraintValidator.validate(
            "This agreement includes a termination clause.", [c]
        )
        assert result.passed
        assert result.violation_count == 0

    def test_must_include_fail(self):
        c = Constraint(name="req", must_include=["termination clause"])
        result = ConstraintValidator.validate("This is a simple agreement.", [c])
        assert not result.passed
        assert result.violation_count == 1

    def test_must_not_include_fail(self):
        c = Constraint(name="no_legal", must_not_include=["legal advice"])
        result = ConstraintValidator.validate(
            "This document does not constitute legal advice.", [c]
        )
        assert not result.passed

    def test_max_length_pass(self):
        c = Constraint(name="len", max_length=100)
        result = ConstraintValidator.validate("Short text.", [c])
        assert result.passed

    def test_max_length_fail(self):
        c = Constraint(name="len", max_length=10)
        result = ConstraintValidator.validate("This is a longer text than allowed.", [c])
        assert not result.passed

    def test_regex_pass(self):
        c = Constraint(name="json", regex_patterns=[r"^\{.*\}$"])
        result = ConstraintValidator.validate('{"key": "val"}', [c])
        assert result.passed

    def test_regex_fail(self):
        c = Constraint(name="json", regex_patterns=[r"^\{.*\}$"])
        result = ConstraintValidator.validate("plain text", [c])
        assert not result.passed

    def test_formal_tone_pass(self):
        c = Constraint(name="tone", tone="formal")
        result = ConstraintValidator.validate(
            "Furthermore, the agreement shall be executed forthwith.", [c]
        )
        assert result.passed

    def test_formal_tone_fail(self):
        c = Constraint(name="tone", tone="formal")
        result = ConstraintValidator.validate("Hey yeah gonna do it lol", [c])
        assert not result.passed


class TestToneAnalyzer:
    def test_analyze_returns_profile(self):
        profile = ToneAnalyzer.analyze("Hello, this is a test sentence.")
        assert 0.0 <= profile.formality_score <= 100.0
        assert -100.0 <= profile.sentiment_score <= 100.0
        assert 0.0 <= profile.assertiveness_score <= 100.0
        assert profile.reading_grade >= 0.0

    def test_compare_returns_shift(self):
        shift = ToneAnalyzer.compare("Hello world.", "Furthermore, we must acknowledge.")
        assert isinstance(shift.formality_delta, float)
        assert isinstance(shift.reading_grade_delta, float)

    def test_formal_text_higher_formality(self):
        formal = "Furthermore, the aforementioned clause shall be enacted henceforth."
        casual = "Hey yeah gonna do this thing lol"
        formal_profile = ToneAnalyzer.analyze(formal)
        casual_profile = ToneAnalyzer.analyze(casual)
        assert formal_profile.formality_score > casual_profile.formality_score


class TestHallucinationDetector:
    def test_no_new_entities(self):
        risk = HallucinationDetector.detect(
            new_output="The CEO is John Doe.",
            old_output="The CEO is John Doe.",
        )
        assert risk.risk_level == "low"
        assert len(risk.new_entities) == 0

    def test_new_entity_without_context(self):
        risk = HallucinationDetector.detect(
            new_output="The CEO is Jane Smith.",
            old_output="The CEO is John Doe.",
        )
        assert risk.risk_level in ("medium", "high")
        assert "Jane" in risk.new_entities or "Jane Smith" in risk.new_entities

    def test_unsupported_entity_with_context(self):
        risk = HallucinationDetector.detect(
            new_output="The CEO is Jane Smith.",
            old_output="The CEO is John Doe.",
            context="John Doe is the CEO of the company.",
        )
        assert len(risk.unsupported_entities) > 0


class TestFactualAnalyzer:
    def test_no_reference(self):
        result = FactualAnalyzer.analyze("Some output.", None)
        assert not result.has_reference
        assert result.drift_level == "none"

    def test_identical_reference(self):
        text = "The contract includes a termination clause effective January."
        result = FactualAnalyzer.analyze(text, text)
        assert result.has_reference
        assert result.overlap_score > 0.9
        assert result.drift_level == "none"

    def test_major_drift(self):
        result = FactualAnalyzer.analyze(
            "Apples are red fruits.",
            "Quantum computers use qubits for computations.",
        )
        assert result.has_reference
        assert result.drift_level in ("moderate", "major")


class TestSemanticPrecisionRecall:
    def test_identical_precision_recall_one(self):
        text = "The cat sat on the mat in the sun."
        result = SemanticAnalyzer.analyze(text, text)
        assert result.precision_score == 1.0
        assert result.recall_score == 1.0

    def test_precision_collapse_detected(self):
        """High cosine similarity but low lexical precision should show in drift level."""
        old = "The contract terminates on June 30 and includes liability clauses."
        new = "A completely different topic about space and rockets."
        result = SemanticAnalyzer.analyze(old, new)
        # Precision should be low (few new words are in old)
        assert result.precision_score < 0.6

    def test_precision_recall_in_dict(self):
        result = SemanticAnalyzer.analyze("hello world", "hello earth")
        d = result.to_dict()
        assert "precision_score" in d
        assert "recall_score" in d
        assert 0.0 <= d["precision_score"] <= 1.0
        assert 0.0 <= d["recall_score"] <= 1.0

    def test_drift_level_uses_precision(self):
        """Drift level should be 'extreme' when precision collapses even if
        topic similarity might otherwise be moderate."""
        old = "Revenue grew 15% year-over-year driven by enterprise contracts."
        new = "The history of ancient Rome spans over one thousand years."
        result = SemanticAnalyzer.analyze(old, new)
        # Precision/recall will be very low for completely different content
        assert result.drift_level in ("significant", "extreme")


class TestHallucinationImproved:
    def test_sentence_start_not_entity(self):
        """Words at sentence starts should NOT be flagged as new entities."""
        old = "The project is complete."
        new = "Adoption of new practices is complete."
        risk = HallucinationDetector.detect(new, old)
        # "Adoption" is at sentence start → should not be a new entity
        assert "Adoption" not in risk.new_entities

    def test_numeric_claim_detection(self):
        """Numeric claims introduced in new output should be detected."""
        old = "Revenue increased significantly last year."
        new = "Revenue increased by 42% last year."
        risk = HallucinationDetector.detect(new, old)
        assert len(risk.new_numeric_claims) >= 1

    def test_unsupported_numeric_claims_with_context(self):
        """Numeric claims not in context should be flagged as unsupported."""
        old = "Revenue increased significantly last year."
        new = "Revenue increased by 42% last year."
        context = "Revenue increased significantly."
        risk = HallucinationDetector.detect(new, old, context=context)
        assert len(risk.unsupported_numeric_claims) >= 1

    def test_supported_numeric_claim_not_flagged(self):
        """Numeric claims present in context should NOT be unsupported."""
        old = "Revenue increased last year."
        new = "Revenue increased by 42% last year."
        context = "Revenue increased by 42% last year according to the report."
        risk = HallucinationDetector.detect(new, old, context=context)
        assert len(risk.unsupported_numeric_claims) == 0

    def test_numeric_claims_in_dict(self):
        old = "The price went up."
        new = "The price went up by $10 million."
        risk = HallucinationDetector.detect(new, old)
        d = risk.to_dict()
        assert "new_numeric_claims" in d
        assert "unsupported_numeric_claims" in d


class TestGating:
    def _make_result(self):
        from promptarchive.analysis.engine import AnalysisEngine
        from promptarchive.core.prompt import PromptSnapshot

        def snap(v, out):
            return PromptSnapshot(
                prompt_id="p", version=v, content="", output=out,
                model="gpt-4", temperature=0.0,
            )

        engine = AnalysisEngine()
        return engine.analyze(
            snap("v1", "Hello world, this is a test."),
            snap("v2", "Hello world, this is a test."),
        )

    def test_gate_passes_identical(self):
        from promptarchive.analysis.gating import GateEvaluator, GatingThresholds
        from promptarchive.analysis.engine import AnalysisEngine
        from promptarchive.core.prompt import PromptSnapshot

        def snap(v, out):
            return PromptSnapshot(
                prompt_id="p", version=v, content="", output=out,
                model="gpt-4", temperature=0.0,
            )

        engine = AnalysisEngine()
        result = engine.analyze(snap("v1", "Hello world."), snap("v2", "Hello world."))
        gating = GateEvaluator().evaluate(result)
        assert gating.passed

    def test_gate_fails_on_constraint_violation(self):
        from promptarchive.analysis.gating import GateEvaluator, GatingThresholds
        from promptarchive.analysis.engine import AnalysisEngine
        from promptarchive.core.prompt import PromptSnapshot, Constraint

        c = Constraint(name="formal", tone="formal")
        old = PromptSnapshot(
            prompt_id="p", version="v1", content="", output="Furthermore, we proceed.",
            model="gpt-4", temperature=0.0, constraints=[c],
        )
        new = PromptSnapshot(
            prompt_id="p", version="v2", content="", output="Hey yeah gonna do it lol",
            model="gpt-4", temperature=0.0, constraints=[c],
        )
        engine = AnalysisEngine()
        result = engine.analyze(old, new)
        gating = GateEvaluator().evaluate(result)
        assert not gating.passed
        assert any(lr.layer == "constraints" and not lr.passed for lr in gating.layer_results)

    def test_gating_thresholds_from_dict(self):
        from promptarchive.analysis.gating import GatingThresholds
        data = {
            "config_version": "2.0",
            "min_semantic_similarity": 0.80,
            "max_precision_drop": 0.20,
            "fail_on_schema_change": True,
        }
        t = GatingThresholds.from_dict(data)
        assert t.config_version == "2.0"
        assert t.min_semantic_similarity == 0.80
        assert t.fail_on_schema_change is True

    def test_gating_thresholds_to_dict(self):
        from promptarchive.analysis.gating import GatingThresholds
        t = GatingThresholds()
        d = t.to_dict()
        assert "config_version" in d
        assert "min_semantic_similarity" in d
        assert "hallucination_fail_levels" in d

    def test_gating_result_failed_layers(self):
        from promptarchive.analysis.gating import GatingResult, LayerGateResult
        g = GatingResult(
            passed=False,
            layer_results=[
                LayerGateResult(layer="semantic", passed=True),
                LayerGateResult(layer="hallucination", passed=False, reason="risk: high"),
            ],
        )
        assert len(g.failed_layers) == 1
        assert g.failed_layers[0].layer == "hallucination"

    def test_gating_in_engine_analyze(self):
        from promptarchive.analysis.engine import AnalysisEngine
        from promptarchive.analysis.gating import GatingThresholds
        from promptarchive.core.prompt import PromptSnapshot

        t = GatingThresholds()
        engine = AnalysisEngine(thresholds=t)
        old = PromptSnapshot(
            prompt_id="p", version="v1", content="", output="Hello world.",
            model="gpt-4", temperature=0.0,
        )
        new = PromptSnapshot(
            prompt_id="p", version="v2", content="", output="Hello world.",
            model="gpt-4", temperature=0.0,
        )
        result = engine.analyze(old, new)
        assert result.gating is not None
        assert result.gating.passed

    def test_gating_result_in_to_dict(self):
        from promptarchive.analysis.engine import AnalysisEngine
        from promptarchive.analysis.gating import GatingThresholds
        from promptarchive.core.prompt import PromptSnapshot

        engine = AnalysisEngine(thresholds=GatingThresholds())
        old = PromptSnapshot(
            prompt_id="p", version="v1", content="", output="Hello.",
            model="gpt-4", temperature=0.0,
        )
        new = PromptSnapshot(
            prompt_id="p", version="v2", content="", output="Hello.",
            model="gpt-4", temperature=0.0,
        )
        d = engine.analyze(old, new).to_dict()
        assert "gating" in d
        assert "passed" in d["gating"]
        assert "layer_results" in d["gating"]
