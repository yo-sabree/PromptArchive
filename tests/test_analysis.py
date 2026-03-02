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
        diff = StructuralAnalyzer.analyze("hello world", "goodbye world")
        assert not diff.has_schema_change
        assert diff.total_changes == 0


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
