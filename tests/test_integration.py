"""Integration tests: engine + report + CLI."""

import json
import os
import sys
import tempfile
import pytest

from promptarchive.core.prompt import Constraint, Prompt, PromptSnapshot
from promptarchive.analysis.engine import AnalysisEngine
from promptarchive.analysis.report import RegressionReport


def _make_snapshot(prompt_id, version, output, constraints=None, context=None):
    return PromptSnapshot(
        prompt_id=prompt_id,
        version=version,
        content="Say hello",
        output=output,
        model="gpt-4",
        temperature=0.7,
        constraints=constraints or [],
        context=context,
    )


class TestAnalysisEngine:
    def setup_method(self):
        self.engine = AnalysisEngine()

    def test_identical_snapshots(self):
        text = "Hello, this is a standard greeting message."
        old = _make_snapshot("greet", "v1", text)
        new = _make_snapshot("greet", "v2", text)
        result = self.engine.analyze(old, new)
        assert result.risk_level in ("low",)
        assert result.semantic.drift_level in ("none", "minimal")

    def test_with_constraint_violation(self):
        c = Constraint(name="formal", tone="formal")
        old = _make_snapshot("greet", "v1", "Furthermore, we shall proceed.", [c])
        new = _make_snapshot("greet", "v2", "Hey yeah gonna do it lol", [c])
        result = self.engine.analyze(old, new)
        assert not result.constraints.passed
        assert result.risk_level in ("medium", "high", "critical")

    def test_with_reference_answer(self):
        old = _make_snapshot("greet", "v1", "The contract ends on June 30.")
        new = _make_snapshot("greet", "v2", "The agreement terminates in December.")
        result = self.engine.analyze(old, new, reference_answer="The contract ends on June 30.")
        assert result.factual.has_reference

    def test_result_to_dict(self):
        old = _make_snapshot("greet", "v1", "Hello world")
        new = _make_snapshot("greet", "v2", "Hello earth")
        result = self.engine.analyze(old, new)
        d = result.to_dict()
        assert "semantic" in d
        assert "structural" in d
        assert "tone" in d
        assert "constraints" in d
        assert "hallucination" in d
        assert "factual" in d
        assert d["prompt_id"] == "greet"


class TestRegressionReport:
    def setup_method(self):
        self.engine = AnalysisEngine()
        old = _make_snapshot("greet", "v1", "Hello world, this is a greeting.")
        new = _make_snapshot("greet", "v2", "Hi there, this is also a greeting.")
        self.result = self.engine.analyze(old, new)

    def test_text_report_contains_sections(self):
        text = RegressionReport.generate_text(self.result)
        assert "REGRESSION REPORT" in text
        assert "LAYER 1: SEMANTIC SIMILARITY" in text
        assert "LAYER 2: STRUCTURAL ANALYSIS" in text
        assert "LAYER 3: TONE SHIFT" in text
        assert "LAYER 4: CONSTRAINTS" in text
        assert "LAYER 5: HALLUCINATION RISK" in text
        assert "LAYER 6: FACTUAL DRIFT" in text

    def test_json_report_valid(self):
        json_str = RegressionReport.generate_json(self.result)
        d = json.loads(json_str)
        assert d["prompt_id"] == "greet"
        assert "risk_level" in d


class TestCLI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_dir = os.getcwd()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self.orig_dir)

    def _run_cli(self, args):
        from promptarchive.cli import main
        return main(args)

    def test_init(self):
        rc = self._run_cli(["init"])
        assert rc == 0
        assert os.path.isdir(".promptarchive")

    def test_list_prompts_empty(self):
        self._run_cli(["init"])
        rc = self._run_cli(["list-prompts"])
        assert rc == 0

    def test_snapshot_and_log(self):
        self._run_cli(["init"])
        # Create an output file
        with open("output.txt", "w") as fh:
            fh.write("Hello, world!")
        rc = self._run_cli(["snapshot", "greet", "output.txt", "--model", "gpt-4", "--temperature", "0.7"])
        assert rc == 0
        rc = self._run_cli(["log", "greet"])
        assert rc == 0

    def test_diff_requires_two_snapshots(self):
        self._run_cli(["init"])
        with open("output.txt", "w") as fh:
            fh.write("Hello!")
        self._run_cli(["snapshot", "greet", "output.txt"])
        rc = self._run_cli(["diff", "greet"])
        assert rc != 0  # needs 2 snapshots

    def test_diff_with_two_snapshots(self):
        self._run_cli(["init"])
        with open("v1.txt", "w") as fh:
            fh.write("Hello, this is version one of the greeting.")
        with open("v2.txt", "w") as fh:
            fh.write("Hi there, version two says something different.")
        self._run_cli(["snapshot", "greet", "v1.txt"])
        self._run_cli(["snapshot", "greet", "v2.txt"])
        rc = self._run_cli(["diff", "greet"])
        assert rc == 0

    def test_diff_json_format(self):
        self._run_cli(["init"])
        with open("v1.txt", "w") as fh:
            fh.write("Hello world.")
        with open("v2.txt", "w") as fh:
            fh.write("Hi earth.")
        self._run_cli(["snapshot", "greet", "v1.txt"])
        self._run_cli(["snapshot", "greet", "v2.txt"])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run_cli(["diff", "greet", "--format", "json"])
        assert rc == 0
        d = json.loads(buf.getvalue())
        assert "risk_level" in d
