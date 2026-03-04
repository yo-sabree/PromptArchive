# PromptArchive
<img width="1035" height="540" alt="logo" src="https://github.com/user-attachments/assets/06b9e776-bf42-45d9-8a8d-65590be40dac" />

**A simple, Git-native way to version and manage your prompts locally.**

PromptArchive is a lightweight tool to manage, version, and regression-test your LLM prompts — completely offline and private. It tracks prompt behavior, detects semantic drift, flags hallucinations, enforces output constraints, and runs regression tests with no cloud APIs and no telemetry.

---

## Table of Contents

1. [Why PromptArchive?](#why-promptarchive)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Python API](#python-api)
6. [CLI Reference](#cli-reference)
   - [init](#init)
   - [register](#register)
   - [snapshot](#snapshot)
   - [log](#log)
   - [list-prompts](#list-prompts)
   - [show](#show)
   - [diff](#diff)
   - [validate](#validate)
   - [search](#search)
   - [stats](#stats)
   - [export / import-archive](#export--import-archive)
   - [delete](#delete)
   - [scan](#scan)
7. [Analysis Layers](#analysis-layers)
8. [CI/CD Gating](#cicd-gating)
9. [Privacy Design](#privacy-design)
10. [Encryption at Rest](#encryption-at-rest)
11. [Contributing](#contributing)
12. [License](#license)

---

## Why PromptArchive?

LLM prompts drift over time. A prompt that produced reliable legal summaries in January may produce subtly different — or dangerously wrong — output in June after a model upgrade, a temperature tweak, or a system-prompt change. Most teams notice the regression only after a user complaint.

**PromptArchive solves this by treating prompts like code:**

| Problem | Solution |
|---|---|
| "Did the summary change after we updated the model?" | Snapshot every output and `diff` any two versions |
| "Is the new output still formally worded?" | Declare a `Constraint` and validate automatically |
| "Did the LLM invent new facts?" | Hallucination detection flags new entities and numeric claims |
| "We need CI to block bad prompt changes" | Pass/fail gating with configurable thresholds |
| "GDPR: must not log PII" | Built-in PII scanner with redaction, all local |
| "Can't send data to a third-party service" | 100 % offline — no outbound connections, ever |

---

## Installation

```bash
pip install promptarchive
```

Enable local sentence-transformer embeddings (optional, improves semantic analysis):

```bash
pip install "promptarchive[semantic]"
```

Enable AES-256-GCM encryption at rest (optional):

```bash
pip install "promptarchive[secure]"
```

---

## Quick Start

```bash
# 1. Initialize in your project
promptarchive init
# Initialized PromptArchive at .promptarchive
# Track '.promptarchive/' with Git for version control.

# 2. Register a prompt definition
promptarchive register prompt.json
# Registered prompt 'contract_summary' (Contract Summary Generator).

# 3. Save an LLM output as a snapshot
echo "This agreement includes a termination clause effective upon 30-day written notice." > v1.txt
promptarchive snapshot contract_summary v1.txt --model gpt-4 --temperature 0.7
# Snapshot v1 saved: .promptarchive/snapshots/contract_summary/v1.json

# 4. Diff two snapshots
promptarchive diff contract_summary
```

---

## Core Concepts

### Prompt
A prompt has an `id`, `name`, `content` (the actual template), optional `description`, `tags`, and a list of `Constraint` objects.

### Constraint
Rules that every output must satisfy: required phrases, forbidden phrases, regex patterns, length limits, and tone checks (formal/casual).

### PromptSnapshot
An immutable record of: the prompt content used, the LLM output, model name, temperature, timestamp, and optional context document.

### Analysis Engine
Six analysis layers that compare two snapshots and detect regressions: semantic similarity, structural diff, tone shift, constraint violations, hallucination risk, and factual drift.

### Gating
A pass/fail verdict suitable for CI/CD pipelines — returns a non-zero exit code when thresholds are breached.

---

## Python API

### Define a Prompt with Constraints

```python
from promptarchive import Prompt, Constraint, PromptRegistry

constraints = [
    Constraint(
        name="legal_tone",
        must_include=["termination clause"],
        must_not_include=["legal advice"],
        tone="formal",
        max_length=500,
    ),
]

prompt = Prompt(
    id="contract_summary",
    name="Contract Summary Generator",
    content="Summarize this contract: {document}",
    constraints=constraints,
    tags=["legal", "critical"],
)

registry = PromptRegistry()
registry.register(prompt)
```

### Capture Behavioral Snapshots

```python
from promptarchive.storage.snapshots import SnapshotStore

output_v1 = "This agreement includes a termination clause effective upon 30-day written notice."
output_v2 = "The contract terminates with a termination clause after 60 days notice."

snapshot_v1 = prompt.add_snapshot(output=output_v1, model="gpt-4", temperature=0.7)
snapshot_v2 = prompt.add_snapshot(output=output_v2, model="gpt-4o", temperature=0.5)

store = SnapshotStore()
store.save_snapshot(snapshot_v1)
store.save_snapshot(snapshot_v2)
```

### Validate Constraints Without Snapshotting

Use this to check an output before committing it — great for CI pre-flight checks.

```python
from promptarchive.analysis.constraints import ConstraintValidator

result = ConstraintValidator.validate(output_v1, prompt.constraints)
print(result.passed)           # True
print(result.violation_count)  # 0

bad_output = "Hey, this contract is basically gonna end kinda soon lol"
result2 = ConstraintValidator.validate(bad_output, prompt.constraints)
print(result2.passed)          # False
for v in result2.violations:
    print(f"[{v.constraint_name}] {v.constraint_type}: {v.message}")
# [legal_tone] must_include: Required phrase not found: 'termination clause'
# [legal_tone] tone: Detected casual language
```

### Run Regression Analysis

```python
from promptarchive.analysis.engine import AnalysisEngine
from promptarchive.analysis.report import RegressionReport

engine = AnalysisEngine()
result = engine.analyze(
    old_snapshot=snapshot_v1,
    new_snapshot=snapshot_v2,
    reference_answer="This agreement includes a termination clause upon 30-day written notice.",
)

print(result.risk_level)               # "low" | "medium" | "high" | "critical"
print(result.has_changes)              # True/False
print(result.semantic.drift_level)     # "none" | "minimal" | "moderate" | "significant" | "extreme"
print(result.constraints.passed)       # True/False
print(result.hallucination.risk_level) # "low" | "medium" | "high"
print(result.factual.drift_level)      # "none" | "minor" | "moderate" | "major"

# Human-readable report
print(RegressionReport.generate_text(result))

# Machine-readable JSON
import json
d = json.loads(RegressionReport.generate_json(result))
print(d["risk_level"])
```

**Sample output:**

```
REGRESSION REPORT: contract_summary

Comparing: v1 → v2
Risk Level: MEDIUM
Has Changes: True

LAYER 1: SEMANTIC SIMILARITY
  Similarity Score: 34.82%
  Lexical Precision: 36.36%  ⚠  precision collapse
  Lexical Recall:    33.33%
  Drift Level: extreme
  Direction: decreased
  Method: tfidf

LAYER 2: STRUCTURAL DIFF
  Schema Changes: False
  Lines: +1 -1
  ──────────────────────────────────────────────────────────
  --- old
  +++ new
  @@ -1 +1 @@
  -This agreement includes a termination clause effective upon 30-day written notice.
  +The contract terminates with a termination clause after 60 days notice.

LAYER 3: TONE SHIFT
  Formality:      +0.0%
  Sentiment:      -50.0%
  Assertiveness:  +0.0%
  Reading Grade:  -2.3

LAYER 4: CONSTRAINTS
  Status: ✓ All passed

LAYER 5: HALLUCINATION RISK
  Risk Level: medium
  Confidence: 55%
  New Named Entities:     0
  Unsupported Entities:   0
  New Numeric Claims:     1
  Unsupported Num Claims: 0

LAYER 6: FACTUAL DRIFT
  Overlap Score: 66.67%
  Drift Level: minor
  Missing facts: 30-day, agreement, effective, includes, notice, upon, written
  Extra content: 60, after, days, terminates
```

### Scan PII Before Snapshotting

```python
from promptarchive.privacy.pii import PIIDetector

text = "Contact Alice at alice@company.com or call 555-867-5309."
report = PIIDetector.scan(text)

print(report.has_pii)  # True
for f in report.findings:
    print(f"[{f.label}] {f.value!r} at pos {f.start}")
# [email] 'alice@company.com' at pos 17
# [phone] '555-867-5309' at pos 43

clean = PIIDetector.redact(text)
print(clean)
# Contact Alice at [EMAIL] or call [PHONE].
```

### Persist and Load the Registry

```python
import os

registry.save(".promptarchive/registry.json")

# Load later
from promptarchive.core.registry import PromptRegistry
registry2 = PromptRegistry(registry_path=".promptarchive/registry.json")
p = registry2.get("contract_summary")
print(p.name)  # Contract Summary Generator
```

---

## CLI Reference

### `init`

Initialize a `.promptarchive/` directory with Git support. Run once per project.

```bash
promptarchive init
```

**Output:**
```
Initialized PromptArchive at .promptarchive
Track '.promptarchive/' with Git for version control.
```

**What it creates:**
```
.promptarchive/
  snapshots/        ← versioned output files
  .gitignore        ← ignores *.tmp
  README.md         ← marker file
```

---

### `register`

Register a prompt from a JSON definition file. This enables constraint validation, the `validate` command, and shows names in `list-prompts`.

```bash
promptarchive register prompt.json
```

**`prompt.json` example:**
```json
{
  "id": "contract_summary",
  "name": "Contract Summary Generator",
  "content": "Summarize this contract: {document}",
  "description": "Produces a formal summary of legal contracts.",
  "constraints": [
    {
      "name": "legal_tone",
      "must_include": ["termination clause"],
      "must_not_include": ["legal advice"],
      "tone": "formal",
      "max_length": 500
    }
  ],
  "tags": ["legal", "critical"]
}
```

**Output:**
```
Registered prompt 'contract_summary' (Contract Summary Generator).
```

---

### `snapshot`

Save an LLM output file as a versioned snapshot. Each call increments the version (`v1`, `v2`, …).

```bash
promptarchive snapshot <prompt_id> <output_file> [--model NAME] [--temperature FLOAT] [--context FILE_OR_STRING]
```

**Example:**
```bash
echo "This agreement includes a termination clause." > output.txt
promptarchive snapshot contract_summary output.txt --model gpt-4 --temperature 0.7
```

**Output:**
```
Snapshot v1 saved: .promptarchive/snapshots/contract_summary/v1.json
```

The `--context` flag accepts a file path or raw string and is used by the hallucination detector to determine which entities are supported by the source document.

---

### `log`

Show the snapshot history for a prompt.

```bash
promptarchive log contract_summary
```

**Output:**
```
History for 'contract_summary':
Version    Model                  Temp  Timestamp
-----------------------------------------------------------------
v1         gpt-4                  0.70  2026-03-04 10:00:00
v2         gpt-4o                 0.50  2026-03-04 10:05:00
```

---

### `list-prompts`

List all tracked prompts with snapshot counts and registered names.

```bash
promptarchive list-prompts
```

**Output:**
```
ID                              Snapshots  Name
------------------------------------------------------------
contract_summary                        2  Contract Summary Generator
email_drafter                           5  (unregistered)
```

---

### `show`

Show the full details of a snapshot.

```bash
promptarchive show <prompt_id> [--version v2] [--format json]
```

**Example:**
```bash
promptarchive show contract_summary --version v1
```

**Output:**
```
Prompt:      contract_summary
Version:     v1
Model:       gpt-4
Temperature: 0.7
Timestamp:   2026-03-04 10:00:00 UTC

--- PROMPT CONTENT ---
Summarize this contract: {document}

--- OUTPUT ---
This agreement includes a termination clause effective upon 30-day written notice.
```

**JSON output** (`--format json`):
```bash
promptarchive show contract_summary --format json
```
```json
{
  "prompt_id": "contract_summary",
  "version": "v2",
  "content": "Summarize this contract: {document}",
  "output": "The contract terminates with a termination clause after 60 days notice.",
  "model": "gpt-4o",
  "temperature": 0.5,
  "timestamp": "2026-03-04T10:05:00+00:00",
  "metadata": {},
  "constraints": [],
  "context": null
}
```

---

### `diff`

Compare two snapshots and display a full regression report. By default, compares the last two snapshots. Use `--from-version`/`--to-version` to compare any two specific versions.

```bash
promptarchive diff <prompt_id> \
  [--from-version v1] [--to-version v3] \
  [--reference FILE] \
  [--format json] \
  [--gate] [--config gate.json]
```

**Default (last two snapshots):**
```bash
promptarchive diff contract_summary
```

**Compare specific versions (new in v1.0):**
```bash
promptarchive diff contract_summary --from-version v1 --to-version v3
```

**With a reference answer (enables factual drift layer):**
```bash
promptarchive diff contract_summary --reference expected_summary.txt
```

**JSON output:**
```bash
promptarchive diff contract_summary --format json
```

**Gating (non-zero exit when thresholds are breached):**
```bash
promptarchive diff contract_summary --gate
promptarchive diff contract_summary --config gate_thresholds.json
```

Sample `gate_thresholds.json`:
```json
{
  "config_version": "1.0",
  "min_semantic_similarity": 0.80,
  "max_precision_drop": 0.20,
  "fail_on_constraint_violation": true,
  "fail_on_schema_change": true,
  "hallucination_fail_levels": ["medium", "high"],
  "max_unsupported_entities": 0,
  "max_unsupported_numeric_claims": 0,
  "min_factual_overlap": 0.70,
  "factual_fail_levels": ["moderate", "major"],
  "overall_fail_levels": ["high", "critical"]
}
```

---

### `validate`

Validate an output file against a registered prompt's constraints **without** creating a snapshot. Useful as a pre-flight check in CI before committing an output.

```bash
promptarchive validate <prompt_id> <output_file> [--format json]
```

**Example — passing:**
```bash
echo "This agreement includes a termination clause." > output.txt
promptarchive validate contract_summary output.txt
```
```
✓ All 1 constraint(s) passed.
```
Exit code: **0**

**Example — failing:**
```bash
echo "Hey, the contract is basically gonna end soon lol" > bad.txt
promptarchive validate contract_summary bad.txt
```
```
✗ 2 violation(s) found (out of 1 constraint(s)):
  [legal_tone] must_include: Required phrase not found: 'termination clause'
  [legal_tone] tone: Detected casual language
```
Exit code: **1** (can be used to block CI pipelines)

**JSON output:**
```bash
promptarchive validate contract_summary bad.txt --format json
```
```json
{
  "passed": false,
  "violation_count": 2,
  "violations": [
    {
      "constraint_name": "legal_tone",
      "constraint_type": "must_include",
      "message": "Required phrase not found: 'termination clause'"
    },
    {
      "constraint_name": "legal_tone",
      "constraint_type": "tone",
      "message": "Detected casual language"
    }
  ]
}
```

---

### `search`

Search snapshots by model, date range, or keyword.

```bash
promptarchive search [--model NAME] [--prompt-id ID] [--keyword TEXT] [--since DATE] [--until DATE]
```

**Examples:**
```bash
# All snapshots from gpt-4
promptarchive search --model gpt-4
# Prompt ID                      Ver      Model                Timestamp
# ---------------------------------------------------------------------------
# contract_summary               v1       gpt-4                2026-03-04 10:00:00

# Keyword in content or output
promptarchive search --keyword "termination" --prompt-id contract_summary

# Date range
promptarchive search --since 2026-01-01 --until 2026-06-30
```

---

### `stats`

Show aggregate statistics across all stored snapshots.

```bash
promptarchive stats [--format json]
```

**Output:**
```
Total prompts:        1
Total snapshots:      2
Avg output length:    76.5 chars
Oldest snapshot:      2026-03-04T10:00:00+00:00
Newest snapshot:      2026-03-04T10:05:00+00:00
Models used:
  gpt-4                          1 snapshot(s)
  gpt-4o                         1 snapshot(s)
```

**JSON output:**
```bash
promptarchive stats --format json
```
```json
{
  "total_prompts": 1,
  "total_snapshots": 2,
  "models": {
    "gpt-4": 1,
    "gpt-4o": 1
  },
  "oldest": "2026-03-04T10:00:00+00:00",
  "newest": "2026-03-04T10:05:00+00:00",
  "avg_output_length": 76.5
}
```

---

### `export` / `import-archive`

Back up all snapshots to a ZIP file and restore them.

```bash
# Export
promptarchive export backup.zip
# Exported 2 snapshot file(s) to 'backup.zip'.

# Import (skip existing by default)
promptarchive import-archive backup.zip
# Imported 0 snapshot file(s) from 'backup.zip'.

# Import with overwrite
promptarchive import-archive backup.zip --overwrite
# Imported 2 snapshot file(s) from 'backup.zip'.
```

---

### `delete`

Delete a specific snapshot version or all snapshots for a prompt. Deleted files are overwritten with zeros before removal (secure deletion).

```bash
# Delete one version
promptarchive delete contract_summary --version v1
# Deleted snapshot v1 for 'contract_summary'.

# Delete all versions
promptarchive delete contract_summary --all
# Deleted 2 snapshot(s) for 'contract_summary'.
```

---

### `scan`

Scan a file for Personally Identifiable Information (PII) **before** committing it as a snapshot. All analysis is local — no data leaves your machine. Exits with code 1 if PII is found, making it suitable for CI hooks.

```bash
promptarchive scan output.txt
# WARNING: 2 PII finding(s) detected:
#   [email] at position 17: 'alice@company.com'
#   [phone] at position 43: '555-867-5309'
# Exit code: 1

# Also write a redacted copy
promptarchive scan output.txt --redact
# WARNING: 2 PII finding(s) detected:
#   [email] at position 17: 'alice@company.com'
#   [phone] at position 43: '555-867-5309'
# Redacted copy written to 'output.txt.redacted'.
```

Detected PII categories: **email**, **phone**, **SSN**, **credit card**, **IP address**, **API key**.

---

## Analysis Layers

The `diff` command runs six independent analysis layers:

| Layer | What it detects | Use case |
|---|---|---|
| **1. Semantic Similarity** | Embedding/TF-IDF cosine distance + lexical precision/recall | Did the meaning change? |
| **2. Structural Diff** | JSON schema drift (added/removed keys, type changes); line-level unified diff for plain text | Did the output format change? |
| **3. Tone Shift** | Formality, sentiment, assertiveness, Flesch-Kincaid reading grade | Did the language register shift? |
| **4. Constraint Validation** | `must_include`, `must_not_include`, regex patterns, length, tone rules | Did the output violate declared rules? |
| **5. Hallucination Risk** | New named entities and numeric claims not present in the old output or context | Did the model invent new facts? |
| **6. Factual Drift** | Word-level F1 overlap against a reference answer | Did the output diverge from ground truth? |

**Risk levels** (`low` / `medium` / `high` / `critical`) are computed from a weighted combination of all six layers.

**Drift levels** used by the semantic layer:
- `none` — effectively identical
- `minimal` — minor wording changes
- `moderate` — noticeable rephrasing
- `significant` — substantial content change
- `extreme` — almost entirely different

---

## CI/CD Gating

PromptArchive can act as a quality gate in your CI pipeline. When the `--gate` flag or a `--config` file is used, `diff` returns **exit code 1** if any threshold is breached.

**GitHub Actions example:**

```yaml
- name: Run prompt regression test
  run: |
    promptarchive diff contract_summary \
      --reference expected_output.txt \
      --config gate_thresholds.json
  # Fails the build if the gate does not pass
```

**Configurable thresholds:**

| Parameter | Default | Description |
|---|---|---|
| `min_semantic_similarity` | `0.70` | Minimum cosine/TF-IDF similarity |
| `max_precision_drop` | `0.30` | Maximum allowed lexical-precision drop |
| `fail_on_constraint_violation` | `true` | Any constraint violation fails the gate |
| `fail_on_schema_change` | `false` | JSON schema changes fail the gate |
| `hallucination_fail_levels` | `["high"]` | Risk levels that fail the gate |
| `max_unsupported_entities` | `0` | Max new entities not in context |
| `max_unsupported_numeric_claims` | `0` | Max new numeric claims not in context |
| `min_factual_overlap` | `0.70` | Minimum word-overlap F1 vs reference |
| `factual_fail_levels` | `["major"]` | Factual drift levels that fail the gate |
| `overall_fail_levels` | `["high", "critical"]` | Overall risk levels that fail the gate |

---

## Privacy Design

| Feature | Implementation |
|---|---|
| No telemetry | Zero outbound connections |
| Local embeddings | `sentence-transformers` runs fully offline |
| PII detection | Regex-only, stdlib `re`, no external deps |
| Encryption at rest | AES-256-GCM via `cryptography` (optional) |
| Secure deletion | Files overwritten with zeros before `unlink` |
| Git-native | Snapshots are plain JSON — diff/blame in Git |

---

## Encryption at Rest

Requires `pip install "promptarchive[secure]"`.

```python
from promptarchive.storage.snapshots import SnapshotStore

store = SnapshotStore(passphrase="my-secret-passphrase")
store.save_snapshot(snapshot)
# Written as .promptarchive/snapshots/contract_summary/v1.json.enc
```

Each encrypted file uses a random **salt + nonce** so no two ciphertexts are identical even for the same plaintext. Keys are derived via **PBKDF2-HMAC-SHA256 (200,000 iterations)** following OWASP guidelines.

---

## Contributing

Contributions are welcome! Here is how to get started:

### Setup

```bash
git clone https://github.com/yo-sabree/PromptArchive.git
cd PromptArchive
pip install -e ".[dev]"
```

### Run the Test Suite

```bash
python -m pytest tests/ -v
```

All tests must pass before submitting a pull request.

### Project Structure

```
promptarchive/
  cli.py              ← CLI entry point (all commands)
  core/
    prompt.py         ← Prompt, PromptSnapshot, Constraint data models
    registry.py       ← PromptRegistry (in-memory + JSON persistence)
  analysis/
    engine.py         ← AnalysisEngine orchestrator + RegressionResult
    semantic.py       ← Layer 1: TF-IDF / sentence-transformer similarity
    structural.py     ← Layer 2: JSON schema diff + unified text diff
    tone.py           ← Layer 3: formality, sentiment, assertiveness
    constraints.py    ← Layer 4: rule-based constraint validation
    hallucination.py  ← Layer 5: named entity + numeric claim tracking
    factual.py        ← Layer 6: word-overlap F1 vs reference
    gating.py         ← Pass/fail gating with configurable thresholds
    report.py         ← Human-readable and JSON report generation
  storage/
    snapshots.py      ← SnapshotStore (save/load/search/export/import)
  privacy/
    pii.py            ← PIIDetector (scan + redact)
tests/
  test_core.py        ← Prompt, Constraint, PromptSnapshot
  test_analysis.py    ← All 6 analysis layers + gating
  test_storage.py     ← SnapshotStore, PromptRegistry, search, export
  test_privacy.py     ← PII detection and redaction
  test_integration.py ← End-to-end CLI tests
```

### How to Add a New Analysis Layer

1. Create `promptarchive/analysis/my_layer.py` with a result dataclass and an analyzer class.
2. Import and call it in `AnalysisEngine.analyze()` (`engine.py`).
3. Add the result to `RegressionResult` and its `to_dict()`.
4. Render it in `RegressionReport.generate_text()` and `generate_json()` (`report.py`).
5. Add tests in `tests/test_analysis.py`.

### How to Add a New CLI Command

1. Write a `cmd_<name>(args)` function in `cli.py`.
2. Add a sub-parser in `build_parser()`.
3. Register the handler in the `handlers` dict inside `main()`.
4. Add an integration test in `tests/test_integration.py`.

### Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -q`)
- [ ] New features include tests
- [ ] Public APIs have docstrings
- [ ] README updated if a new command or feature is added
- [ ] No new external dependencies without discussion

### Reporting Bugs

Open an [issue](https://github.com/yo-sabree/PromptArchive/issues) with:
- Python version and OS
- Minimal reproduction steps
- Expected vs actual behavior

---

## The MIT License (MIT)

Copyright © 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
