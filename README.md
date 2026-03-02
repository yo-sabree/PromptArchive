# PromptArchive

**Local, Git-native prompt version control and regression testing.**

A lightweight framework for tracking prompt behavior, detecting semantic drift, and running regression tests—completely offline, no cloud APIs, no telemetry.

## Installation

```bash
pip install promptarchive
```

## Quick Start

### 1. Initialize in Your Project

```bash
promptarchive init
```

### 2. Define a Prompt with Constraints

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

### 3. Capture Behavioral Snapshots

```python
snapshot = prompt.add_snapshot(
    output=output,
    model="gpt-4",
    temperature=0.7,
    context=original_document,
)

from promptarchive.storage.snapshots import SnapshotStore
store = SnapshotStore()
store.save_snapshot(snapshot)
```

### 4. Run Regression Tests

```python
from promptarchive.analysis.engine import AnalysisEngine
from promptarchive.analysis.report import RegressionReport

engine = AnalysisEngine()
report = engine.analyze(
    old_snapshot=old_version,
    new_snapshot=new_version,
    reference_answer="Expected summary text here",
)

print(RegressionReport.generate_text(report))
```

## CLI Usage

```bash
promptarchive init
promptarchive list-prompts
promptarchive log greeting_prompt
promptarchive snapshot greeting_prompt output.txt --model gpt-4 --temperature 0.7
promptarchive diff greeting_prompt
promptarchive diff greeting_prompt --reference expected.txt
promptarchive diff greeting_prompt --format json
```

## License

MIT
