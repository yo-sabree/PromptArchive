# PromptArchive

**Local, Git-native prompt version control and regression testing.**

A lightweight framework for tracking prompt behavior, detecting semantic drift, and running regression tests—completely offline, no cloud APIs, no telemetry.

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

### Core commands

```bash
promptarchive init                        # Initialize .promptarchive/
promptarchive list-prompts                # List all tracked prompts
promptarchive register prompt.json        # Register a prompt from a JSON file
promptarchive log <prompt_id>             # Show snapshot history
promptarchive snapshot <id> output.txt --model gpt-4 --temperature 0.7
promptarchive diff <prompt_id>            # Compare last two snapshots
promptarchive diff <prompt_id> --reference expected.txt
promptarchive diff <prompt_id> --format json
```

### Viewing and managing snapshots

```bash
promptarchive show <prompt_id>                     # Show latest snapshot
promptarchive show <prompt_id> --version v2        # Show a specific version
promptarchive show <prompt_id> --format json       # Machine-readable output
promptarchive delete <prompt_id> --version v1      # Delete one snapshot
promptarchive delete <prompt_id> --all             # Delete all snapshots
```

### Search

```bash
promptarchive search --model gpt-4
promptarchive search --keyword "termination" --prompt-id contract_summary
promptarchive search --since 2024-01-01 --until 2024-12-31
```

### Statistics

```bash
promptarchive stats                   # Aggregate stats (text)
promptarchive stats --format json     # Machine-readable
```

### Export & Import (backup / restore)

```bash
promptarchive export backup.zip               # Export all snapshots
promptarchive import-archive backup.zip       # Import (skip existing)
promptarchive import-archive backup.zip --overwrite
```

### Privacy: PII scanning

Scan a file for Personally Identifiable Information **before** committing it as a
snapshot.  All analysis is local — no data leaves your machine.

```bash
promptarchive scan output.txt           # Exits 1 if PII detected
promptarchive scan output.txt --redact  # Also writes output.txt.redacted
```

Detected PII categories: email, phone, SSN, credit card, IP address, API key.

You can also use the `PIIDetector` API directly:

```python
from promptarchive.privacy.pii import PIIDetector

report = PIIDetector.scan(text)
if report.has_pii:
    print(report.to_dict())
    clean = PIIDetector.redact(text)
```

### Encryption at rest (requires `promptarchive[secure]`)

```python
from promptarchive.storage.snapshots import SnapshotStore

store = SnapshotStore(passphrase="my-secret-passphrase")
store.save_snapshot(snapshot)   # Written as .json.enc (AES-256-GCM)
```

Each encrypted file uses a random salt + nonce so no two ciphertexts are
identical even for the same plaintext.  Keys are derived via PBKDF2-HMAC-SHA256
(200 000 iterations).

## Privacy design

| Feature | Implementation |
|---|---|
| No telemetry | Zero outbound connections |
| Local embeddings | `sentence-transformers` runs fully offline |
| PII detection | Regex-only, stdlib `re`, no external deps |
| Encryption at rest | AES-256-GCM via `cryptography` (optional) |
| Secure deletion | Files overwritten with zeros before `unlink` |
| Git-native | Snapshots are plain JSON — diff/blame in Git |

## License

MIT
