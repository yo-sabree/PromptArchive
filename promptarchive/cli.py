"""Command-line interface for PromptArchive."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from promptarchive.analysis.engine import AnalysisEngine
from promptarchive.analysis.report import RegressionReport
from promptarchive.core.registry import PromptRegistry
from promptarchive.storage.snapshots import SnapshotStore, init_archive

_DEFAULT_BASE = ".promptarchive"
_REGISTRY_FILE = os.path.join(_DEFAULT_BASE, "registry.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store() -> SnapshotStore:
    return SnapshotStore(base_dir=_DEFAULT_BASE)


def _get_registry() -> PromptRegistry:
    if os.path.isfile(_REGISTRY_FILE):
        return PromptRegistry(registry_path=_REGISTRY_FILE)
    return PromptRegistry()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    path = init_archive(_DEFAULT_BASE)
    print(f"Initialized PromptArchive at {path}")
    print("Track '.promptarchive/' with Git for version control.")
    return 0


def cmd_list_prompts(args: argparse.Namespace) -> int:
    registry = _get_registry()
    store = _get_store()

    # Combine registered prompts with prompts that have snapshots
    prompt_ids_from_store = set(store.list_prompt_ids())
    registered = {p.id for p in registry.list_prompts()}
    all_ids = sorted(registered | prompt_ids_from_store)

    if not all_ids:
        print("No prompts found. Use 'promptarchive snapshot' to add one.")
        return 0

    print(f"{'ID':<30} {'Snapshots':>10}  {'Name'}")
    print("-" * 60)
    for pid in all_ids:
        snapshots = store.list_snapshots(pid)
        prompt = registry.get(pid)
        name = prompt.name if prompt else "(unregistered)"
        print(f"{pid:<30} {len(snapshots):>10}  {name}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    store = _get_store()
    snapshots = store.list_snapshots(args.prompt_id)
    if not snapshots:
        print(f"No snapshots found for prompt '{args.prompt_id}'.")
        return 1

    print(f"History for '{args.prompt_id}':")
    print(f"{'Version':<10} {'Model':<20} {'Temp':>6}  {'Timestamp'}")
    print("-" * 65)
    for s in snapshots:
        ts = s.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{s.version:<10} {s.model:<20} {s.temperature:>6.2f}  {ts}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    output_file: str = args.output_file
    if not os.path.isfile(output_file):
        print(f"Error: output file '{output_file}' not found.", file=sys.stderr)
        return 1

    with open(output_file, encoding="utf-8") as fh:
        output_text = fh.read()

    # Determine next version
    store = _get_store()
    existing = store.list_snapshots(args.prompt_id)
    version = f"v{len(existing) + 1}"

    # Build a minimal snapshot (no Prompt object required)
    from promptarchive.core.prompt import PromptSnapshot
    from datetime import datetime, timezone

    # Load prompt content from registry if available
    registry = _get_registry()
    prompt = registry.get(args.prompt_id)
    content = prompt.content if prompt else ""
    constraints = prompt.constraints if prompt else []

    context_text: Optional[str] = None
    if args.context:
        if os.path.isfile(args.context):
            with open(args.context, encoding="utf-8") as fh:
                context_text = fh.read()
        else:
            context_text = args.context

    snapshot = PromptSnapshot(
        prompt_id=args.prompt_id,
        version=version,
        content=content,
        output=output_text,
        model=args.model,
        temperature=args.temperature,
        context=context_text,
        constraints=constraints,
    )

    path = store.save_snapshot(snapshot)
    print(f"Snapshot {version} saved: {path}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    store = _get_store()
    snapshots = store.list_snapshots(args.prompt_id)

    if len(snapshots) < 2:
        print(
            f"Need at least 2 snapshots for '{args.prompt_id}'. "
            f"Found {len(snapshots)}.",
            file=sys.stderr,
        )
        return 1

    old_snapshot = snapshots[-2]
    new_snapshot = snapshots[-1]

    reference: Optional[str] = None
    if args.reference:
        if os.path.isfile(args.reference):
            with open(args.reference, encoding="utf-8") as fh:
                reference = fh.read()
        else:
            reference = args.reference

    engine = AnalysisEngine()
    result = engine.analyze(
        old_snapshot=old_snapshot,
        new_snapshot=new_snapshot,
        reference_answer=reference,
    )

    if args.format == "json":
        print(RegressionReport.generate_json(result))
    else:
        print(RegressionReport.generate_text(result))

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptarchive",
        description="Local, Git-native prompt version control and regression testing.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # init
    sub.add_parser("init", help="Initialize .promptarchive/ in the current directory.")

    # list-prompts
    sub.add_parser("list-prompts", help="List all tracked prompts.")

    # log
    log_parser = sub.add_parser("log", help="Show snapshot history for a prompt.")
    log_parser.add_argument("prompt_id", help="Prompt ID")

    # snapshot
    snap_parser = sub.add_parser("snapshot", help="Snapshot an LLM output file.")
    snap_parser.add_argument("prompt_id", help="Prompt ID")
    snap_parser.add_argument("output_file", help="Path to a file containing the LLM output")
    snap_parser.add_argument("--model", default="unknown", help="Model name (default: unknown)")
    snap_parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature used (default: 0.0)"
    )
    snap_parser.add_argument(
        "--context", default=None, help="Context file or string for hallucination detection"
    )

    # diff
    diff_parser = sub.add_parser(
        "diff", help="Compare the two most recent snapshots of a prompt."
    )
    diff_parser.add_argument("prompt_id", help="Prompt ID")
    diff_parser.add_argument(
        "--reference",
        default=None,
        help="Path to file or string with expected reference answer",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "init": cmd_init,
        "list-prompts": cmd_list_prompts,
        "log": cmd_log,
        "snapshot": cmd_snapshot,
        "diff": cmd_diff,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
