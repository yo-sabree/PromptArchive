"""Command-line interface for PromptArchive."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from promptarchive.analysis.engine import AnalysisEngine
from promptarchive.analysis.report import RegressionReport
from promptarchive.core.prompt import Prompt, Constraint
from promptarchive.core.registry import PromptRegistry
from promptarchive.privacy.pii import PIIDetector
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


def cmd_register(args: argparse.Namespace) -> int:
    """Register a prompt from a JSON file."""
    if not os.path.isfile(args.json_file):
        print(f"Error: file '{args.json_file}' not found.", file=sys.stderr)
        return 1
    with open(args.json_file, encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        prompt = Prompt.from_dict(data)
    except (KeyError, TypeError) as exc:
        print(f"Error: invalid prompt JSON – {exc}", file=sys.stderr)
        return 1
    registry = _get_registry()
    registry.register(prompt)
    os.makedirs(_DEFAULT_BASE, exist_ok=True)
    registry.save(_REGISTRY_FILE)
    print(f"Registered prompt '{prompt.id}' ({prompt.name}).")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show the full details of a specific snapshot."""
    store = _get_store()
    snapshots = store.list_snapshots(args.prompt_id)
    if not snapshots:
        print(f"No snapshots found for '{args.prompt_id}'.", file=sys.stderr)
        return 1

    # Pick the requested version or the latest
    version: Optional[str] = getattr(args, "version", None)
    if version:
        snap = next((s for s in snapshots if s.version == version), None)
        if snap is None:
            print(
                f"Version '{version}' not found for '{args.prompt_id}'.",
                file=sys.stderr,
            )
            return 1
    else:
        snap = snapshots[-1]

    if args.format == "json":
        print(json.dumps(snap.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Prompt:      {snap.prompt_id}")
        print(f"Version:     {snap.version}")
        print(f"Model:       {snap.model}")
        print(f"Temperature: {snap.temperature}")
        print(f"Timestamp:   {snap.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if snap.content:
            print(f"\n--- PROMPT CONTENT ---\n{snap.content}")
        print(f"\n--- OUTPUT ---\n{snap.output}")
        if snap.context:
            print(f"\n--- CONTEXT ---\n{snap.context}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a snapshot version or all snapshots for a prompt."""
    store = _get_store()
    version: Optional[str] = getattr(args, "version", None)
    if version:
        ok = store.delete_snapshot(args.prompt_id, version)
        if ok:
            print(f"Deleted snapshot {version} for '{args.prompt_id}'.")
        else:
            print(
                f"Snapshot {version} not found for '{args.prompt_id}'.",
                file=sys.stderr,
            )
            return 1
    else:
        if not getattr(args, "all", False):
            print(
                "Pass --all to delete every snapshot for the prompt.",
                file=sys.stderr,
            )
            return 1
        count = store.delete_prompt(args.prompt_id)
        print(f"Deleted {count} snapshot(s) for '{args.prompt_id}'.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search snapshots by model, date range, or keyword."""
    from datetime import datetime, timezone

    store = _get_store()

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Invalid --since date '{args.since}'. Use ISO format.", file=sys.stderr)
            return 1

    until = None
    if args.until:
        try:
            until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Invalid --until date '{args.until}'. Use ISO format.", file=sys.stderr)
            return 1

    results = store.search_snapshots(
        prompt_id=args.prompt_id or None,
        model=args.model or None,
        since=since,
        until=until,
        keyword=args.keyword or None,
    )

    if not results:
        print("No snapshots matched the search criteria.")
        return 0

    print(f"{'Prompt ID':<30} {'Ver':<8} {'Model':<20} {'Timestamp'}")
    print("-" * 75)
    for s in results:
        ts = s.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{s.prompt_id:<30} {s.version:<8} {s.model:<20} {ts}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show aggregate statistics across all snapshots."""
    store = _get_store()
    stats = store.get_stats()
    if args.format == "json":
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Total prompts:        {stats['total_prompts']}")
        print(f"Total snapshots:      {stats['total_snapshots']}")
        print(f"Avg output length:    {stats['avg_output_length']} chars")
        if stats.get("oldest"):
            print(f"Oldest snapshot:      {stats['oldest']}")
            print(f"Newest snapshot:      {stats['newest']}")
        if stats.get("models"):
            print("Models used:")
            for model, count in sorted(stats["models"].items()):
                print(f"  {model:<30} {count} snapshot(s)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export all snapshots to a ZIP archive."""
    store = _get_store()
    count = store.export_archive(args.output)
    print(f"Exported {count} snapshot file(s) to '{args.output}'.")
    return 0


def cmd_import_archive(args: argparse.Namespace) -> int:
    """Import snapshots from a ZIP archive."""
    if not os.path.isfile(args.archive):
        print(f"Error: archive '{args.archive}' not found.", file=sys.stderr)
        return 1
    store = _get_store()
    count = store.import_archive(args.archive, overwrite=args.overwrite)
    print(f"Imported {count} snapshot file(s) from '{args.archive}'.")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a file for PII before committing it as a snapshot."""
    if not os.path.isfile(args.file):
        print(f"Error: file '{args.file}' not found.", file=sys.stderr)
        return 1
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    report = PIIDetector.scan(text)
    if not report.has_pii:
        print("No PII detected.")
        return 0

    print(f"WARNING: {len(report.findings)} PII finding(s) detected:")
    for finding in report.findings:
        snippet = finding.value[:40] + "..." if len(finding.value) > 40 else finding.value
        print(f"  [{finding.label}] at position {finding.start}: {snippet!r}")
    if args.redact:
        redacted_path = args.file + ".redacted"
        with open(redacted_path, "w", encoding="utf-8") as fh:
            fh.write(PIIDetector.redact(text))
        print(f"Redacted copy written to '{redacted_path}'.")
    return 1  # non-zero exit so CI pipelines can act on it


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

    # register
    reg_parser = sub.add_parser(
        "register", help="Register a prompt from a JSON file."
    )
    reg_parser.add_argument(
        "json_file",
        help=(
            "Path to a JSON file containing the prompt definition "
            "(keys: id, name, content, and optionally description, constraints, tags)."
        ),
    )

    # show
    show_parser = sub.add_parser("show", help="Show full details of a snapshot.")
    show_parser.add_argument("prompt_id", help="Prompt ID")
    show_parser.add_argument(
        "--version", default=None, help="Snapshot version (default: latest)"
    )
    show_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # delete
    del_parser = sub.add_parser("delete", help="Delete a snapshot or all snapshots for a prompt.")
    del_parser.add_argument("prompt_id", help="Prompt ID")
    del_parser.add_argument(
        "--version", default=None, help="Delete a specific version only"
    )
    del_parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL snapshots for the prompt (required when --version is omitted)",
    )

    # search
    search_parser = sub.add_parser("search", help="Search snapshots by model, date, or keyword.")
    search_parser.add_argument(
        "--prompt-id", default=None, dest="prompt_id", help="Filter by prompt ID"
    )
    search_parser.add_argument("--model", default=None, help="Filter by model name (substring)")
    search_parser.add_argument(
        "--since", default=None, help="Include snapshots after this date (ISO 8601)"
    )
    search_parser.add_argument(
        "--until", default=None, help="Include snapshots before this date (ISO 8601)"
    )
    search_parser.add_argument(
        "--keyword", default=None, help="Search term in prompt content or output"
    )

    # stats
    stats_parser = sub.add_parser("stats", help="Show aggregate statistics.")
    stats_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # export
    export_parser = sub.add_parser("export", help="Export all snapshots to a ZIP archive.")
    export_parser.add_argument("output", help="Path for the output ZIP file")

    # import-archive
    import_parser = sub.add_parser(
        "import-archive", help="Import snapshots from a ZIP archive."
    )
    import_parser.add_argument("archive", help="Path to the ZIP archive")
    import_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing snapshots (default: skip)",
    )

    # scan
    scan_parser = sub.add_parser(
        "scan", help="Scan a file for PII before snapshotting it."
    )
    scan_parser.add_argument("file", help="Path to the file to scan")
    scan_parser.add_argument(
        "--redact",
        action="store_true",
        help="Write a redacted copy alongside the original",
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
        "register": cmd_register,
        "show": cmd_show,
        "delete": cmd_delete,
        "search": cmd_search,
        "stats": cmd_stats,
        "export": cmd_export,
        "import-archive": cmd_import_archive,
        "scan": cmd_scan,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
