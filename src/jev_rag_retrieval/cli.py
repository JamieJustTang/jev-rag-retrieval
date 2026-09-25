"""CLI for newline-delimited or array JSON record exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .retrieval import from_sivtr_workset, redact_for_remote, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank sessions and passages from a prefiltered record export")
    parser.add_argument("--query", required=True, help="The user's concrete information need")
    parser.add_argument("--input", required=True, type=Path, help="JSON array, JSONL, sivtr --json WorkSet, or - for stdin")
    parser.add_argument("--session-candidates", type=int, default=30)
    parser.add_argument("--passage-candidates", type=int, default=80)
    parser.add_argument("--session-limit", type=int, default=8)
    parser.add_argument("--passages-per-session", type=int, default=3)
    parser.add_argument("--no-jev", action="store_true", help="Use local lexical ranking only")
    parser.add_argument("--compact", action="store_true", help="Print bounded, credential-redacted passage previews")
    args = parser.parse_args()
    data = sys.stdin.read() if str(args.input) == "-" else args.input.read_text(encoding="utf-8")
    data = data.strip()
    if not data:
        parser.error("input is empty")
    try:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = [json.loads(line) for line in data.splitlines() if line.strip()]
        records = from_sivtr_workset(parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(records, list):
            raise ValueError("input must be a JSON array or JSONL")
        result = retrieve(
            args.query, records,
            session_candidates=args.session_candidates,
            passage_candidates=args.passage_candidates,
            session_limit=args.session_limit,
            passages_per_session=args.passages_per_session,
            use_jev=False if args.no_jev else None,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.compact:
        for session in result["sessions"]:
            session["title"] = redact_for_remote(session["title"][:200])
            for passage in session["passages"]:
                passage["text"] = redact_for_remote(passage["text"][:300])
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
