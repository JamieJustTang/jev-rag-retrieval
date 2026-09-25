# jev-rag-retrieval

Find the session, then find the passage that matters.

This small Python tool ranks a supplied set of session records in two stages. Local BM25 creates a shortlist. Optional [Jev / TypeSafe System One](https://docs.typesafe.ai/cookbooks/rerank_typesafe) scores each candidate against the actual question: first sessions, then bounded passages inside the selected sessions. The result retains the source `workref` and chunk offsets, so an agent can open the original record before answering.

Jev **reranks** candidates. It cannot recover a session absent from the input. The caller should gather a generous, diverse shortlist using its archive search, aliases, dates and workspace browsing. Apply hard time, workspace and source filters before passing records here. No fixed score threshold decides truth.

## Install

```bash
uv tool install 'git+https://github.com/JamieJustTang/jev-rag-retrieval.git' --with typesafe-sdk
```

Without Jev, install the same tool without `--with typesafe-sdk`; local ranking still works. For Jev, set `TYPESAFE_API_KEY` in your own environment. Do not store it in this repository, command arguments or a session export. The official [TypeSafe Python SDK guide](https://docs.typesafe.ai/sdk/python) describes the key and client. Configuring a key opts in to sending the query and selected excerpts to the TypeSafe API. Each excerpt is bounded to 2,500 characters. Common credential patterns and the configured Jev key are redacted before transmission; this cannot detect every possible secret, so select records carefully. The tool never sends a complete archive automatically.

## Input and output

The quickest route from sivtr is its structured search output. Keep the temporary file private because it contains original conversation data:

```bash
umask 077
sivtr s all:agent 'parser regression' --last 3d --limit 30 --json > /private/path/sivtr-hits.json 2> /private/path/sivtr-errors.log
jev-rag --query 'Was the parser regression fixed?' --input /private/path/sivtr-hits.json --compact
```

The tool reads sivtr's `records` array, extracts user and assistant messages from each WorkRecord, redacts common credential formats, and retains each record's WorkRef. It does not send tool action payloads by default. Broaden the sivtr search or open more records when a test result only appears in tool output.

For other archives, prepare a JSON array or JSONL file of records already within the requested scope:

```json
{"session_id":"s1","workspace":"docs","workref":"@hits[1]","text":"The parser test failed; the fix was verified.","title":"Parser repair","timestamp":"2026-09-25T10:00:00+08:00"}
```

`session_id`, `workref` and `text` are required. `workspace`, `title` and `timestamp` are optional. `workref` must be an openable source reference from the caller's archive. Keep exported session data in a private temporary file, then remove it when finished.

```bash
jev-rag --query 'Why did the parser test fail, and was it fixed?' --input records.jsonl --session-limit 8
```

Use `--input -` for stdin. Use `--no-jev` for fully local ranking. `--compact` bounds each passage preview to 300 characters and redacts common credential formats; use it when an agent reads stdout. The JSON result contains ordered `sessions`, each with ordered `passages`, `workref`, `part` and (for split records) character offsets. `candidate_sessions`, `input_records`, `mode` and `warnings` reveal coverage and fallback. The tool ranks evidence; it does not synthesize an answer or claim a test passed.

## Retrieval flow

1. Search the archive broadly with lexical queries, aliases and time browsing. Keep the source references.
2. Export relevant records with a stable session ID. Do not pass irrelevant private records.
3. Run this tool. Its local BM25 stage chooses up to 30 sessions and 80 passages by default; Jev scores those pairs when configured.
4. Open the original references for the ranked passages, read enough surrounding context and verify any final claim. Expand the candidate pool when coverage is thin.

The JSON schema is archive agnostic. It works with [sivtr](https://github.com/Ariestar/sivtr), a different memory index, or an application that exports sessions. Jev errors fall back to local ranking with a warning type. For repeated research, avoid re-sending unchanged private excerpts unless needed. A WorkRef from an actively changing session can become stale; resync the archive and reopen it before treating the result as evidence.

## Development

```bash
uv run --extra jev python -m unittest discover -s tests -v
```

The design follows the official [TypeSafe reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe) and the `Noul` criteria pattern. This project is an independent integration; it is not an official TypeSafe package.
