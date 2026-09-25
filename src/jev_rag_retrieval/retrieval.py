"""High-recall local shortlist followed by optional Jev pairwise reranking."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


def tokens(value: str) -> list[str]:
    words = re.findall(r"[a-z0-9_./-]+|[\u3400-\u9fff]+", value.lower())
    out: list[str] = []
    for word in words:
        if re.search(r"[\u3400-\u9fff]", word):
            out.extend([word[i : i + 2] for i in range(max(0, len(word) - 1))])
            if len(word) == 1:
                out.append(word)
        else:
            out.append(word)
    return out


def bm25(query: str, documents: list[str]) -> list[float]:
    """Small, dependency-free first pass; caller may supply its own shortlist."""
    q = set(tokens(query))
    if not q or not documents:
        return [0.0] * len(documents)
    corpus = [Counter(tokens(doc)) for doc in documents]
    lengths = [sum(c.values()) for c in corpus]
    average = max(1.0, sum(lengths) / len(lengths))
    df = Counter(term for counts in corpus for term in counts)
    scores = []
    for counts, length in zip(corpus, lengths):
        score = 0.0
        for term in q:
            freq = counts[term]
            if freq:
                idf = math.log(1 + (len(corpus) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


def split_passages(record: dict[str, Any], max_chars: int = 1400, overlap: int = 180) -> list[dict[str, Any]]:
    """Bound remote excerpts while keeping each segment tied to its source WorkRef."""
    content = record["text"]
    if len(content) <= max_chars:
        return [dict(record, part=1)]
    chunks = []
    start = 0
    part = 1
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            boundary = content.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        chunks.append(dict(record, text=content[start:end], part=part, char_start=start, char_end=end))
        if end == len(content):
            break
        start = max(start + 1, end - overlap)
        part += 1
    return chunks


def _jev_score(query: str, candidate: str, stage: str) -> float:
    from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

    noun = "session" if stage == "session" else "passage"
    question = Noul(
        instructions=f"Does this {noun} contain specific evidence useful for answering the user's request?",
        criteria=NoulCriteria(
            true="It directly supports an answer, decision, outcome, change, failure, or required context for the request, even with different wording.",
            false="It only shares broad vocabulary, is tool noise, or has no concrete bearing on the request.",
        ),
    )
    with TypeSafeClient(timeout=45.0) as client:
        response = client.system_one(
            state={"user_request": query[:1500], "candidate": candidate[:2500]},
            questions={"relevant": question},
        )
    answer = getattr(response, "answers", None) or getattr(response, "nouls", None)
    return float(answer["relevant"].noul)


def _rank(
    query: str,
    items: list[dict[str, Any]],
    text_of: Callable[[dict[str, Any]], str],
    *,
    stage: str,
    shortlist: int,
    use_jev: bool,
    scorer: Callable[[str, str, str], float] | None,
    workers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    lexical = bm25(query, [text_of(item) for item in items])
    ranked = sorted(zip(items, lexical), key=lambda pair: pair[1], reverse=True)[:shortlist]
    result = [dict(item, lexical_score=round(score, 6), jev_score=None) for item, score in ranked]
    warnings: list[str] = []
    if not use_jev or not result:
        return result, warnings
    score_fn = scorer or _jev_score
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        futures = {pool.submit(score_fn, query, text_of(item), stage): i for i, item in enumerate(result)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                score = float(future.result())
                if not 0 <= score <= 1 or not math.isfinite(score):
                    raise ValueError("score outside [0, 1]")
                result[i]["jev_score"] = round(score, 6)
            except Exception as exc:
                warnings.append(f"Jev {stage} scoring failed for candidate {i + 1}: {type(exc).__name__}")
    result.sort(key=lambda item: (item["jev_score"] is not None, item["jev_score"] if item["jev_score"] is not None else item["lexical_score"]), reverse=True)
    return result, warnings


def retrieve(
    query: str,
    records: list[dict[str, Any]],
    *,
    session_candidates: int = 30,
    passage_candidates: int = 80,
    session_limit: int = 8,
    passages_per_session: int = 3,
    use_jev: bool | None = None,
    scorer: Callable[[str, str, str], float] | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Return ranked candidates and stable refs; the caller must verify original evidence.

    Input records have session_id, text and workref; workspace/title/timestamp are optional.
    Apply time, workspace and source constraints before calling this function.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if any(x < 1 for x in (session_candidates, passage_candidates, session_limit, passages_per_session)):
        raise ValueError("candidate and output limits must be positive")
    if use_jev is None:
        use_jev = bool(os.environ.get("TYPESAFE_API_KEY"))
    if use_jev and scorer is None and not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("TYPESAFE_API_KEY is required for Jev")
    clean: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for record in records:
        if not all(isinstance(record.get(field), str) and record[field] for field in ("session_id", "text", "workref")):
            raise ValueError("each record needs nonempty session_id, text and workref strings")
        if record["workref"] in seen_refs:
            continue
        seen_refs.add(record["workref"])
        clean.append(record)
    segments = [part for record in clean for part in split_passages(record)]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in segments:
        groups[(str(record.get("workspace", "")), record["session_id"])].append(record)
    sessions = []
    for (workspace, session_id), entries in groups.items():
        passage_scores = bm25(query, [entry["text"] for entry in entries])
        best = sorted(zip(entries, passage_scores), key=lambda pair: pair[1], reverse=True)[:3]
        title = str(entries[0].get("title", ""))
        sessions.append({
            "session_id": session_id,
            "workspace": workspace,
            "title": title,
            "timestamp": max((str(entry.get("timestamp") or "") for entry in entries), default=""),
            "evidence_preview": "\n".join(entry["text"][:650] for entry, _ in best)[:2000],
            "record_count": len(entries),
        })
    ranked_sessions, warnings = _rank(
        query, sessions,
        lambda item: f"{item['title']}\n{item['evidence_preview']}",
        stage="session", shortlist=session_candidates, use_jev=use_jev, scorer=scorer, workers=workers,
    )
    selected = ranked_sessions[:session_limit]
    selected_ids = {(item["workspace"], item["session_id"]) for item in selected}
    passage_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in segments:
        key = (str(entry.get("workspace", "")), entry["session_id"])
        if key in selected_ids:
            passage_groups[key].append(entry)
    per_session = max(passages_per_session, passage_candidates // max(1, len(selected)))
    passages = []
    for group in passage_groups.values():
        scores = bm25(query, [entry["text"] for entry in group])
        passages.extend(entry for entry, _ in sorted(zip(group, scores), key=lambda pair: pair[1], reverse=True)[:per_session])
    ranked_passages, passage_warnings = _rank(
        query, passages, lambda item: item["text"], stage="passage",
        shortlist=passage_candidates, use_jev=use_jev, scorer=scorer, workers=workers,
    )
    warnings.extend(passage_warnings)
    chosen: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for passage in ranked_passages:
        key = (str(passage.get("workspace", "")), passage["session_id"])
        if len(chosen[key]) < passages_per_session:
            chosen[key].append(passage)
    output = []
    for session in selected:
        key = (session["workspace"], session["session_id"])
        item = {k: v for k, v in session.items() if k != "evidence_preview"}
        item["passages"] = chosen[key]
        output.append(item)
    return {
        "query": query,
        "mode": ("jev" if any(item["jev_score"] is not None for item in ranked_sessions + ranked_passages) else "lexical_fallback") if use_jev else "lexical",
        "input_records": len(clean),
        "input_passages": len(segments),
        "input_sessions": len(sessions),
        "candidate_sessions": len(ranked_sessions),
        "sessions": output,
        "warnings": warnings,
        "note": "Scores rank supplied candidates only. Open original workrefs before making claims.",
    }
