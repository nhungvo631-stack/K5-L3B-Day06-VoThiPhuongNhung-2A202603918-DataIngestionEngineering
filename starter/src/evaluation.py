"""Compute evidence-based retrieval metrics and retain per-query traces."""
import re
import unicodedata
from collections import defaultdict
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold_records = _gold_records(question)
    traced_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        matched_gold = next(
            (gold for gold in gold_records if _matches_evidence(hit, gold)),
            None,
        )
        traced_hits.append(
            {
                **hit,
                "rank": rank,
                "evidence_hit": matched_gold is not None,
                "matched_gold": matched_gold,
            }
        )
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "filters": question.get("filter") or {},
        "gold_evidence": gold_records,
        "hits": traced_hits,
        "hit_at_1": bool(traced_hits and traced_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in traced_hits[:3]),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)
    return {
        f"{strategy}_{split}": {
            "n": len(items),
            "hit_at_1": {
                "count": sum(bool(item["hit_at_1"]) for item in items),
                "rate": sum(bool(item["hit_at_1"]) for item in items) / len(items),
            },
            "evidence_hit_at_3": {
                "count": sum(bool(item["evidence_hit_at_3"]) for item in items),
                "rate": sum(bool(item["evidence_hit_at_3"]) for item in items) / len(items),
            },
        }
        for (strategy, split), items in sorted(groups.items())
        if items
    }


def _gold_records(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives is not None:
        if not alternatives:
            raise ValueError(f"Question {question.get('id')} has no gold evidence.")
        return alternatives
    required = ("gold_source_id", "gold_page", "evidence_quote")
    if any(key not in question for key in required):
        raise ValueError(f"Question {question.get('id')} is missing gold evidence fields.")
    return [
        {
            "gold_source_id": question["gold_source_id"],
            "gold_page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
        }
    ]


def _matches_evidence(hit: dict[str, Any], gold: dict[str, Any]) -> bool:
    metadata = hit.get("metadata") or {}
    source_id = gold.get("gold_source_id")
    page_id = gold.get("gold_page")
    source_matches = (
        source_id is None
        or metadata.get("document_id") == source_id
        or metadata.get("source") == source_id
    )
    try:
        page_matches = page_id is None or int(metadata.get("page")) == int(page_id)
    except (TypeError, ValueError):
        page_matches = False
    quote = _normalized(gold.get("evidence_quote", ""))
    text = _normalized(hit.get("text", ""))
    return bool(source_matches and page_matches and quote and quote in text)


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", value).strip()
