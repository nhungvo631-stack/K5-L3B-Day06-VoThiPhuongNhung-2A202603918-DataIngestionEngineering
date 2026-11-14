"""Run the fixed A/B/C ingestion and retrieval benchmark."""
import json
import os
import time
from pathlib import Path
from typing import Any

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import ChromaVectorStore

STARTER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COHERENCE_SAMPLE_PAGES = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Configuration loading requires python-dotenv.") from exc
    load_dotenv(STARTER_DIR / ".env")

    data_dir = STARTER_DIR / "data"
    results_dir = STARTER_DIR / "results"
    pdf_path = Path(os.getenv("CORPUS_PDF", str(data_dir / "FBL_PassItOn_Stories_Volume_2.pdf")))
    manifest_path = Path(os.getenv("CORPUS_MANIFEST", str(data_dir / "manifest.json")))
    questions_path = Path(os.getenv("BENCHMARK_PATH", str(data_dir / "questions.json")))
    persist_directory = Path(
        os.getenv("CHROMA_PATH", str(results_dir / "chroma"))
    )
    for path in (pdf_path, manifest_path, questions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {path}")

    manifest = _read_json(manifest_path)
    questions = _read_json(questions_path)
    content_pages = manifest["content_pages"]
    _validate_benchmark(questions, content_pages)

    source_id = manifest["source_id"]
    source_url = manifest["source_url"]
    all_pages = load_pdf(pdf_path, source_id, source_url)
    content_page_map = {int(item["page"]): item for item in content_pages}
    pages = [
        _with_story_metadata(page, content_page_map[page.page])
        for page in all_pages
        if page.page in content_page_map
    ]
    if len(pages) != len(content_pages):
        found = {page.page for page in pages}
        missing = sorted(set(content_page_map) - found)
        raise ValueError(f"Manifest pages were not extractable from the PDF: {missing}")
    if len(pages) < 30:
        raise ValueError(
            f"Corpus has {len(pages)} eligible content pages; at least 30 are required."
        )
    _validate_evidence(questions, pages)

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    embedder = MyEmbedder(model_name, device=os.getenv("EMBEDDING_DEVICE", "cpu"))
    store = ChromaVectorStore(persist_directory, embedder.dimensions)
    store.create_schema()

    configurations = {
        "A": lambda page: fixed_size(page, size=500, overlap=50),
        "B": lambda page: by_paragraph(page, max_size=700),
        "C": lambda page: by_paragraph(page, max_size=700),
    }
    vector_cache: dict[str, list[float]] = {}
    strategy_chunks: dict[str, list[Chunk]] = {}
    strategy_stats: dict[str, dict[str, Any]] = {}
    total_ingest_start = time.perf_counter()
    for strategy, chunk_page in configurations.items():
        chunks = [chunk for page in pages for chunk in chunk_page(page)]
        if not chunks:
            raise ValueError(f"Strategy {strategy} produced no chunks.")
        started = time.perf_counter()
        missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in vector_cache]
        if missing_chunks:
            new_vectors = embedder.embed_documents([chunk.text for chunk in missing_chunks])
            vector_cache.update(
                (chunk.chunk_id, vector)
                for chunk, vector in zip(missing_chunks, new_vectors)
            )
        vectors = [vector_cache[chunk.chunk_id] for chunk in chunks]
        store.upsert(strategy, chunks, vectors)
        elapsed = time.perf_counter() - started
        lengths = [len(chunk.text) for chunk in chunks]
        strategy_chunks[strategy] = chunks
        strategy_stats[strategy] = {
            "chunk_count": len(chunks),
            "characters": sum(lengths),
            "mean_chunk_characters": sum(lengths) / len(lengths),
            "min_chunk_characters": min(lengths),
            "max_chunk_characters": max(lengths),
            "ingest_and_index_seconds": round(elapsed, 3),
            "embedding_calls_reused": strategy == "C",
        }

    traces: list[dict[str, Any]] = []
    for question in questions:
        query_vector = embedder.embed_query(question["question"])
        for strategy in configurations:
            applied_filter = question.get("filter") if strategy == "C" else None
            hits = store.search(
                strategy,
                query_vector,
                top_k=3,
                filters=applied_filter,
            )
            trace = evaluate_query(
                {**question, "filter": applied_filter},
                hits,
            )
            trace["strategy"] = strategy
            trace["applied_filter"] = applied_filter or {}
            traces.append(trace)

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(results_dir / "traces.jsonl", traces)
    retrieval_summary = summarize(traces)
    _write_json(
        results_dir / "retrieval_summary.json",
        retrieval_summary,
    )
    _write_json(
        results_dir / "run_metadata.json",
        {
            "source_id": source_id,
            "source_url": source_url,
            "content_page_count": len(pages),
            "content_page_ids": sorted(content_page_map),
            "model": model_name,
            "embedding_dimensions": embedder.dimensions,
            "embedding_device": os.getenv("EMBEDDING_DEVICE", "cpu"),
            "vector_store": store.backend_info(),
            "metric": "cosine",
            "retrieval_top_k": 3,
            "cost_usd": 0,
            "total_ingest_and_index_seconds": round(
                time.perf_counter() - total_ingest_start, 3
            ),
            "strategies": strategy_stats,
            "selection_rule": (
                "Choose on dev by Evidence Hit@3; use grounding and resource cost "
                "as tie-breakers. Test is held out and must not be used for tuning."
            ),
            "manual_audit_status": "pending human labels; no labels are inferred",
        },
    )
    _write_manual_audit_template(
        results_dir / "manual_audit_template.json",
        pages,
        strategy_chunks,
        traces,
    )
    print(json.dumps(retrieval_summary, indent=2))
    print(f"Saved top-3 traces and audit template under {results_dir}")


def _with_story_metadata(page: DocumentPage, manifest_page: dict[str, Any]) -> DocumentPage:
    page.metadata.update(
        {
            "story_id": int(manifest_page["story_id"]),
            "story_title": manifest_page["story_title"],
        }
    )
    return page


def _validate_benchmark(
    questions: list[dict[str, Any]],
    content_pages: list[dict[str, Any]],
) -> None:
    if len(questions) < 20:
        raise ValueError(f"Benchmark has {len(questions)} questions; at least 20 are required.")
    counts = {
        split: sum(question.get("split") == split for question in questions)
        for split in ("dev", "test")
    }
    if counts["dev"] < 15 or counts["test"] < 5:
        raise ValueError(f"Benchmark requires at least 15 dev / 5 test questions; found {counts}.")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Benchmark question IDs must be unique.")
    filter_questions = [question for question in questions if question.get("filter")]
    if len(filter_questions) < 3 or not any(
        question["split"] == "test" for question in filter_questions
    ):
        raise ValueError("Benchmark requires 3 metadata-filter questions, including a test item.")
    page_ids = {int(item["page"]) for item in content_pages}
    if len(page_ids) < 30:
        raise ValueError(f"Manifest has only {len(page_ids)} eligible content pages.")


def _validate_evidence(
    questions: list[dict[str, Any]],
    pages: list[DocumentPage],
) -> None:
    page_map = {page.page: page for page in pages}
    for question in questions:
        page_number = int(question["gold_page"])
        page = page_map.get(page_number)
        if page is None:
            raise ValueError(
                f"Question {question['id']} gold page {page_number} is not in the corpus."
            )
        if question["gold_source_id"] != page.document_id:
            raise ValueError(f"Question {question['id']} gold source does not match the corpus.")
        if not _contains_normalized(page.text, question["evidence_quote"]):
            raise ValueError(
                f"Question {question['id']} evidence_quote was not found on "
                f"physical page {page_number}."
            )
        if question.get("filter"):
            expected_story = int(question["filter"].get("story_id", -1))
            if expected_story != int(page.metadata["story_id"]):
                raise ValueError(
                    f"Question {question['id']} filter story_id does not match its stated gold page."
                )


def _contains_normalized(text: str, phrase: str) -> bool:
    import re
    import unicodedata

    normalize = lambda value: re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", value).casefold(),
    ).strip()
    return bool(phrase and normalize(phrase) in normalize(text))


def _write_manual_audit_template(
    path: Path,
    pages: list[DocumentPage],
    strategy_chunks: dict[str, list[Chunk]],
    traces: list[dict[str, Any]],
) -> None:
    page_map = {page.page: page for page in pages}
    coherence: list[dict[str, Any]] = []
    for page_number in COHERENCE_SAMPLE_PAGES:
        page = page_map.get(page_number)
        if page is None:
            raise ValueError(f"Preselected coherence page {page_number} is not in corpus.")
        source_offset = min(150, len(page.text) - 1)
        for strategy, chunks in strategy_chunks.items():
            chunk = next(
                (
                    candidate
                    for candidate in chunks
                    if candidate.page == page_number
                    and candidate.start_char <= source_offset < candidate.end_char
                ),
                None,
            )
            if chunk is None:
                raise ValueError(
                    f"Strategy {strategy} has no chunk covering page {page_number}, "
                    f"offset {source_offset}."
                )
            coherence.append(
                {
                    "strategy": strategy,
                    "page": page_number,
                    "source_offset_char": source_offset,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "score_0_2": None,
                    "rationale": "",
                }
            )

    test_ids = {
        question_id
        for question_id, split in _question_splits(traces).items()
        if split == "test"
    }
    metadata_audit: list[dict[str, Any]] = []
    grounding: list[dict[str, Any]] = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        for hit in trace["hits"]:
            metadata_audit.append(
                {
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "rank": hit["rank"],
                    "chunk_id": hit["chunk_id"],
                    "metadata": hit["metadata"],
                    "metadata_complete_and_correct": None,
                    "filter_compliant": None,
                    "rationale": "",
                }
            )
        grounding.append(
            {
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "top3_chunk_ids": [hit["chunk_id"] for hit in trace["hits"][:3]],
                "score_0_2": None,
                "rationale": "",
            }
        )
    _write_json(
        path,
        {
            "instructions": (
                "Fill labels manually after reviewing the source and retrieved chunks. "
                "Null scores are intentionally blank; do not treat automated matches "
                "as human grounding or coherence labels."
            ),
            "coherence": coherence,
            "metadata_top3_test": metadata_audit,
            "grounding_test": grounding,
            "test_question_ids": sorted(test_ids),
        },
    )


def _question_splits(traces: list[dict[str, Any]]) -> dict[str, str]:
    return {trace["question_id"]: trace["split"] for trace in traces}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
