"""Read and normalize source data while preserving physical-page provenance."""
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> list[DocumentPage]:
    """Extract text from each physical page, logging unreadable and empty pages."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf; install starter/requirements.txt."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception:
            logger.exception(
                "Could not extract document_id=%s page=%d from %s",
                document_id,
                page_number,
                path,
            )
            raise
        if not text:
            logger.warning(
                "No extractable text: document_id=%s page=%d source=%s",
                document_id,
                page_number,
                source,
            )
            continue
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=dict(metadata or {}),
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding paragraph breaks."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()
