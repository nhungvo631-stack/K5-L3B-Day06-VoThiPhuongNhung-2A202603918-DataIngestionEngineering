"""Fixed-size and paragraph-aware chunking with source offsets."""
import hashlib
import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into overlapping character windows."""
    if size < 1:
        raise ValueError("size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size.")
    if not page.text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        chunks.append(_make_chunk(page, start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group adjacent paragraphs up to max_size and split oversized paragraphs."""
    if max_size < 1:
        raise ValueError("max_size must be positive.")
    if not page.text:
        return []

    paragraphs = [
        match.span()
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)", page.text, re.DOTALL)
    ]
    grouped: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for start, end in paragraphs:
        if end - start > max_size:
            if pending is not None:
                grouped.append(pending)
                pending = None
            grouped.extend(_split_range(page.text, start, end, max_size))
        elif pending is None:
            pending = (start, end)
        elif end - pending[0] <= max_size:
            pending = (pending[0], end)
        else:
            grouped.append(pending)
            pending = (start, end)
    if pending is not None:
        grouped.append(pending)
    return [_make_chunk(page, start, end) for start, end in grouped]


def _split_range(text: str, start: int, end: int, max_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    while start < end:
        hard_end = min(start + max_size, end)
        if hard_end < end:
            boundary = text.rfind(" ", start + 1, hard_end + 1)
            if boundary > start:
                hard_end = boundary + 1
        ranges.append((start, hard_end))
        start = hard_end
    return ranges


def _make_chunk(page: DocumentPage, start: int, end: int) -> Chunk:
    text = page.text[start:end]
    identity = f"{page.document_id}\0{page.page}\0{start}\0{end}\0{text}"
    return Chunk(
        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=dict(page.metadata),
        start_char=start,
        end_char=end,
    )
