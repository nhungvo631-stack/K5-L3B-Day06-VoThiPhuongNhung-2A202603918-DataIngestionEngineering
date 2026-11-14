"""Persistent Chroma vector search using cosine distance and metadata filters."""
import math
from pathlib import Path
from typing import Any

from .models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, dimensions: int) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive.")
        self.persist_directory = Path(persist_directory)
        self.dimensions = dimensions
        self._client = None

    def create_schema(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chroma storage requires chromadb; install starter/requirements.txt."
            ) from exc
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert current chunks and delete obsolete IDs from this strategy."""
        collection = self._collection(strategy)
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique within a strategy snapshot.")
        for vector in vectors:
            self._validate_vector(vector)

        if chunks:
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in chunks],
            )
        current_ids = set(collection.get()["ids"])
        obsolete_ids = current_ids.difference(ids)
        if obsolete_ids:
            collection.delete(ids=sorted(obsolete_ids))

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_vector(query_vector)
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query["where"] = filters
        result = self._collection(strategy).query(**query)
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": 1.0 - float(distance),
                "metadata": metadata or {},
            }
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def backend_info(self) -> dict[str, str]:
        try:
            from importlib.metadata import version

            chroma_version = version("chromadb")
        except ImportError as exc:
            raise RuntimeError("Chroma is not installed.") from exc
        return {
            "vector_store": "Chroma Persistent",
            "version": chroma_version,
            "metric": "cosine",
            "search": "HNSW approximate",
            "persist_directory": str(self.persist_directory),
        }

    def _collection(self, strategy: str):
        if self._client is None:
            raise RuntimeError("Call create_schema() before using the vector store.")
        collection_name = f"day06_{strategy.lower()}"
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, received {len(vector)}."
            )
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("Embedding values must be finite numbers.")


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            raise ValueError(f"Chroma metadata {key!r} must be a scalar value.")
    metadata.update(
        {
            "document_id": chunk.document_id,
            "source": chunk.source,
            "page": chunk.page,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
    )
    return metadata
