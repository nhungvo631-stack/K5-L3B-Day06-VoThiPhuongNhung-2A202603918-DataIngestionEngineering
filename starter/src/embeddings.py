"""Local Hugging Face transformer embeddings with mean pooling and normalization."""
import math
from typing import Protocol, Sequence


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with stored document vectors."""
        ...


class MyEmbedder:
    """Generate real dense vectors locally; failures do not fall back to mock vectors."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require PyTorch and Transformers in the active "
                "Python environment."
            ) from exc
        if not model_name.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty.")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("EMBEDDING_DEVICE must be 'cpu' or a CUDA device.")
        self.model_name = model_name
        self._torch = torch
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        dimensions = self._model.config.hidden_size
        if not isinstance(dimensions, int) or dimensions < 1:
            raise RuntimeError(f"Model {model_name!r} did not report valid dimensions.")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings.")

        result: list[list[float]] = []
        for offset in range(0, len(texts), 32):
            batch = list(texts[offset : offset + 32])
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self._device) for name, tensor in tokens.items()}
            with self._torch.inference_mode():
                token_vectors = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
                sentence_vectors = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                sentence_vectors = self._torch.nn.functional.normalize(
                    sentence_vectors,
                    p=2,
                    dim=1,
                )
            result.extend(
                [[float(value) for value in vector] for vector in sentence_vectors.cpu().tolist()]
            )

        if len(result) != len(texts):
            raise RuntimeError("Embedding model returned an unexpected number of vectors.")
        if any(
            len(vector) != self.dimensions
            or any(not math.isfinite(value) for value in vector)
            for vector in result
        ):
            raise RuntimeError("Embedding model returned an invalid vector.")
        return result
