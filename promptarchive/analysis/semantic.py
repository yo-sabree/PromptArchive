"""Semantic similarity analysis using local embeddings (sentence-transformers)
with a lightweight TF-IDF cosine fallback when sentence-transformers is absent."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SemanticResult:
    """Result of a semantic similarity comparison."""

    similarity_score: float  # 0.0 – 1.0
    drift_level: str  # "none" | "minimal" | "moderate" | "significant" | "extreme"
    direction: str  # "increased" | "decreased" | "stable"
    method: str  # "embeddings" | "tfidf"
    # Lexical precision/recall: independent of embedding similarity so that
    # surface-topic similarity cannot mask content-level regressions.
    precision_score: float = 1.0  # fraction of new-output tokens found in old output
    recall_score: float = 1.0    # fraction of old-output tokens preserved in new output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_score": round(self.similarity_score, 4),
            "drift_level": self.drift_level,
            "direction": self.direction,
            "method": self.method,
            "precision_score": round(self.precision_score, 4),
            "recall_score": round(self.recall_score, 4),
        }


def _drift_level(similarity: float, precision: float, recall: float) -> str:
    """Derive drift level from similarity AND lexical precision/recall.

    Pure cosine/embedding similarity can remain high even when the new output
    covers very different content (same topic, different facts).  We therefore
    use the *minimum* of similarity, precision and recall as the governing
    signal so that a precision collapse is never masked.
    """
    effective = min(similarity, precision, recall)
    if effective >= 0.95:
        return "none"
    if effective >= 0.85:
        return "minimal"
    if effective >= 0.70:
        return "moderate"
    if effective >= 0.50:
        return "significant"
    return "extreme"


def _direction(old_score: float, new_score: float) -> str:
    """Direction relative to perfect identity (1.0)."""
    # 'increased' means the new output is MORE similar to old than a baseline.
    # For a single comparison we report whether similarity increased vs previous.
    # Without prior history we describe how the NEW output changed vs OLD.
    # We treat the old output as the reference and describe drift direction.
    if new_score > old_score + 0.01:
        return "increased"
    if new_score < old_score - 0.01:
        return "decreased"
    return "stable"


# ---------------------------------------------------------------------------
# TF-IDF cosine fallback (no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    total = len(tokens) or 1
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1 / total
    return [tf.get(w, 0.0) for w in vocab]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0 if mag_a == mag_b == 0 else 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def _tfidf_similarity(text1: str, text2: str) -> float:
    tok1 = _tokenize(text1)
    tok2 = _tokenize(text2)
    vocab = sorted(set(tok1) | set(tok2))
    if not vocab:
        return 1.0
    v1 = _tfidf_vector(tok1, vocab)
    v2 = _tfidf_vector(tok2, vocab)
    return _cosine(v1, v2)


def _lexical_precision_recall(old_text: str, new_text: str) -> tuple:
    """Return (precision, recall) based on token-level overlap.

    precision = |new ∩ old| / |new|   (how much of new is in old)
    recall    = |new ∩ old| / |old|   (how much of old is preserved in new)

    Both are token-set overlap so they complement cosine similarity by
    capturing *content coverage*, not just directional similarity.
    """
    old_tokens = set(_tokenize(old_text))
    new_tokens = set(_tokenize(new_text))
    if not old_tokens and not new_tokens:
        return 1.0, 1.0
    intersection = old_tokens & new_tokens
    precision = len(intersection) / len(new_tokens) if new_tokens else 0.0
    recall = len(intersection) / len(old_tokens) if old_tokens else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    """Semantic similarity between two LLM outputs."""

    _model: Optional[Any] = None
    _model_loaded: bool = False

    @classmethod
    def _get_model(cls) -> Optional[Any]:
        if cls._model_loaded:
            return cls._model
        cls._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            cls._model = None
        return cls._model

    @classmethod
    def analyze(cls, old_output: str, new_output: str) -> SemanticResult:
        """Compute semantic similarity between two outputs."""
        model = cls._get_model()
        if model is not None:
            try:
                import numpy as np  # type: ignore

                embeddings = model.encode([old_output, new_output])
                score = float(
                    np.dot(embeddings[0], embeddings[1])
                    / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10)
                )
                score = max(0.0, min(1.0, score))
                method = "embeddings"
            except Exception:
                score = _tfidf_similarity(old_output, new_output)
                method = "tfidf"
        else:
            score = _tfidf_similarity(old_output, new_output)
            method = "tfidf"

        precision, recall = _lexical_precision_recall(old_output, new_output)

        return SemanticResult(
            similarity_score=score,
            drift_level=_drift_level(score, precision, recall),
            direction=_direction(1.0, score),
            method=method,
            precision_score=precision,
            recall_score=recall,
        )
