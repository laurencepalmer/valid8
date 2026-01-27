"""Cross-encoder re-ranking for improved search result quality."""

from typing import Optional

from backend.config import get_settings


class Reranker:
    """Re-rank search results using a cross-encoder model."""

    def __init__(self):
        self._model = None
        self._device = None
        self.settings = get_settings()

    @property
    def device(self):
        """Detect and cache the best available device."""
        if self._device is None:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        return self._device

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None and self.settings.use_reranking:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.settings.cross_encoder_model,
                device=self.device
            )
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None,
        document_key: str = 'document',
    ) -> list[dict]:
        """
        Re-rank results using cross-encoder.

        Args:
            query: Original search query
            results: List of search results to re-rank
            top_k: Number of results to return (default: settings.rerank_top_k or len(results))
            document_key: Key in result dict containing document text

        Returns:
            Re-ranked results with cross-encoder scores
        """
        if not self.settings.use_reranking or not results:
            return results

        if self.model is None:
            return results

        # Determine how many to rerank and return
        rerank_k = self.settings.rerank_top_k
        if top_k is None:
            top_k = min(5, len(results))  # Default to top 5

        # Only rerank top candidates
        candidates = results[:rerank_k]

        if not candidates:
            return results

        # Prepare query-document pairs for cross-encoder
        pairs = [
            (query, candidate.get(document_key, ''))
            for candidate in candidates
        ]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Attach scores to results
        for i, score in enumerate(scores):
            candidates[i]['cross_encoder_score'] = float(score)

        # Sort by cross-encoder score
        reranked = sorted(candidates, key=lambda x: x.get('cross_encoder_score', 0), reverse=True)

        # Return top_k results, using cross-encoder score as similarity
        final_results = []
        for result in reranked[:top_k]:
            result_copy = result.copy()
            # Preserve original similarity but update with reranking score for display
            result_copy['original_similarity'] = result_copy.get('similarity', 0)
            result_copy['similarity'] = result_copy.get('cross_encoder_score', result_copy.get('similarity', 0))
            final_results.append(result_copy)

        return final_results

    def rerank_for_code(
        self,
        highlighted_text: str,
        code_results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank code search results for paper highlight queries.

        Args:
            highlighted_text: Highlighted text from paper
            code_results: Code search results
            top_k: Number of results to return

        Returns:
            Re-ranked code results
        """
        return self.rerank(
            query=highlighted_text,
            results=code_results,
            top_k=top_k,
            document_key='document',
        )

    def rerank_for_paper(
        self,
        code_snippet: str,
        paper_results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank paper search results for code highlight queries.

        Args:
            code_snippet: Code snippet being analyzed
            paper_results: Paper section search results
            top_k: Number of results to return

        Returns:
            Re-ranked paper results
        """
        return self.rerank(
            query=code_snippet,
            results=paper_results,
            top_k=top_k,
            document_key='document',
        )


# Global reranker instance
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
