"""Hybrid search combining embedding-based and BM25 search with RRF fusion."""

from typing import Optional

from backend.config import get_settings
from backend.services.search.bm25_search import get_bm25_index


class HybridSearcher:
    """Combines embedding search with BM25 using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid searcher.

        Args:
            rrf_k: RRF constant (default 60, standard value from literature)
        """
        self.rrf_k = rrf_k
        self.settings = get_settings()
        self._bm25_index = get_bm25_index()

    def reciprocal_rank_fusion(
        self,
        embedding_results: list[dict],
        bm25_results: list[dict],
        embedding_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each ranking list

        Args:
            embedding_results: Results from embedding search (ordered by similarity)
            bm25_results: Results from BM25 search (ordered by BM25 score)
            embedding_weight: Weight for embedding results (default 0.6)
            bm25_weight: Weight for BM25 results (default 0.4)

        Returns:
            Fused results ordered by RRF score
        """
        # Build score dict by document ID
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        # Process embedding results
        for rank, result in enumerate(embedding_results, start=1):
            doc_id = result.get('id', str(rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + embedding_weight / (self.rrf_k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result.copy()
            # Preserve embedding similarity
            doc_map[doc_id]['embedding_similarity'] = result.get('similarity', 0)

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get('id', str(hash(result.get('document', '')[:100])))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (self.rrf_k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result.copy()
            # Preserve BM25 score
            doc_map[doc_id]['bm25_score'] = result.get('bm25_score', 0)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc['rrf_score'] = rrf_scores[doc_id]
            # Use RRF score as similarity for consistent interface
            doc['similarity'] = rrf_scores[doc_id]
            results.append(doc)

        return results

    async def search_code_hybrid(
        self,
        query: str,
        embedding_results: list[dict],
        n_results: int = 10,
        embedding_weight: float = 0.6,
    ) -> list[dict]:
        """
        Perform hybrid search on code.

        Args:
            query: Search query
            embedding_results: Pre-computed embedding search results
            n_results: Number of final results to return
            embedding_weight: Weight for embedding results (BM25 gets 1 - embedding_weight)

        Returns:
            Fused search results
        """
        # Get BM25 results
        bm25_results = self._bm25_index.search_code(query, n_results=n_results * 2)

        if not bm25_results:
            # Fall back to embedding-only results
            return embedding_results[:n_results]

        # Fuse results
        bm25_weight = 1.0 - embedding_weight
        fused = self.reciprocal_rank_fusion(
            embedding_results,
            bm25_results,
            embedding_weight=embedding_weight,
            bm25_weight=bm25_weight,
        )

        return fused[:n_results]

    async def search_paper_hybrid(
        self,
        query: str,
        embedding_results: list[dict],
        n_results: int = 10,
        embedding_weight: float = 0.6,
    ) -> list[dict]:
        """
        Perform hybrid search on paper content.

        Args:
            query: Search query
            embedding_results: Pre-computed embedding search results
            n_results: Number of final results to return
            embedding_weight: Weight for embedding results

        Returns:
            Fused search results
        """
        # Get BM25 results
        bm25_results = self._bm25_index.search_paper(query, n_results=n_results * 2)

        if not bm25_results:
            return embedding_results[:n_results]

        # Fuse results
        bm25_weight = 1.0 - embedding_weight
        fused = self.reciprocal_rank_fusion(
            embedding_results,
            bm25_results,
            embedding_weight=embedding_weight,
            bm25_weight=bm25_weight,
        )

        return fused[:n_results]


# Global hybrid searcher instance
_hybrid_searcher: Optional[HybridSearcher] = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get the global hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher
