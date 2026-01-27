"""BM25 keyword-based search for code and paper content."""

import re
from typing import Optional
from rank_bm25 import BM25Okapi

from backend.config import get_settings


class BM25Index:
    """BM25 index for keyword-based search."""

    def __init__(self):
        self._code_index: Optional[BM25Okapi] = None
        self._code_documents: list[dict] = []
        self._paper_index: Optional[BM25Okapi] = None
        self._paper_documents: list[dict] = []
        self.settings = get_settings()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing."""
        # Convert to lowercase
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text)
        # Filter very short tokens and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        }
        return [t for t in tokens if len(t) > 1 and t not in stop_words]

    def build_code_index(self, documents: list[dict]) -> None:
        """
        Build BM25 index for code documents.

        Args:
            documents: List of dicts with 'document' (text) and 'metadata' keys
        """
        self._code_documents = documents
        if not documents:
            self._code_index = None
            return

        # Tokenize all documents
        tokenized_docs = [
            self._tokenize(doc.get('document', '') or '')
            for doc in documents
        ]
        self._code_index = BM25Okapi(tokenized_docs)

    def build_paper_index(self, documents: list[dict]) -> None:
        """
        Build BM25 index for paper documents.

        Args:
            documents: List of dicts with 'document' (text) and 'metadata' keys
        """
        self._paper_documents = documents
        if not documents:
            self._paper_index = None
            return

        # Tokenize all documents
        tokenized_docs = [
            self._tokenize(doc.get('document', '') or '')
            for doc in documents
        ]
        self._paper_index = BM25Okapi(tokenized_docs)

    def search_code(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Search code index with BM25.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of matching documents with BM25 scores
        """
        if self._code_index is None or not self._code_documents:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._code_index.get_scores(tokenized_query)

        # Get top N indices sorted by score
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]

        results = []
        for idx, score in scored_indices:
            if score > 0:  # Only include documents with positive scores
                doc = self._code_documents[idx].copy()
                doc['bm25_score'] = float(score)
                results.append(doc)

        return results

    def search_paper(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Search paper index with BM25.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of matching documents with BM25 scores
        """
        if self._paper_index is None or not self._paper_documents:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._paper_index.get_scores(tokenized_query)

        # Get top N indices sorted by score
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]

        results = []
        for idx, score in scored_indices:
            if score > 0:
                doc = self._paper_documents[idx].copy()
                doc['bm25_score'] = float(score)
                results.append(doc)

        return results

    def clear_code_index(self) -> None:
        """Clear the code index."""
        self._code_index = None
        self._code_documents = []

    def clear_paper_index(self) -> None:
        """Clear the paper index."""
        self._paper_index = None
        self._paper_documents = []


# Global BM25 index instance
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get the global BM25 index instance."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
