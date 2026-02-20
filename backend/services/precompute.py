import asyncio
import hashlib
import re
from contextlib import asynccontextmanager
from typing import Optional

from backend.models.analysis import HighlightAnalysisResponse


# Matches "Paper: ...\n", "Page ...\n", "Section: ...\n" header lines added during indexing
_CHROMADB_PREFIX_RE = re.compile(
    r'^(?:Paper:.*\n|Page\s.*\n|Section:.*\n)+', re.MULTILINE
)

_SECTION_NAME_RE = re.compile(r'^Section:\s*(.+?)(?:\s*\(.*\))?\s*$', re.MULTILINE)

# Sections that rarely map to code — skip during precompute
_SKIP_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "conclusion", "conclusions", "acknowledgments", "acknowledgements",
    "future work", "limitations",
}


def _strip_doc_prefix(doc: str) -> str:
    """Remove ChromaDB metadata prefixes to get raw paper content."""
    return _CHROMADB_PREFIX_RE.sub('', doc).strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single space for cache matching."""
    return re.sub(r'\s+', ' ', text.strip()).lower()


class PrecomputeCache:
    """Cache for pre-computed paper-to-code mappings."""

    def __init__(self):
        self.cache: dict[str, HighlightAnalysisResponse] = {}
        # Store (normalized_chunk_text, cache_key) for substring matching
        self._chunk_index: list[tuple[str, str]] = []
        self.is_running: bool = False
        self.progress: int = 0
        self.total: int = 0
        # Pause gate — starts unblocked; user queries block it temporarily
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # not paused
        self._active_queries: int = 0

    def _make_key(self, text: str) -> str:
        """Hash of normalized text (whitespace-collapsed, lowercase, first 500 chars)."""
        normalized = _normalize_whitespace(text)[:500]
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, text: str) -> Optional[HighlightAnalysisResponse]:
        """Lookup by exact hash, then fall back to substring matching."""
        # Exact match
        key = self._make_key(text)
        if key in self.cache:
            return self.cache[key]

        # Substring match — find a cached chunk that contains (or is contained by) the query
        normalized = _normalize_whitespace(text)
        if len(normalized) < 20:
            return None

        for chunk_text, chunk_key in self._chunk_index:
            if normalized in chunk_text or chunk_text in normalized:
                return self.cache.get(chunk_key)

        return None

    def put(self, text: str, result: HighlightAnalysisResponse):
        key = self._make_key(text)
        self.cache[key] = result
        normalized = _normalize_whitespace(text)
        self._chunk_index.append((normalized, key))

    def pause(self):
        """Pause precompute — called when a user query arrives."""
        self._active_queries += 1
        self._resume_event.clear()

    def resume(self):
        """Resume precompute — called when user query finishes."""
        self._active_queries = max(0, self._active_queries - 1)
        if self._active_queries == 0:
            self._resume_event.set()

    async def wait_if_paused(self):
        """Block until precompute is allowed to continue."""
        await self._resume_event.wait()

    @asynccontextmanager
    async def user_query(self):
        """Context manager: pauses precompute for the duration of a user query."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def clear(self):
        self.cache.clear()
        self._chunk_index.clear()
        self.progress = 0
        self.total = 0
        self.is_running = False


_precompute_cache: Optional[PrecomputeCache] = None


def get_precompute_cache() -> PrecomputeCache:
    global _precompute_cache
    if _precompute_cache is None:
        _precompute_cache = PrecomputeCache()
    return _precompute_cache


async def precompute_mappings():
    """Run after both paper and codebase are indexed.

    Iterates all paper chunks from ChromaDB and pre-computes the full
    highlight analysis pipeline for each, storing results in the cache
    so that user highlights return instantly.
    """
    from backend.services.embeddings import get_embedding_service
    from backend.services.logging import logger

    cache = get_precompute_cache()
    cache.clear()
    cache.is_running = True

    embedding_service = get_embedding_service()
    paper_collection = embedding_service._paper_collection
    if not paper_collection:
        cache.is_running = False
        return

    all_chunks = paper_collection.get(include=["documents"])
    documents = all_chunks.get("documents", [])
    cache.total = len(documents)
    logger.info(f"[precompute] starting for {len(documents)} paper chunks")

    import asyncio

    semaphore = asyncio.Semaphore(8)

    skipped = 0

    def _should_skip(doc: str) -> bool:
        """Skip chunks from sections unlikely to reference code."""
        m = _SECTION_NAME_RE.search(doc)
        if not m:
            return False
        section = re.sub(r'^\d+\.?\s*', '', m.group(1).strip().lower()).strip()
        return section in _SKIP_SECTIONS

    async def process_chunk(i: int, doc: str):
        nonlocal skipped
        if _should_skip(doc):
            skipped += 1
            cache.progress += 1
            return
        raw_content = _strip_doc_prefix(doc)
        if not raw_content:
            cache.progress += 1
            return

        await cache.wait_if_paused()
        async with semaphore:
            try:
                from backend.services.alignment import analyze_highlight

                result = await analyze_highlight(raw_content, n_results=5)
                cache.put(raw_content, result)
                cache.progress += 1
                logger.info(f"[precompute] chunk {i+1}/{cache.total} done — {len(result.code_references)} refs")
            except Exception as e:
                cache.progress += 1
                logger.error(f"[precompute] chunk {i+1}/{cache.total} failed: {e}", exc_info=True)

    await asyncio.gather(*(process_chunk(i, doc) for i, doc in enumerate(documents)))

    cache.is_running = False
    logger.info(f"[precompute] done — cached {len(cache.cache)}/{cache.total} chunks, skipped {skipped} non-code sections")
