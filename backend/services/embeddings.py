from typing import Optional
import hashlib
import time
from collections import OrderedDict
from threading import Lock
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.models.codebase import Codebase, CodeChunk
from backend.models.paper import Paper
from backend.models.semantic_chunk import SemanticCodeChunk
from backend.services.code_loader import chunk_code, chunk_code_semantic
from backend.services.state import app_state
from backend.services.search.bm25_search import get_bm25_index


class LRUCache:
    """Thread-safe LRU cache for embeddings."""

    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[str, list[float]] = OrderedDict()
        self.maxsize = maxsize
        self.lock = Lock()

    def _make_key(self, text: str) -> str:
        """Create a hash key for a text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding for text."""
        key = self._make_key(text)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                self.cache[key] = embedding
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)


class TTLCache:
    """Thread-safe TTL cache for search results."""

    def __init__(self, maxsize: int = 500, ttl: int = 300):
        self.cache: dict[str, tuple[float, list[dict]]] = {}
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = Lock()

    def _make_key(self, query: str, n_results: int, collection: str, filters: Optional[dict]) -> str:
        """Create a hash key for search parameters."""
        filter_str = str(sorted(filters.items())) if filters else ""
        key_str = f"{query}:{n_results}:{collection}:{filter_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, n_results: int, collection: str, filters: Optional[dict]) -> Optional[list[dict]]:
        """Get cached search results if not expired."""
        key = self._make_key(query, n_results, collection, filters)
        with self.lock:
            if key in self.cache:
                timestamp, results = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return results
                else:
                    del self.cache[key]
        return None

    def put(self, query: str, n_results: int, collection: str, filters: Optional[dict], results: list[dict]) -> None:
        """Cache search results."""
        key = self._make_key(query, n_results, collection, filters)
        with self.lock:
            # Evict expired entries if at capacity
            if len(self.cache) >= self.maxsize:
                now = time.time()
                expired = [k for k, (ts, _) in self.cache.items() if now - ts >= self.ttl]
                for k in expired:
                    del self.cache[k]
                # If still at capacity, remove oldest
                if len(self.cache) >= self.maxsize:
                    oldest = min(self.cache.items(), key=lambda x: x[1][0])
                    del self.cache[oldest[0]]
            self.cache[key] = (time.time(), results)


class EmbeddingService:
    def __init__(self):
        self._local_model = None
        self._chroma_client = None
        self._collection = None
        self._paper_collection = None
        self._device = None
        self.settings = get_settings()
        # Initialize caches
        self._embedding_cache = LRUCache(maxsize=self.settings.embedding_cache_size)
        self._search_cache = TTLCache(
            maxsize=self.settings.search_cache_size,
            ttl=self.settings.search_cache_ttl
        )
        self._paper_search_cache = TTLCache(
            maxsize=self.settings.search_cache_size,
            ttl=self.settings.search_cache_ttl
        )

    @property
    def device(self):
        """Detect and cache the best available device (GPU/MPS/CPU)."""
        if self._device is None:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"  # Apple Silicon GPU
            else:
                self._device = "cpu"
        return self._device

    @property
    def local_model(self):
        if self._local_model is None and self.settings.use_local_embeddings:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(
                self.settings.local_embedding_model,
                device=self.device
            )
        return self._local_model

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client(
                ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=self.settings.chroma_persist_dir,
                )
            )
        return self._chroma_client

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Sync version - only works with local embeddings. Uses LRU cache."""
        if self.settings.use_local_embeddings and self.local_model:
            results: list[list[float]] = []
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []

            # Check cache for each text
            for i, text in enumerate(texts):
                cached = self._embedding_cache.get(text)
                if cached is not None:
                    results.append(cached)
                else:
                    results.append([])  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Compute embeddings for uncached texts
            if uncached_texts:
                embeddings = self.local_model.encode(uncached_texts, convert_to_numpy=True)
                for idx, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                    emb_list = embedding.tolist()
                    self._embedding_cache.put(text, emb_list)
                    results[uncached_indices[idx]] = emb_list

            return results

        raise RuntimeError(
            "Sync get_embeddings() requires use_local_embeddings=True. "
            "Use get_embeddings_async() for API-based embeddings."
        )

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float]]:
        """Async version with LRU cache support."""
        if self.settings.use_local_embeddings and self.local_model:
            results: list[list[float]] = []
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []

            # Check cache for each text
            for i, text in enumerate(texts):
                cached = self._embedding_cache.get(text)
                if cached is not None:
                    results.append(cached)
                else:
                    results.append([])  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Compute embeddings for uncached texts
            if uncached_texts:
                embeddings = self.local_model.encode(uncached_texts, convert_to_numpy=True)
                for idx, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                    emb_list = embedding.tolist()
                    self._embedding_cache.put(text, emb_list)
                    results[uncached_indices[idx]] = emb_list

            return results

        from backend.services.ai import get_ai_provider

        provider = get_ai_provider()
        return await provider.get_embeddings(texts)

    async def index_codebase(
        self,
        codebase: Codebase,
        collection_name: str = "code_chunks",
        use_semantic: bool = True,
    ):
        """
        Index a codebase for semantic search.

        Args:
            codebase: The Codebase to index
            collection_name: Name of the ChromaDB collection
            use_semantic: If True, use semantic AST-based chunking (default).
                         If False, use legacy line-based chunking.
        """
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self._collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self.settings.hnsw_m,
                "hnsw:construction_ef": self.settings.hnsw_construction_ef,
            },
        )

        if use_semantic:
            await self._index_semantic(codebase)
        else:
            await self._index_legacy(codebase)

    async def _index_legacy(self, codebase: Codebase):
        """Index using legacy line-based chunking."""
        chunks = chunk_code(codebase)

        if not chunks:
            return

        total_chunks = len(chunks)
        app_state.set_index_progress(0, total_chunks)

        # Collect all documents for BM25 indexing
        all_bm25_docs: list[dict] = []

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            documents = [
                f"File: {c.relative_path}\nLines {c.start_line}-{c.end_line}:\n{c.content}"
                for c in batch
            ]

            embeddings = await self.get_embeddings_async(documents)

            ids = [f"chunk_{i + j}" for j in range(len(batch))]

            metadatas = [
                {
                    "file_path": c.file_path,
                    "relative_path": c.relative_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": c.language,
                }
                for c in batch
            ]

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Collect for BM25
            for j, doc in enumerate(documents):
                all_bm25_docs.append({
                    'id': ids[j],
                    'document': doc,
                    'metadata': metadatas[j],
                })

            app_state.set_index_progress(min(i + batch_size, total_chunks), total_chunks)

        # Build BM25 index
        bm25_index = get_bm25_index()
        bm25_index.build_code_index(all_bm25_docs)

    async def _index_semantic(self, codebase: Codebase):
        """Index using semantic AST-based chunking."""
        chunks = chunk_code_semantic(codebase)

        if not chunks:
            return

        total_chunks = len(chunks)
        app_state.set_index_progress(0, total_chunks)

        # Collect all documents for BM25 indexing
        all_bm25_docs: list[dict] = []

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Use search_text for embedding (optimized for semantic search)
            search_texts = [c.search_text for c in batch]
            embeddings = await self.get_embeddings_async(search_texts)

            ids = [c.chunk_id for c in batch]

            # Store raw content as document for retrieval
            documents = [c.content for c in batch]

            # Rich metadata for filtering
            metadatas = [
                {
                    "file_path": c.metadata.file_path,
                    "relative_path": c.metadata.relative_path,
                    "start_line": c.metadata.start_line,
                    "end_line": c.metadata.end_line,
                    "language": c.metadata.language,
                    "symbol_type": c.metadata.symbol_type.value,
                    "symbol_name": c.metadata.symbol_name,
                    "qualified_name": c.metadata.qualified_name,
                    "parent_scope": c.metadata.parent_scope or "",
                    "has_conditionals": c.behavior.has_conditionals,
                    "has_loops": c.behavior.has_loops,
                    "has_error_handling": c.behavior.has_error_handling,
                    "has_async": c.behavior.has_async,
                    "complexity_tier": c.behavior.complexity_tier,
                    "is_sub_chunk": c.is_sub_chunk,
                    "docstring": (c.docstring or "")[:500],  # Truncate long docstrings
                }
                for c in batch
            ]

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Collect for BM25 (use search_text for better keyword matching)
            for j, (chunk, search_text) in enumerate(zip(batch, search_texts)):
                all_bm25_docs.append({
                    'id': ids[j],
                    'document': f"{search_text}\n{documents[j]}",  # Combine search text and content
                    'metadata': metadatas[j],
                })

            app_state.set_index_progress(min(i + batch_size, total_chunks), total_chunks)

        # Build BM25 index
        bm25_index = get_bm25_index()
        bm25_index.build_code_index(all_bm25_docs)

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "code_chunks",
        filters: Optional[dict] = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Search for similar code chunks.

        Args:
            query: The search query
            n_results: Number of results to return
            collection_name: Name of the collection to search
            filters: Optional ChromaDB where clause for filtering results.
                    Example filters:
                    - {"language": "python"} - only Python files
                    - {"symbol_type": "function"} - only functions
                    - {"has_error_handling": True} - only code with error handling
                    - {"$and": [{"language": "python"}, {"has_async": True}]}
            use_cache: Whether to use TTL cache for results (default True)

        Returns:
            List of matching results with document, metadata, and similarity score
        """
        # Check cache first
        if use_cache:
            cached = self._search_cache.get(query, n_results, collection_name, filters)
            if cached is not None:
                return cached

        if self._collection is None:
            try:
                self._collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                return []

        query_embedding = (await self.get_embeddings_async([query]))[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            query_kwargs["where"] = filters

        results = self._collection.query(**query_kwargs)

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                matches.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                    }
                )

        # Cache results
        if use_cache and matches:
            self._search_cache.put(query, n_results, collection_name, filters, matches)

        return matches

    async def search_similar_async(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "code_chunks",
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Async alias for search_similar with filter support."""
        return await self.search_similar(query, n_results, collection_name, filters)

    async def index_paper(self, paper: Paper, collection_name: str = "paper_chunks"):
        """Index paper content for code-to-paper search."""
        from backend.services.paper_parser import chunk_paper

        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self._paper_collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self.settings.hnsw_m,
                "hnsw:construction_ef": self.settings.hnsw_construction_ef,
            },
        )

        chunks = chunk_paper(paper)

        if not chunks:
            app_state.set_paper_indexed(True)
            return

        total_chunks = len(chunks)
        app_state.set_paper_index_progress(0, total_chunks)

        # Collect all documents for BM25 indexing
        all_bm25_docs: list[dict] = []

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Include section metadata in embedded text for better semantic matching
            documents = []
            for c in batch:
                section_name = c.get("section_name", "")
                section_type = c.get("section_type", "")
                section_prefix = ""
                if section_name:
                    section_prefix = f"Section: {section_name}"
                    if section_type:
                        section_prefix += f" ({section_type})"
                    section_prefix += "\n"
                doc = f"Paper: {paper.name}\nPage {c.get('page', 'N/A')}\n{section_prefix}{c['content']}"
                documents.append(doc)

            embeddings = await self.get_embeddings_async(documents)

            ids = [f"paper_chunk_{i + j}" for j in range(len(batch))]

            metadatas = [
                {
                    "paper_name": paper.name,
                    "page": c.get("page") or 0,  # ChromaDB doesn't accept None
                    "start_idx": c["start_idx"],
                    "end_idx": c["end_idx"],
                    "section_name": c.get("section_name") or "",
                    "section_type": c.get("section_type") or "",
                }
                for c in batch
            ]

            self._paper_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Collect for BM25
            for j, doc in enumerate(documents):
                all_bm25_docs.append({
                    'id': ids[j],
                    'document': doc,
                    'metadata': metadatas[j],
                })

            app_state.set_paper_index_progress(
                min(i + batch_size, total_chunks), total_chunks
            )

        # Build BM25 index for papers
        bm25_index = get_bm25_index()
        bm25_index.build_paper_index(all_bm25_docs)

        app_state.set_paper_indexed(True)

    async def search_paper_similar(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "paper_chunks",
        use_cache: bool = True,
    ) -> list[dict]:
        """Search for similar paper sections given a code query."""
        # Check cache first
        if use_cache:
            cached = self._paper_search_cache.get(query, n_results, collection_name, None)
            if cached is not None:
                return cached

        if self._paper_collection is None:
            try:
                self._paper_collection = self.chroma_client.get_collection(
                    collection_name
                )
            except Exception:
                return []

        query_embedding = (await self.get_embeddings_async([query]))[0]

        results = self._paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                matches.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                    }
                )

        # Cache results
        if use_cache and matches:
            self._paper_search_cache.put(query, n_results, collection_name, None, matches)

        return matches


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
