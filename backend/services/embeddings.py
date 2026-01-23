from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.models.codebase import Codebase, CodeChunk
from backend.models.paper import Paper
from backend.models.semantic_chunk import SemanticCodeChunk
from backend.services.code_loader import chunk_code, chunk_code_semantic
from backend.services.state import app_state


class EmbeddingService:
    def __init__(self):
        self._local_model = None
        self._chroma_client = None
        self._collection = None
        self._paper_collection = None
        self._device = None
        self.settings = get_settings()

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
        """Sync version - only works with local embeddings."""
        if self.settings.use_local_embeddings and self.local_model:
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        raise RuntimeError(
            "Sync get_embeddings() requires use_local_embeddings=True. "
            "Use get_embeddings_async() for API-based embeddings."
        )

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float]]:
        if self.settings.use_local_embeddings and self.local_model:
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

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
            metadata={"hnsw:space": "cosine"},
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

            app_state.set_index_progress(min(i + batch_size, total_chunks), total_chunks)

    async def _index_semantic(self, codebase: Codebase):
        """Index using semantic AST-based chunking."""
        chunks = chunk_code_semantic(codebase)

        if not chunks:
            return

        total_chunks = len(chunks)
        app_state.set_index_progress(0, total_chunks)

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

            app_state.set_index_progress(min(i + batch_size, total_chunks), total_chunks)

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "code_chunks",
        filters: Optional[dict] = None,
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

        Returns:
            List of matching results with document, metadata, and similarity score
        """
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
            metadata={"hnsw:space": "cosine"},
        )

        chunks = chunk_paper(paper)

        if not chunks:
            app_state.set_paper_indexed(True)
            return

        total_chunks = len(chunks)
        app_state.set_paper_index_progress(0, total_chunks)

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            documents = [
                f"Paper: {paper.name}\nPage {c.get('page', 'N/A')}:\n{c['content']}"
                for c in batch
            ]

            embeddings = await self.get_embeddings_async(documents)

            ids = [f"paper_chunk_{i + j}" for j in range(len(batch))]

            metadatas = [
                {
                    "paper_name": paper.name,
                    "page": c.get("page") or 0,  # ChromaDB doesn't accept None
                    "start_idx": c["start_idx"],
                    "end_idx": c["end_idx"],
                }
                for c in batch
            ]

            self._paper_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            app_state.set_paper_index_progress(
                min(i + batch_size, total_chunks), total_chunks
            )

        app_state.set_paper_indexed(True)

    async def search_paper_similar(
        self, query: str, n_results: int = 5, collection_name: str = "paper_chunks"
    ) -> list[dict]:
        """Search for similar paper sections given a code query."""
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

        return matches


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
