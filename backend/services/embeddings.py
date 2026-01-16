from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.models.codebase import Codebase, CodeChunk
from backend.models.paper import Paper
from backend.services.code_loader import chunk_code
from backend.services.state import app_state


class EmbeddingService:
    def __init__(self):
        self._local_model = None
        self._chroma_client = None
        self._collection = None
        self._paper_collection = None
        self.settings = get_settings()

    @property
    def local_model(self):
        if self._local_model is None and self.settings.use_local_embeddings:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.settings.local_embedding_model)
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

    async def index_codebase(self, codebase: Codebase, collection_name: str = "code_chunks"):
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self._collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = chunk_code(codebase)

        if not chunks:
            return

        total_chunks = len(chunks)
        app_state.set_index_progress(0, total_chunks)

        batch_size = 500  # Larger batches = fewer API calls
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

            # Update progress
            app_state.set_index_progress(min(i + batch_size, total_chunks), total_chunks)

    async def search_similar(
        self, query: str, n_results: int = 5, collection_name: str = "code_chunks"
    ) -> list[dict]:
        if self._collection is None:
            try:
                self._collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                return []

        query_embedding = (await self.get_embeddings_async([query]))[0]

        results = self._collection.query(
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

    async def search_similar_async(
        self, query: str, n_results: int = 5, collection_name: str = "code_chunks"
    ) -> list[dict]:
        return await self.search_similar(query, n_results, collection_name)

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
                    "page": c.get("page"),
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
