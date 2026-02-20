from pydantic_settings import BaseSettings
from typing import Literal, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # AI Provider Configuration
    ai_provider: Literal["openai", "anthropic", "ollama", "mlx"] = "ollama"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    # MLX (Apple Silicon only)
    mlx_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    # Embedding Configuration
    use_local_embeddings: bool = True
    local_embedding_model: str = "all-mpnet-base-v2"
    local_embedding_dim: int = 768

    # Caching Configuration
    embedding_cache_size: int = 1000  # LRU cache entries for embeddings
    search_cache_size: int = 500  # TTL cache entries for search results
    search_cache_ttl: int = 300  # 5 minutes TTL for search cache

    # Reranking Configuration
    use_reranking: bool = True
    rerank_top_k: int = 20  # Re-rank top K results before returning
    cross_encoder_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    # HNSW Configuration
    hnsw_m: int = 32  # Max connections per node
    hnsw_construction_ef: int = 128  # Construction-time search breadth
    hnsw_search_ef: int = 64  # Query-time search breadth

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    clone_dir: str = "./data/repos"

    # Application
    max_file_size_mb: int = 50
    supported_code_extensions: list[str] = [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
        ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
        ".scala", ".r", ".m", ".mm", ".cs", ".vue", ".svelte"
    ]

    # CORS Configuration
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
