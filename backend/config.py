from pydantic_settings import BaseSettings
from typing import Literal, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # AI Provider Configuration
    ai_provider: Literal["openai", "anthropic", "ollama"] = "ollama"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # Embedding Configuration
    use_local_embeddings: bool = True
    # paraphrase-MiniLM-L3-v2 is ~2x faster than all-MiniLM-L6-v2 with slightly lower quality
    local_embedding_model: str = "paraphrase-MiniLM-L3-v2"

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
