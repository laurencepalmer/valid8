"""Language configuration registry for semantic chunking."""

from typing import Optional

from backend.services.semantic_chunking.languages.base import LanguageConfig
from backend.services.semantic_chunking.languages.python import PythonConfig
from backend.services.semantic_chunking.languages.javascript import JavaScriptConfig
from backend.services.semantic_chunking.languages.typescript import TypeScriptConfig
from backend.services.semantic_chunking.languages.java import JavaConfig
from backend.services.semantic_chunking.languages.go import GoConfig
from backend.services.semantic_chunking.languages.rust import RustConfig
from backend.services.semantic_chunking.languages.c import CConfig
from backend.services.semantic_chunking.languages.cpp import CppConfig
from backend.services.semantic_chunking.languages.fallback import FallbackConfig


# Registry mapping language names to their configurations
_LANGUAGE_REGISTRY: dict[str, type[LanguageConfig]] = {
    "python": PythonConfig,
    "javascript": JavaScriptConfig,
    "typescript": TypeScriptConfig,
    "java": JavaConfig,
    "go": GoConfig,
    "rust": RustConfig,
    "c": CConfig,
    "cpp": CppConfig,
}

# File extensions to language mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}


def get_language_config(language: str) -> LanguageConfig:
    """Get the language configuration for a given language."""
    config_class = _LANGUAGE_REGISTRY.get(language.lower())
    if config_class:
        return config_class()
    return FallbackConfig()


def get_language_for_extension(extension: str) -> str:
    """Get the language name for a file extension."""
    return EXTENSION_TO_LANGUAGE.get(extension.lower(), "unknown")


def is_supported_language(language: str) -> bool:
    """Check if a language has tree-sitter support."""
    return language.lower() in _LANGUAGE_REGISTRY


__all__ = [
    "LanguageConfig",
    "get_language_config",
    "get_language_for_extension",
    "is_supported_language",
]
