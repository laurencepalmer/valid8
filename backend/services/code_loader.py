import os
import shutil
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from git import Repo

from backend.config import get_settings
from backend.models.codebase import Codebase, CodeFile, CodeChunk
from backend.models.semantic_chunk import SemanticCodeChunk
from backend.services.semantic_chunking import SemanticChunker


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".m": "objectivec",
    ".mm": "objectivec",
    ".cs": "csharp",
    ".vue": "vue",
    ".svelte": "svelte",
}


def get_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext, "text")


def should_include_file(file_path: str, settings=None) -> bool:
    if settings is None:
        settings = get_settings()

    path = Path(file_path)

    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        ".pytest_cache",
        ".mypy_cache",
    }
    if any(part in skip_dirs for part in path.parts):
        return False

    return path.suffix.lower() in settings.supported_code_extensions


def load_local_codebase(path: str) -> Codebase:
    settings = get_settings()
    path = os.path.abspath(os.path.expanduser(path))

    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    files = []
    total_lines = 0

    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if not should_include_file(file_path, settings):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                relative_path = os.path.relpath(file_path, path)
                line_count = content.count("\n") + 1
                total_lines += line_count

                files.append(
                    CodeFile(
                        path=file_path,
                        relative_path=relative_path,
                        content=content,
                        language=get_language(file_path),
                        line_count=line_count,
                    )
                )
            except (IOError, OSError):
                continue

    return Codebase(
        path=path,
        name=os.path.basename(path),
        source_type="local",
        files=files,
        total_lines=total_lines,
    )


def load_github_codebase(url: str) -> Codebase:
    settings = get_settings()

    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise ValueError("URL must be a GitHub repository URL")

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub URL format")

    repo_name = path_parts[1].replace(".git", "")
    clone_path = os.path.join(settings.clone_dir, repo_name)

    # Cache repos - only re-clone if not already present
    if os.path.exists(clone_path):
        try:
            # Verify it's a valid git repo and pull latest
            existing_repo = Repo(clone_path)
            existing_repo.remotes.origin.fetch(depth=1)
            existing_repo.head.reset(commit='origin/HEAD', working_tree=True)
        except Exception:
            # If anything fails, re-clone
            shutil.rmtree(clone_path)
            Repo.clone_from(url, clone_path, depth=1)
    else:
        Repo.clone_from(url, clone_path, depth=1)

    codebase = load_local_codebase(clone_path)
    codebase.source_type = "github"
    codebase.github_url = url

    return codebase


def chunk_code(
    codebase: Codebase, chunk_size: int = 100, overlap: int = 20
) -> list[CodeChunk]:
    """
    Chunk code using naive line-based approach.

    This is the legacy chunking method. For semantic chunking,
    use chunk_code_semantic() instead.
    """
    chunks = []

    for file in codebase.files:
        lines = file.content.split("\n")
        total_lines = len(lines)

        start = 0
        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            chunk_content = "\n".join(lines[start:end])

            chunks.append(
                CodeChunk(
                    file_path=file.path,
                    relative_path=file.relative_path,
                    start_line=start + 1,
                    end_line=end,
                    content=chunk_content,
                    language=file.language,
                )
            )

            if end >= total_lines:
                break
            start += chunk_size - overlap

    return chunks


def chunk_code_semantic(codebase: Codebase) -> list[SemanticCodeChunk]:
    """
    Chunk code using semantic AST-based approach.

    This method extracts functions, classes, and methods as atomic units
    with rich metadata for improved semantic search.

    Args:
        codebase: The Codebase to chunk

    Returns:
        List of SemanticCodeChunk objects with rich metadata
    """
    chunker = SemanticChunker()
    chunks = []

    for file in codebase.files:
        file_chunks = chunker.chunk_file(file)
        chunks.extend(file_chunks)

    return chunks


def get_code_context(
    codebase: Codebase,
    file_path: str,
    start_line: int,
    end_line: int,
    context_lines: int = 5,
) -> Optional[str]:
    for file in codebase.files:
        if file.relative_path == file_path or file.path == file_path:
            lines = file.content.split("\n")
            actual_start = max(0, start_line - 1 - context_lines)
            actual_end = min(len(lines), end_line + context_lines)
            return "\n".join(lines[actual_start:actual_end])
    return None
