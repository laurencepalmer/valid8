"""Main semantic chunker orchestrating the chunking pipeline."""

import re
from typing import Optional

from backend.models.codebase import CodeFile
from backend.models.semantic_chunk import (
    SemanticCodeChunk,
    SymbolType,
    StructuralMetadata,
    BehaviorDescriptors,
    DependencyInfo,
)
from backend.services.semantic_chunking.ast_parser import get_ast_parser
from backend.services.semantic_chunking.symbol_extractor import (
    get_symbol_extractor,
    ExtractedSymbol,
)
from backend.services.semantic_chunking.behavior_analyzer import get_behavior_analyzer
from backend.services.semantic_chunking.text_generator import get_text_generator
from backend.services.semantic_chunking.name_normalizer import normalize_symbol_name
from backend.services.semantic_chunking.languages import (
    get_language_config,
    is_supported_language,
)


# Approximate token limit for sub-chunking (roughly 4 chars per token)
MAX_CHUNK_CHARS = 6000  # ~1500 tokens


class SemanticChunker:
    """
    Orchestrates semantic code chunking.

    Pipeline:
    1. Detect language
    2. Parse with tree-sitter (or fallback for unsupported languages)
    3. Extract symbols (functions, classes, methods)
    4. For each symbol:
       - Extract docstring
       - Analyze behavior
       - Extract dependencies
       - Generate search text
    5. Split large symbols into sub-chunks with context header
    """

    def __init__(self):
        self.parser = get_ast_parser()
        self.extractor = get_symbol_extractor()
        self.behavior_analyzer = get_behavior_analyzer()
        self.text_generator = get_text_generator()

    def chunk_file(self, file: CodeFile) -> list[SemanticCodeChunk]:
        """
        Chunk a single code file into semantic chunks.

        Args:
            file: The CodeFile to chunk

        Returns:
            List of SemanticCodeChunk objects
        """
        language = file.language

        # Use fallback for unsupported languages
        if not is_supported_language(language):
            return self._fallback_chunk(file)

        # Extract symbols using AST
        symbols = self.extractor.extract_symbols(
            source=file.content,
            language=language,
            file_path=file.path,
            relative_path=file.relative_path,
        )

        if not symbols:
            # No symbols found - use fallback chunking
            return self._fallback_chunk(file)

        chunks = []
        source_bytes = file.content.encode("utf-8")

        for symbol in symbols:
            # Analyze behavior
            behavior = self.behavior_analyzer.analyze(
                node=symbol.node,
                source_bytes=source_bytes,
                language=language,
            )

            # Extract dependencies
            dependencies = self._extract_dependencies(
                symbol=symbol,
                source_bytes=source_bytes,
                language=language,
            )

            # Extract inline comments
            inline_comments = self._extract_inline_comments(
                symbol=symbol,
                source_bytes=source_bytes,
            )

            # Normalize name tokens
            normalized_tokens = normalize_symbol_name(symbol.name)

            # Generate search text
            search_text = self.text_generator.generate_search_text(
                metadata=symbol.metadata,
                behavior=behavior,
                dependencies=dependencies,
                docstring=symbol.docstring,
                normalized_tokens=normalized_tokens,
            )

            # Check if we need to sub-chunk
            if len(symbol.content) > MAX_CHUNK_CHARS:
                sub_chunks = self._create_sub_chunks(
                    symbol=symbol,
                    behavior=behavior,
                    dependencies=dependencies,
                    inline_comments=inline_comments,
                    normalized_tokens=normalized_tokens,
                    search_text=search_text,
                )
                chunks.extend(sub_chunks)
            else:
                # Create single chunk
                chunk_id = self.text_generator.generate_chunk_id(symbol.metadata)
                chunk = SemanticCodeChunk(
                    chunk_id=chunk_id,
                    content=symbol.content,
                    search_text=search_text,
                    metadata=symbol.metadata,
                    behavior=behavior,
                    dependencies=dependencies,
                    docstring=symbol.docstring,
                    inline_comments=inline_comments,
                    normalized_name_tokens=normalized_tokens,
                )
                chunks.append(chunk)

        return chunks

    def _fallback_chunk(self, file: CodeFile) -> list[SemanticCodeChunk]:
        """
        Fallback line-based chunking for unsupported languages.

        Args:
            file: The CodeFile to chunk

        Returns:
            List of SemanticCodeChunk objects with minimal metadata
        """
        chunks = []
        lines = file.content.split("\n")
        chunk_size = 100
        overlap = 20

        start = 0
        chunk_index = 0

        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            chunk_content = "\n".join(lines[start:end])

            metadata = StructuralMetadata(
                symbol_type=SymbolType.UNKNOWN,
                symbol_name=f"chunk_{chunk_index}",
                qualified_name=f"{file.relative_path}:chunk_{chunk_index}",
                file_path=file.path,
                relative_path=file.relative_path,
                language=file.language,
                start_line=start + 1,
                end_line=end,
            )

            behavior = BehaviorDescriptors()
            dependencies = DependencyInfo()

            search_text = f"code chunk in {file.language} file {file.relative_path}, lines {start + 1} to {end}."

            chunk_id = f"{file.relative_path}:chunk_{chunk_index}:{start + 1}"
            chunk = SemanticCodeChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                search_text=search_text,
                metadata=metadata,
                behavior=behavior,
                dependencies=dependencies,
            )
            chunks.append(chunk)

            chunk_index += 1
            if end >= len(lines):
                break
            start += chunk_size - overlap

        return chunks

    def _create_sub_chunks(
        self,
        symbol: ExtractedSymbol,
        behavior: BehaviorDescriptors,
        dependencies: DependencyInfo,
        inline_comments: list[str],
        normalized_tokens: list[str],
        search_text: str,
    ) -> list[SemanticCodeChunk]:
        """
        Split a large symbol into sub-chunks with context headers.

        Args:
            symbol: The extracted symbol
            behavior: Behavior descriptors
            dependencies: Dependency info
            inline_comments: Inline comments
            normalized_tokens: Normalized name tokens
            search_text: Generated search text

        Returns:
            List of sub-chunks
        """
        chunks = []
        content = symbol.content
        lines = content.split("\n")

        # Create context header
        header_lines = []
        if symbol.metadata.signature:
            header_lines.append(f"# Context: {symbol.metadata.signature}")
        else:
            header_lines.append(f"# Context: {symbol.metadata.symbol_type.value} {symbol.name}")
        if symbol.docstring:
            doc_preview = symbol.docstring[:100].replace("\n", " ")
            header_lines.append(f"# {doc_preview}...")
        header = "\n".join(header_lines) + "\n"

        # Calculate chunk boundaries
        target_chunk_size = MAX_CHUNK_CHARS - len(header)
        current_chunk_lines = []
        current_size = 0
        sub_chunk_index = 0
        parent_chunk_id = self.text_generator.generate_chunk_id(symbol.metadata)

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > target_chunk_size and current_chunk_lines:
                # Create sub-chunk
                chunk_content = header + "\n".join(current_chunk_lines)
                sub_metadata = StructuralMetadata(
                    symbol_type=symbol.metadata.symbol_type,
                    symbol_name=symbol.metadata.symbol_name,
                    qualified_name=symbol.metadata.qualified_name,
                    file_path=symbol.metadata.file_path,
                    relative_path=symbol.metadata.relative_path,
                    language=symbol.metadata.language,
                    start_line=symbol.metadata.start_line + (i - len(current_chunk_lines)),
                    end_line=symbol.metadata.start_line + i - 1,
                    parent_scope=symbol.metadata.parent_scope,
                    signature=symbol.metadata.signature,
                    decorators=symbol.metadata.decorators,
                    parameters=symbol.metadata.parameters,
                    return_type=symbol.metadata.return_type,
                )

                chunk_id = self.text_generator.generate_chunk_id(
                    sub_metadata, sub_chunk_index
                )
                sub_search_text = f"{search_text} (part {sub_chunk_index + 1})"

                chunk = SemanticCodeChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    search_text=sub_search_text,
                    metadata=sub_metadata,
                    behavior=behavior,
                    dependencies=dependencies,
                    docstring=symbol.docstring if sub_chunk_index == 0 else None,
                    inline_comments=inline_comments if sub_chunk_index == 0 else [],
                    normalized_name_tokens=normalized_tokens,
                    is_sub_chunk=True,
                    sub_chunk_index=sub_chunk_index,
                    parent_chunk_id=parent_chunk_id,
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_lines = []
                current_size = 0
                sub_chunk_index += 1

            current_chunk_lines.append(line)
            current_size += line_size

        # Create final sub-chunk
        if current_chunk_lines:
            chunk_content = header + "\n".join(current_chunk_lines)
            sub_metadata = StructuralMetadata(
                symbol_type=symbol.metadata.symbol_type,
                symbol_name=symbol.metadata.symbol_name,
                qualified_name=symbol.metadata.qualified_name,
                file_path=symbol.metadata.file_path,
                relative_path=symbol.metadata.relative_path,
                language=symbol.metadata.language,
                start_line=symbol.metadata.start_line + (len(lines) - len(current_chunk_lines)),
                end_line=symbol.metadata.end_line,
                parent_scope=symbol.metadata.parent_scope,
                signature=symbol.metadata.signature,
                decorators=symbol.metadata.decorators,
                parameters=symbol.metadata.parameters,
                return_type=symbol.metadata.return_type,
            )

            chunk_id = self.text_generator.generate_chunk_id(sub_metadata, sub_chunk_index)
            sub_search_text = f"{search_text} (part {sub_chunk_index + 1})"

            chunk = SemanticCodeChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                search_text=sub_search_text,
                metadata=sub_metadata,
                behavior=behavior,
                dependencies=dependencies,
                docstring=symbol.docstring if sub_chunk_index == 0 else None,
                inline_comments=inline_comments if sub_chunk_index == 0 else [],
                normalized_name_tokens=normalized_tokens,
                is_sub_chunk=True,
                sub_chunk_index=sub_chunk_index,
                parent_chunk_id=parent_chunk_id,
            )
            chunks.append(chunk)

        return chunks

    def _extract_dependencies(
        self,
        symbol: ExtractedSymbol,
        source_bytes: bytes,
        language: str,
    ) -> DependencyInfo:
        """Extract dependency information from a symbol."""
        config = get_language_config(language)
        content = symbol.content

        # Detect framework patterns
        framework_patterns = []
        for framework, patterns in config.get_framework_patterns().items():
            for pattern in patterns:
                if pattern in content:
                    if framework not in framework_patterns:
                        framework_patterns.append(framework)
                    break

        # Extract external calls (simple heuristic: look for dotted names)
        external_calls = []
        call_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\s*\("
        for match in re.finditer(call_pattern, content):
            call = match.group(1)
            if call not in external_calls:
                external_calls.append(call)

        return DependencyInfo(
            imports=[],  # Imports are at file level, not symbol level
            external_calls=external_calls[:10],  # Limit to first 10
            framework_patterns=framework_patterns,
        )

    def _extract_inline_comments(
        self,
        symbol: ExtractedSymbol,
        source_bytes: bytes,
    ) -> list[str]:
        """Extract inline comments from a symbol."""
        comments = []
        content = symbol.content

        # Simple pattern matching for common comment styles
        # Single line comments
        for match in re.finditer(r"(?://|#)\s*(.+)$", content, re.MULTILINE):
            comment = match.group(1).strip()
            if comment and len(comment) > 3:  # Skip very short comments
                comments.append(comment)

        # Limit to first 5 comments
        return comments[:5]


# Global chunker instance
_semantic_chunker: Optional[SemanticChunker] = None


def get_semantic_chunker() -> SemanticChunker:
    """Get the global semantic chunker instance."""
    global _semantic_chunker
    if _semantic_chunker is None:
        _semantic_chunker = SemanticChunker()
    return _semantic_chunker
