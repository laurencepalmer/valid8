"""Data models for semantic code chunking."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class SymbolType(str, Enum):
    """Type of code symbol."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    UNKNOWN = "unknown"


class BehaviorDescriptors(BaseModel):
    """Describes the behavioral characteristics of a code chunk."""

    has_conditionals: bool = False
    has_loops: bool = False
    has_validation: bool = False
    has_error_handling: bool = False
    has_async: bool = False
    return_behavior: Literal["void", "value", "conditional", "unknown"] = "unknown"
    has_mutations: bool = False
    is_pure: bool = False
    complexity_tier: Literal["simple", "moderate", "complex"] = "simple"


class DependencyInfo(BaseModel):
    """Dependency information for a code chunk."""

    imports: list[str] = []
    external_calls: list[str] = []
    framework_patterns: list[str] = []


class StructuralMetadata(BaseModel):
    """Structural metadata about a code symbol."""

    symbol_type: SymbolType
    symbol_name: str
    qualified_name: str
    file_path: str
    relative_path: str
    language: str
    start_line: int
    end_line: int
    parent_scope: Optional[str] = None
    signature: Optional[str] = None
    decorators: list[str] = []
    parameters: list[str] = []
    return_type: Optional[str] = None


class SemanticCodeChunk(BaseModel):
    """A semantically meaningful code chunk with rich metadata."""

    chunk_id: str
    content: str
    search_text: str  # NL representation for embedding
    metadata: StructuralMetadata
    behavior: BehaviorDescriptors
    dependencies: DependencyInfo
    docstring: Optional[str] = None
    inline_comments: list[str] = []
    normalized_name_tokens: list[str] = []
    # Sub-chunk support for large symbols
    is_sub_chunk: bool = False
    sub_chunk_index: Optional[int] = None
    parent_chunk_id: Optional[str] = None
