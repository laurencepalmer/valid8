"""Generate natural language search text for semantic code chunks."""

from typing import Optional

from backend.models.semantic_chunk import (
    SemanticCodeChunk,
    SymbolType,
    StructuralMetadata,
    BehaviorDescriptors,
    DependencyInfo,
)
from backend.services.semantic_chunking.name_normalizer import (
    normalize_symbol_name,
    tokens_to_readable,
)


class TextGenerator:
    """Generate natural language descriptions for code chunks."""

    def generate_search_text(
        self,
        metadata: StructuralMetadata,
        behavior: BehaviorDescriptors,
        dependencies: DependencyInfo,
        docstring: Optional[str] = None,
        normalized_tokens: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a natural language search text for a code chunk.

        This text is optimized for semantic search and embedding.

        Args:
            metadata: Structural metadata about the symbol
            behavior: Behavioral characteristics
            dependencies: Dependency information
            docstring: Optional docstring
            normalized_tokens: Optional pre-normalized name tokens

        Returns:
            A natural language description for embedding
        """
        parts = []

        # Symbol type and name
        symbol_desc = self._describe_symbol(metadata, normalized_tokens)
        parts.append(symbol_desc)

        # Signature phrase
        signature_phrase = self._describe_signature(metadata)
        if signature_phrase:
            parts.append(signature_phrase)

        # Behavior phrases
        behavior_phrases = self._describe_behavior(behavior)
        if behavior_phrases:
            parts.append(behavior_phrases)

        # Complexity
        parts.append(f"{behavior.complexity_tier} complexity.")

        # Framework patterns
        if dependencies.framework_patterns:
            patterns = ", ".join(dependencies.framework_patterns[:3])
            parts.append(f"uses {patterns}.")

        # Docstring snippet
        if docstring:
            # Truncate long docstrings
            doc_snippet = docstring[:200].strip()
            if len(docstring) > 200:
                doc_snippet += "..."
            parts.append(f"documentation: {doc_snippet}")

        return " ".join(parts)

    def _describe_symbol(
        self,
        metadata: StructuralMetadata,
        normalized_tokens: Optional[list[str]] = None,
    ) -> str:
        """Generate a description of the symbol type and name."""
        # Get readable name
        if normalized_tokens:
            readable_name = tokens_to_readable(normalized_tokens)
        else:
            tokens = normalize_symbol_name(metadata.symbol_name)
            readable_name = tokens_to_readable(tokens) if tokens else metadata.symbol_name

        # Symbol type
        type_name = metadata.symbol_type.value

        # Parent scope
        if metadata.parent_scope:
            return f"{type_name} {readable_name} in class {metadata.parent_scope} in {metadata.language} file {metadata.relative_path}."
        else:
            return f"{type_name} {readable_name} in {metadata.language} file {metadata.relative_path}."

    def _describe_signature(self, metadata: StructuralMetadata) -> Optional[str]:
        """Generate a description of the function signature."""
        if metadata.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            return None

        parts = []

        # Parameters
        param_count = len(metadata.parameters)
        if param_count == 0:
            parts.append("takes no parameters")
        elif param_count == 1:
            parts.append(f"takes 1 parameter ({metadata.parameters[0]})")
        else:
            param_names = ", ".join(metadata.parameters[:3])
            if param_count > 3:
                param_names += ", ..."
            parts.append(f"takes {param_count} parameters ({param_names})")

        # Return type
        if metadata.return_type:
            parts.append(f"returns {metadata.return_type}")

        if parts:
            return " and ".join(parts) + "."
        return None

    def _describe_behavior(self, behavior: BehaviorDescriptors) -> Optional[str]:
        """Generate a description of the behavioral characteristics."""
        behaviors = []

        if behavior.has_conditionals:
            behaviors.append("contains conditional logic")
        if behavior.has_loops:
            behaviors.append("contains loops")
        if behavior.has_error_handling:
            behaviors.append("handles errors")
        if behavior.has_async:
            behaviors.append("is asynchronous")
        if behavior.has_validation:
            behaviors.append("performs validation")
        if behavior.is_pure:
            behaviors.append("is a pure function")
        if behavior.has_mutations:
            behaviors.append("mutates state")

        if behaviors:
            return ", ".join(behaviors) + "."
        return None

    def generate_chunk_id(
        self,
        metadata: StructuralMetadata,
        sub_chunk_index: Optional[int] = None,
    ) -> str:
        """
        Generate a unique chunk ID.

        Format: {relative_path}:{symbol_name}:{start_line}[-{sub_chunk_index}]
        """
        base_id = f"{metadata.relative_path}:{metadata.qualified_name}:{metadata.start_line}"
        if sub_chunk_index is not None:
            return f"{base_id}-{sub_chunk_index}"
        return base_id


# Global generator instance
_text_generator: Optional[TextGenerator] = None


def get_text_generator() -> TextGenerator:
    """Get the global text generator instance."""
    global _text_generator
    if _text_generator is None:
        _text_generator = TextGenerator()
    return _text_generator
