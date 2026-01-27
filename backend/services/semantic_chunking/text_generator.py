"""Generate natural language search text for semantic code chunks."""

import re
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


# Algorithm patterns to detect in code names
ALGORITHM_PATTERNS = {
    # Sorting algorithms
    r"(quick|merge|heap|bubble|insertion|selection|radix|counting|bucket)[\s_]?sort": "sorting algorithm",
    r"sort(ed|ing|er)?": "sorting",
    # Search algorithms
    r"(binary|linear|depth[\s_]?first|breadth[\s_]?first|a[\s_]?star)[\s_]?search": "search algorithm",
    r"(bfs|dfs)": "graph traversal",
    # Graph algorithms
    r"(dijkstra|bellman[\s_]?ford|floyd[\s_]?warshall|kruskal|prim|tarjan|kosaraju)": "graph algorithm",
    r"(shortest[\s_]?path|minimum[\s_]?spanning[\s_]?tree|mst)": "graph algorithm",
    # Tree operations
    r"(traverse|preorder|inorder|postorder|level[\s_]?order)": "tree traversal",
    r"(balance|rotate|insert|delete)[\s_]?(tree|node|bst)?": "tree operation",
    # Dynamic programming
    r"(dynamic[\s_]?programming|dp|memoiz|tabulation)": "dynamic programming",
    r"(knapsack|longest[\s_]?common|edit[\s_]?distance|levenshtein)": "dynamic programming",
    # Machine learning
    r"(gradient[\s_]?descent|backprop|forward[\s_]?pass|backward[\s_]?pass)": "machine learning",
    r"(train|predict|fit|transform|inference|embed)": "machine learning",
    r"(loss|optimizer|activation|softmax|relu|sigmoid|tanh)": "neural network",
    r"(attention|transformer|encoder|decoder|lstm|gru|rnn|cnn|conv)": "deep learning",
    # Data structures
    r"(hash[\s_]?map|hash[\s_]?table|dictionary|trie|bloom[\s_]?filter)": "data structure",
    r"(linked[\s_]?list|doubly[\s_]?linked|circular)": "linked list",
    r"(stack|queue|deque|heap|priority[\s_]?queue)": "data structure",
    # String algorithms
    r"(kmp|rabin[\s_]?karp|boyer[\s_]?moore|suffix[\s_]?(tree|array))": "string algorithm",
    r"(tokeniz|pars|regex|pattern[\s_]?match)": "string processing",
    # Math/numerical
    r"(matrix|vector|tensor|dot[\s_]?product|cross[\s_]?product)": "linear algebra",
    r"(fft|fourier|convolution|correlation)": "signal processing",
    r"(prime|factorial|fibonacci|gcd|lcm|modular)": "mathematical",
    # Validation/processing
    r"(validat|sanitiz|normaliz|encod|decod|compress|decompress)": "data processing",
    r"(encrypt|decrypt|hash|sign|verify)": "cryptography",
    # API/Web
    r"(fetch|request|response|endpoint|route|middleware)": "web/API",
    r"(serialize|deserialize|marshal|unmarshal|json|xml)": "serialization",
}


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

        # Detect and add algorithm hints early (important for search matching)
        algorithm_hints = self._detect_algorithm_hints(
            metadata.symbol_name,
            docstring,
            metadata.parameters if hasattr(metadata, 'parameters') else None,
        )
        if algorithm_hints:
            hints_str = ", ".join(algorithm_hints[:3])  # Limit to 3 hints
            parts.append(f"implements {hints_str}.")

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

        # Docstring with smart keyword extraction
        if docstring:
            doc_snippet = self._extract_docstring_keywords(docstring, max_length=200)
            if doc_snippet:
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

    def _detect_algorithm_hints(
        self,
        symbol_name: str,
        docstring: Optional[str] = None,
        parameters: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Detect algorithm patterns from symbol name, docstring, and parameters.

        Returns a list of algorithm hints that can enhance search matching.
        """
        hints: set[str] = set()

        # Check symbol name
        name_lower = symbol_name.lower().replace("_", " ")
        for pattern, hint in ALGORITHM_PATTERNS.items():
            if re.search(pattern, name_lower, re.IGNORECASE):
                hints.add(hint)

        # Check docstring
        if docstring:
            doc_lower = docstring.lower()
            for pattern, hint in ALGORITHM_PATTERNS.items():
                if re.search(pattern, doc_lower, re.IGNORECASE):
                    hints.add(hint)

        # Check parameter names for hints
        if parameters:
            param_text = " ".join(parameters).lower()
            for pattern, hint in ALGORITHM_PATTERNS.items():
                if re.search(pattern, param_text, re.IGNORECASE):
                    hints.add(hint)

        return list(hints)

    def _extract_docstring_keywords(
        self,
        docstring: str,
        max_length: int = 200,
    ) -> str:
        """
        Extract meaningful keywords from docstring with smart truncation.

        Prioritizes the first sentence (usually the summary) and any
        parameter/return descriptions that fit within the limit.
        """
        if not docstring:
            return ""

        docstring = docstring.strip()

        # If short enough, return as-is
        if len(docstring) <= max_length:
            return docstring

        # Try to get the first sentence (summary line)
        first_sentence_match = re.match(r"^([^.!?\n]+[.!?])", docstring)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1).strip()
            if len(first_sentence) <= max_length:
                # Try to add more context if space permits
                remaining = max_length - len(first_sentence) - 4  # " ..."
                if remaining > 50:
                    # Look for Args/Returns sections
                    args_match = re.search(r"(?:Args|Parameters|Arguments):?\s*\n?(.*?)(?=\n\s*\n|Returns|Raises|$)", docstring, re.DOTALL | re.IGNORECASE)
                    returns_match = re.search(r"Returns?:?\s*\n?(.*?)(?=\n\s*\n|Raises|$)", docstring, re.DOTALL | re.IGNORECASE)

                    extras = []
                    if args_match:
                        args_text = args_match.group(1).strip()[:100]
                        if args_text:
                            extras.append(f"params: {args_text}")
                    if returns_match:
                        returns_text = returns_match.group(1).strip()[:50]
                        if returns_text:
                            extras.append(f"returns: {returns_text}")

                    if extras:
                        extra_text = "; ".join(extras)
                        if len(extra_text) <= remaining:
                            return f"{first_sentence} {extra_text}"

                return first_sentence

        # Fall back to simple truncation at word boundary
        truncated = docstring[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]
        return truncated + "..."

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
