"""Analyze code behavior patterns using AST traversal."""

from typing import Optional

import tree_sitter

from backend.models.semantic_chunk import BehaviorDescriptors
from backend.services.semantic_chunking.ast_parser import get_ast_parser
from backend.services.semantic_chunking.languages import get_language_config


class BehaviorAnalyzer:
    """Analyze behavioral characteristics of code symbols."""

    def __init__(self):
        self.parser = get_ast_parser()

    # Node types for pattern detection (defined as class attributes for performance)
    CONDITIONAL_TYPES = frozenset((
        "if_statement", "if_expression", "switch_statement", "switch_expression",
        "expression_switch_statement", "type_switch_statement", "match_expression",
        "conditional_expression", "ternary_expression",
    ))

    LOOP_TYPES = frozenset((
        "for_statement", "for_expression", "for_in_statement", "for_range_loop",
        "enhanced_for_statement", "while_statement", "while_expression",
        "do_statement", "loop_expression",
    ))

    ERROR_HANDLING_TYPES = frozenset(("try_statement", "try_expression", "catch_clause"))

    ASYNC_TYPES = frozenset(("async", "await_expression", "await"))

    VALIDATION_TYPES = frozenset(("assert_statement", "assertion"))

    MUTATION_TYPES = frozenset((
        "assignment_expression", "augmented_assignment_expression", "update_expression",
        "assignment", "augmented_assignment", "short_var_declaration", "assignment_statement",
    ))

    MUTATION_TARGET_TYPES = frozenset((
        "member_expression", "attribute", "field_expression",
        "subscript_expression", "index_expression",
    ))

    RETURN_TYPES = frozenset(("return_statement", "return_expression"))

    CONDITIONAL_PARENT_TYPES = frozenset((
        "if_statement", "if_expression", "match_expression", "switch_statement",
    ))

    def analyze(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        language: str,
    ) -> BehaviorDescriptors:
        """
        Analyze the behavior patterns in a code symbol.

        Uses single-pass tree traversal for performance.

        Args:
            node: The tree-sitter node to analyze
            source_bytes: The source code as bytes
            language: The programming language

        Returns:
            BehaviorDescriptors with detected patterns
        """
        # Initialize counters and flags
        pattern_counts = {
            "conditionals": 0,
            "loops": 0,
            "error_handling": 0,
            "async": 0,
            "validation": 0,
        }

        has_conditionals = False
        has_loops = False
        has_error_handling = False
        has_async = False
        has_validation = False
        has_mutations = False

        # Return behavior tracking
        return_statements = []
        has_conditional_return = False
        has_unconditional_return = False

        # Single-pass tree traversal
        stack = [node]
        while stack:
            descendant = stack.pop()
            node_type = descendant.type

            # Check for conditionals
            if node_type in self.CONDITIONAL_TYPES:
                has_conditionals = True
                pattern_counts["conditionals"] += 1

            # Check for loops
            elif node_type in self.LOOP_TYPES:
                has_loops = True
                pattern_counts["loops"] += 1

            # Check for error handling
            elif node_type in self.ERROR_HANDLING_TYPES:
                has_error_handling = True
                pattern_counts["error_handling"] += 1

            # Check for async
            elif node_type in self.ASYNC_TYPES:
                has_async = True
                pattern_counts["async"] += 1

            # Check for assertions/validation
            elif node_type in self.VALIDATION_TYPES:
                has_validation = True
                pattern_counts["validation"] += 1

            # Check for mutations
            elif node_type in self.MUTATION_TYPES:
                for child in descendant.children:
                    if child.type in self.MUTATION_TARGET_TYPES:
                        has_mutations = True
                        break

            # Check for return statements
            elif node_type in self.RETURN_TYPES:
                return_statements.append(descendant)
                # Check if inside a conditional
                parent = descendant.parent
                is_conditional = False
                while parent and parent != node:
                    if parent.type in self.CONDITIONAL_PARENT_TYPES:
                        has_conditional_return = True
                        is_conditional = True
                        break
                    parent = parent.parent
                if not is_conditional:
                    has_unconditional_return = True

            # Add children to stack (reverse order to maintain traversal order)
            stack.extend(reversed(descendant.children))

        # Determine return behavior
        if not return_statements:
            return_behavior = "void"
        elif has_conditional_return and has_unconditional_return:
            return_behavior = "conditional"
        elif has_conditional_return:
            return_behavior = "conditional"
        else:
            return_behavior = "value"

        # Calculate complexity tier
        complexity_tier = self._calculate_complexity(pattern_counts, node)

        # Determine if function is pure (no mutations, no side effects)
        is_pure = (
            not has_mutations
            and not has_error_handling
            and not has_async
            and pattern_counts["conditionals"] <= 1
        )

        return BehaviorDescriptors(
            has_conditionals=has_conditionals,
            has_loops=has_loops,
            has_validation=has_validation,
            has_error_handling=has_error_handling,
            has_async=has_async,
            return_behavior=return_behavior,
            has_mutations=has_mutations,
            is_pure=is_pure,
            complexity_tier=complexity_tier,
        )

    def _calculate_complexity(
        self, pattern_counts: dict[str, int], node: tree_sitter.Node
    ) -> str:
        """Calculate complexity tier based on pattern counts and size."""
        # Count lines
        line_count = node.end_point.row - node.start_point.row + 1

        # Calculate a complexity score
        score = 0
        score += pattern_counts["conditionals"] * 2
        score += pattern_counts["loops"] * 3
        score += pattern_counts["error_handling"] * 2
        score += pattern_counts["async"] * 1

        # Factor in size
        if line_count > 100:
            score += 5
        elif line_count > 50:
            score += 3
        elif line_count > 20:
            score += 1

        if score <= 3:
            return "simple"
        elif score <= 8:
            return "moderate"
        else:
            return "complex"


# Global analyzer instance
_behavior_analyzer: Optional[BehaviorAnalyzer] = None


def get_behavior_analyzer() -> BehaviorAnalyzer:
    """Get the global behavior analyzer instance."""
    global _behavior_analyzer
    if _behavior_analyzer is None:
        _behavior_analyzer = BehaviorAnalyzer()
    return _behavior_analyzer
