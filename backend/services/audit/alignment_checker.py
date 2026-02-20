"""Check alignment between claims and code behaviors."""

import asyncio
import json
import re
from typing import Optional

from backend.models.audit import (
    AlignmentResult,
    AlignmentStatus,
    Claim,
    ClaimType,
    CodeBehavior,
    IssueTier,
)
from backend.services.ai import get_ai_provider
from backend.services.embeddings import get_embedding_service
from backend.services.logging import logger


# Map claim keywords to extracted value keys from pattern detection
VALUE_KEY_MAPPING = {
    "fold": "cv_folds",
    "cv": "cv_folds",
    "cross-validation": "cv_folds",
    "learning rate": "learning_rate",
    "lr": "learning_rate",
    "batch size": "batch_size",
    "batch": "batch_size",
    "epoch": "epochs",
    "dropout": "dropout",
    "weight decay": "weight_decay",
    "l2": "weight_decay",
    "hidden": "hidden_size",
    "seed": "random_seed",
    "test size": "test_split",
    "split": "test_split",
}

# Tier assignment based on claim type and mismatch severity
TIER_RULES = {
    ClaimType.DATA_SPLIT: {
        "default": IssueTier.RESULTS_INVALID,  # Wrong split = data leakage risk
        "minor_diff": IssueTier.RESULTS_QUESTIONABLE,  # e.g., 5-fold vs 10-fold
    },
    ClaimType.EVALUATION_METRICS: {
        "default": IssueTier.RESULTS_INVALID,  # Wrong metric = invalid results
        "minor_diff": IssueTier.RESULTS_QUESTIONABLE,
    },
    ClaimType.TRAINING_PROCEDURE: {
        "default": IssueTier.RESULTS_QUESTIONABLE,
        "minor_diff": IssueTier.REPRODUCIBILITY_RISK,
    },
    ClaimType.HYPERPARAMETER: {
        "default": IssueTier.RESULTS_QUESTIONABLE,
        "minor_diff": IssueTier.REPRODUCIBILITY_RISK,
    },
    ClaimType.MODEL_ARCHITECTURE: {
        "default": IssueTier.RESULTS_QUESTIONABLE,
        "minor_diff": IssueTier.REPRODUCIBILITY_RISK,
    },
    ClaimType.DATA_PREPROCESSING: {
        "default": IssueTier.RESULTS_QUESTIONABLE,
        "minor_diff": IssueTier.REPRODUCIBILITY_RISK,
    },
}


ALIGNMENT_CHECK_PROMPT = """You are auditing code against a research paper/description claim.

CLAIM from description:
"{claim_text}"

Expected value (if specified): {expected_value}
Claim type: {claim_type}

RELEVANT CODE SECTIONS:
{code_sections}

Does the code align with the claim? Consider:
1. Does the code implement what the claim describes?
2. If the claim specifies a value (e.g., "5-fold CV"), does the code use that exact value?
3. Are there any contradictions between claim and code?
4. Is anything missing that the claim implies should exist?

Respond with JSON:
{{
    "status": "aligned" | "misaligned" | "partial" | "unverified",
    "explanation": "Clear explanation of your determination",
    "tier": "tier1" | "tier2" | "tier3" | "tier4",
    "specific_issues": ["List of specific discrepancies if any"],
    "actual_value_found": "The actual value in code if different from claimed"
}}

Tier guidelines:
- tier1: Results invalid (wrong metric, data leakage, fundamentally different algorithm)
- tier2: Results questionable (different hyperparameters that could affect results)
- tier3: Reproducibility risk (minor differences that probably don't affect results much)
- tier4: Minor discrepancy (cosmetic or clearly intentional differences)

Return ONLY valid JSON, no other text."""


class AlignmentChecker:
    """Check alignment between claims and code behaviors."""

    def __init__(self):
        self.ai_provider = None
        self.embedding_service = None

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract a numeric value from text."""
        # Try to find numeric patterns
        patterns = [
            r"(\d+\.?\d*)",  # Basic number
            r"(\d+e[+-]?\d+)",  # Scientific notation
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def _values_match(
        self,
        expected: str,
        actual: str,
        tolerance: float = 0.01,
    ) -> tuple[bool, str]:
        """
        Compare expected and actual values.

        Returns:
            Tuple of (match_status, explanation)
        """
        if expected is None or actual is None:
            return True, "No specific value to compare"

        expected_str = str(expected).lower().strip()
        actual_str = str(actual).lower().strip()

        # Exact string match
        if expected_str == actual_str:
            return True, f"Exact match: {expected}"

        # Try numeric comparison with tolerance
        expected_num = self._extract_numeric_value(expected)
        actual_num = self._extract_numeric_value(actual)

        if expected_num is not None and actual_num is not None:
            if expected_num == actual_num:
                return True, f"Numeric match: {expected_num}"

            # Check relative tolerance
            if expected_num != 0:
                rel_diff = abs(expected_num - actual_num) / abs(expected_num)
                if rel_diff <= tolerance:
                    return True, f"Values match within tolerance: {expected} ≈ {actual}"

            return False, f"Value mismatch: expected {expected}, found {actual}"

        # Different strings
        return False, f"Mismatch: expected '{expected}', found '{actual}'"

    def _quick_value_check(
        self,
        claim: Claim,
        behaviors: list[CodeBehavior],
    ) -> Optional[AlignmentResult]:
        """
        Quick alignment check using pattern-extracted values.

        Returns an AlignmentResult if we can definitively determine alignment
        from extracted values, None if LLM check is needed.
        """
        if not claim.expected_value:
            return None

        # Find the relevant value key based on claim keywords
        value_key = None
        claim_lower = claim.text.lower()
        for keyword, key in VALUE_KEY_MAPPING.items():
            if keyword in claim_lower:
                value_key = key
                break

        if not value_key:
            return None

        # Check if any behavior has this extracted value
        for behavior in behaviors:
            if not behavior.extracted_values:
                continue

            if value_key in behavior.extracted_values:
                actual_value = behavior.extracted_values[value_key]
                match, explanation = self._values_match(
                    claim.expected_value, actual_value
                )

                if match:
                    return AlignmentResult(
                        claim=claim,
                        status=AlignmentStatus.ALIGNED,
                        matched_code=[behavior],
                        explanation=f"Pattern match: {explanation}",
                        tier=IssueTier.MINOR_DISCREPANCY,
                        confidence=0.95,
                    )
                else:
                    # Determine tier based on claim type
                    tier_rules = TIER_RULES.get(claim.claim_type, {})
                    tier = tier_rules.get("default", IssueTier.RESULTS_QUESTIONABLE)

                    return AlignmentResult(
                        claim=claim,
                        status=AlignmentStatus.MISALIGNED,
                        matched_code=[behavior],
                        explanation=f"Pattern mismatch: {explanation}",
                        tier=tier,
                        specific_issues=[explanation],
                        confidence=0.95,
                    )

        return None

    def _determine_tier(
        self,
        claim: Claim,
        is_minor: bool = False,
    ) -> IssueTier:
        """Determine the appropriate tier for a misalignment."""
        tier_rules = TIER_RULES.get(claim.claim_type, {})
        if is_minor:
            return tier_rules.get("minor_diff", IssueTier.REPRODUCIBILITY_RISK)
        return tier_rules.get("default", IssueTier.RESULTS_QUESTIONABLE)

    async def check_all(
        self,
        claims: list[Claim],
        behaviors: list[CodeBehavior],
        use_quick_check: bool = True,
        max_concurrent: int = 5,
    ) -> list[AlignmentResult]:
        """
        Check alignment for all claims.

        Args:
            claims: List of claims from description
            behaviors: List of code behaviors
            use_quick_check: Try pattern-based value matching before LLM
            max_concurrent: Maximum concurrent LLM calls

        Returns:
            List of AlignmentResult objects
        """
        if self.ai_provider is None:
            self.ai_provider = get_ai_provider()

        # Phase 1: Find relevant behaviors for all claims in parallel
        relevance_tasks = [
            self._find_relevant_behaviors(claim, behaviors)
            for claim in claims
        ]
        all_relevant_behaviors = await asyncio.gather(*relevance_tasks)

        # Phase 2: Separate claims into quick-check vs LLM-needed
        results = [None] * len(claims)  # Preserve order
        llm_needed = []  # (index, claim, relevant_behaviors)
        quick_check_hits = 0

        for i, (claim, relevant_behaviors) in enumerate(zip(claims, all_relevant_behaviors)):
            if not relevant_behaviors:
                results[i] = AlignmentResult(
                    claim=claim,
                    status=AlignmentStatus.UNVERIFIED,
                    explanation="Could not find relevant code for this claim",
                    tier=IssueTier.REPRODUCIBILITY_RISK,
                )
                continue

            # Try quick value-based check first (avoids LLM call)
            if use_quick_check and claim.expected_value:
                quick_result = self._quick_value_check(claim, relevant_behaviors)
                if quick_result:
                    logger.debug(f"Quick check resolved claim: {claim.text[:50]}...")
                    quick_check_hits += 1
                    results[i] = quick_result
                    continue

            # Needs LLM analysis
            llm_needed.append((i, claim, relevant_behaviors))

        # Phase 3: Run LLM checks in parallel batches
        if llm_needed:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def check_with_limit(idx: int, claim: Claim, behaviors: list[CodeBehavior]):
                async with semaphore:
                    return idx, await self._check_alignment(claim, behaviors)

            llm_tasks = [
                check_with_limit(idx, claim, rel_behaviors)
                for idx, claim, rel_behaviors in llm_needed
            ]
            llm_results = await asyncio.gather(*llm_tasks)

            for idx, result in llm_results:
                results[idx] = result

        logger.info(f"Alignment check: {quick_check_hits}/{len(claims)} resolved via quick check, {len(llm_needed)} via LLM (parallel)")
        return results

    async def _find_relevant_behaviors(
        self,
        claim: Claim,
        behaviors: list[CodeBehavior],
    ) -> list[CodeBehavior]:
        """Find code behaviors relevant to a claim."""
        scored_behaviors = []

        # Get relevant value key for this claim
        value_key = None
        claim_lower = claim.text.lower()
        for keyword, key in VALUE_KEY_MAPPING.items():
            if keyword in claim_lower:
                value_key = key
                break

        for behavior in behaviors:
            score = 0

            # Score 1: Type match
            if behavior.behavior_type == claim.claim_type:
                score += 10

            # Score 2: Has relevant extracted value
            if value_key and behavior.extracted_values.get(value_key):
                score += 20  # High priority - direct value match

            # Score 3: Keyword matching
            keywords = claim.keywords + claim.text.lower().split()
            for kw in keywords:
                kw_lower = kw.lower()
                if len(kw_lower) < 3:  # Skip very short keywords
                    continue
                if kw_lower in behavior.description.lower():
                    score += 2
                if kw_lower in behavior.code_snippet.lower():
                    score += 3

            # Score 4: Expected value appears in code
            if claim.expected_value:
                if claim.expected_value in behavior.code_snippet:
                    score += 5
                if claim.expected_value in behavior.description:
                    score += 3

            if score > 0:
                scored_behaviors.append((score, behavior))

        # Sort by score descending
        scored_behaviors.sort(key=lambda x: x[0], reverse=True)

        # Get top matches
        relevant = [b for _, b in scored_behaviors[:5]]

        # If no good matches, try semantic search
        if not relevant:
            if self.embedding_service is None:
                self.embedding_service = get_embedding_service()

            try:
                search_results = await self.embedding_service.search_similar(
                    query=claim.text,
                    n_results=3,
                )
                # Map search results back to behaviors
                for result in search_results:
                    for behavior in behaviors:
                        if behavior.chunk_id == result.get("id"):
                            relevant.append(behavior)
                            break
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        return relevant[:5]

    async def _check_alignment(
        self,
        claim: Claim,
        behaviors: list[CodeBehavior],
    ) -> AlignmentResult:
        """Check alignment between a claim and relevant code."""
        # Build code sections string
        code_sections = []
        for b in behaviors:
            code_sections.append(
                f"File: {b.relative_path} (lines {b.start_line}-{b.end_line})\n"
                f"Description: {b.description}\n"
                f"```\n{b.code_snippet}\n```"
            )

        code_sections_str = "\n\n---\n\n".join(code_sections)

        prompt = ALIGNMENT_CHECK_PROMPT.format(
            claim_text=claim.text,
            expected_value=claim.expected_value or "Not specified",
            claim_type=claim.claim_type.value,
            code_sections=code_sections_str,
        )

        try:
            response = await self.ai_provider.complete(
                prompt=prompt, temperature=0.0,
            )

            return self._parse_response(response, claim, behaviors)

        except Exception as e:
            return AlignmentResult(
                claim=claim,
                status=AlignmentStatus.UNVERIFIED,
                matched_code=behaviors,
                explanation=f"Error checking alignment: {str(e)}",
                tier=IssueTier.MINOR_DISCREPANCY,
            )

    def _parse_response(
        self,
        response: str,
        claim: Claim,
        behaviors: list[CodeBehavior],
    ) -> AlignmentResult:
        """Parse LLM response into AlignmentResult."""
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            # Map status string to enum
            status_str = data.get("status", "unverified")
            status_map = {
                "aligned": AlignmentStatus.ALIGNED,
                "misaligned": AlignmentStatus.MISALIGNED,
                "partial": AlignmentStatus.MISALIGNED,  # Treat partial as misaligned
                "unverified": AlignmentStatus.UNVERIFIED,
                "missing": AlignmentStatus.MISSING,
            }
            status = status_map.get(status_str, AlignmentStatus.UNVERIFIED)

            # Map tier string to enum
            tier_str = data.get("tier", "tier4")
            try:
                tier = IssueTier(tier_str)
            except ValueError:
                tier = IssueTier.MINOR_DISCREPANCY

            return AlignmentResult(
                claim=claim,
                status=status,
                matched_code=behaviors,
                explanation=data.get("explanation", ""),
                tier=tier,
                specific_issues=data.get("specific_issues", []),
            )

        except json.JSONDecodeError:
            return AlignmentResult(
                claim=claim,
                status=AlignmentStatus.UNVERIFIED,
                matched_code=behaviors,
                explanation="Could not parse alignment check response",
                tier=IssueTier.MINOR_DISCREPANCY,
            )
