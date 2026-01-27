"""Generate audit reports and summaries."""

from collections import defaultdict
from typing import Optional

from backend.models.audit import (
    AlignmentResult,
    AlignmentStatus,
    AuditSummary,
    CatastrophicCategory,
    CatastrophicWarning,
    Claim,
    ClaimType,
    IssueTier,
)


# Remediation templates by category
REMEDIATION_TEMPLATES = {
    CatastrophicCategory.DATA_LEAKAGE: {
        "preprocessing_before_split": (
            "Move preprocessing steps (normalization, scaling, feature selection) "
            "AFTER the train/test split. Fit transformers only on training data."
        ),
        "cv_leakage": (
            "Move all preprocessing inside the cross-validation loop. "
            "Each fold should have independent preprocessing."
        ),
        "default": (
            "Review data flow to ensure test data never influences training. "
            "Use sklearn Pipeline to ensure proper separation."
        ),
    },
    CatastrophicCategory.EVALUATION_ERROR: {
        "eval_on_train": (
            "Compute metrics on held-out test set, not training data."
        ),
        "test_reuse": (
            "Use a separate validation set for hyperparameter tuning. "
            "Test set should only be used once for final evaluation."
        ),
        "default": (
            "Review evaluation code to ensure correct data is used for metrics."
        ),
    },
    CatastrophicCategory.TRAINING_ISSUE: {
        "wrong_mode": (
            "Add model.eval() before validation/testing and model.train() before training."
        ),
        "gradient_bug": (
            "Add optimizer.zero_grad() at the start of each training iteration."
        ),
        "default": (
            "Review training loop structure for common issues."
        ),
    },
    CatastrophicCategory.REPRODUCIBILITY: {
        "missing_seed": (
            "Set random seeds at the start: random.seed(42), np.random.seed(42), "
            "torch.manual_seed(42), torch.cuda.manual_seed_all(42)"
        ),
        "default": (
            "Add random seed setting and pin dependency versions."
        ),
    },
}


class ReportGenerator:
    """Generate audit summaries and reports."""

    def get_remediation(
        self,
        warning: CatastrophicWarning,
    ) -> str:
        """Get remediation suggestion for a warning."""
        category_templates = REMEDIATION_TEMPLATES.get(warning.category, {})
        return category_templates.get(
            warning.pattern_type,
            category_templates.get("default", warning.recommendation)
        )

    def group_issues_by_priority(
        self,
        alignment_results: list[AlignmentResult],
        warnings: list[CatastrophicWarning],
    ) -> dict[str, list]:
        """Group all issues by priority tier for actionable review."""
        grouped = {
            "tier1": [],  # Results invalid - fix immediately
            "tier2": [],  # Results questionable - should fix
            "tier3": [],  # Reproducibility risk - good to fix
            "tier4": [],  # Minor - optional
        }

        # Add misaligned claims
        for result in alignment_results:
            if result.status == AlignmentStatus.MISALIGNED:
                grouped[result.tier.value].append({
                    "type": "misalignment",
                    "claim": result.claim.text,
                    "explanation": result.explanation,
                    "issues": result.specific_issues,
                    "files": [c.relative_path for c in result.matched_code],
                })

        # Add warnings
        for warning in warnings:
            if not warning.suppressed:
                grouped[warning.tier.value].append({
                    "type": "warning",
                    "category": warning.category.value,
                    "pattern": warning.pattern_type,
                    "description": warning.description,
                    "file": warning.relative_path,
                    "lines": f"{warning.line_start}-{warning.line_end}",
                    "remediation": self.get_remediation(warning),
                })

        return grouped

    def generate_executive_summary(
        self,
        summary: AuditSummary,
        warnings: list[CatastrophicWarning],
    ) -> str:
        """Generate a brief executive summary for quick review."""
        lines = []

        # Overall status
        if summary.overall_score >= 80 and not any(w.tier == IssueTier.RESULTS_INVALID for w in warnings):
            lines.append("✓ Code generally aligns with description.")
        elif summary.overall_score >= 50:
            lines.append("⚠ Some discrepancies found between code and description.")
        else:
            lines.append("✗ Significant issues found requiring attention.")

        # Critical issues
        tier1_count = sum(1 for w in warnings if w.tier == IssueTier.RESULTS_INVALID and not w.suppressed)
        if tier1_count > 0:
            lines.append(f"\n⚠ CRITICAL: {tier1_count} issue(s) may invalidate results.")

        # Leakage specifically
        leakage_count = sum(
            1 for w in warnings
            if w.category == CatastrophicCategory.DATA_LEAKAGE and not w.suppressed
        )
        if leakage_count > 0:
            lines.append(f"   - {leakage_count} potential data leakage issue(s)")

        # Claims summary
        if summary.total_claims > 0:
            lines.append(f"\nClaims: {summary.aligned_count}/{summary.total_claims} verified")
            if summary.misaligned_count > 0:
                lines.append(f"   - {summary.misaligned_count} misalignment(s)")
            if summary.unverified_count > 0:
                lines.append(f"   - {summary.unverified_count} could not be verified")

        return "\n".join(lines)

    def generate_summary(
        self,
        claims: list[Claim],
        alignment_results: list[AlignmentResult],
        warnings: list[CatastrophicWarning],
    ) -> AuditSummary:
        """
        Generate a summary of the audit results.

        Args:
            claims: All extracted claims
            alignment_results: Results of alignment checking
            warnings: Catastrophic warnings found

        Returns:
            AuditSummary object
        """
        # Count alignment statuses
        aligned_count = sum(1 for r in alignment_results if r.status == AlignmentStatus.ALIGNED)
        misaligned_count = sum(1 for r in alignment_results if r.status == AlignmentStatus.MISALIGNED)
        unverified_count = sum(1 for r in alignment_results if r.status == AlignmentStatus.UNVERIFIED)
        missing_count = sum(1 for r in alignment_results if r.status == AlignmentStatus.MISSING)

        # Count warnings by category
        warnings_by_category = defaultdict(int)
        for w in warnings:
            warnings_by_category[w.category.value] += 1

        # Count warnings by tier
        warnings_by_tier = defaultdict(int)
        for w in warnings:
            warnings_by_tier[w.tier.value] += 1

        # Count critical issues (tier 1)
        critical_issues = sum(1 for w in warnings if w.tier == IssueTier.RESULTS_INVALID)
        critical_issues += sum(1 for r in alignment_results
                               if r.status == AlignmentStatus.MISALIGNED
                               and r.tier == IssueTier.RESULTS_INVALID)

        # Calculate overall score
        overall_score = self._calculate_score(
            total_claims=len(claims),
            aligned_count=aligned_count,
            misaligned_count=misaligned_count,
            warnings=warnings,
            alignment_results=alignment_results,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score=overall_score,
            critical_issues=critical_issues,
            warnings=warnings,
            misaligned_count=misaligned_count,
        )

        return AuditSummary(
            total_claims=len(claims),
            aligned_count=aligned_count,
            misaligned_count=misaligned_count,
            unverified_count=unverified_count,
            missing_count=missing_count,
            catastrophic_warnings_by_category=dict(warnings_by_category),
            catastrophic_warnings_by_tier=dict(warnings_by_tier),
            overall_score=overall_score,
            recommendation=recommendation,
        )

    def _calculate_score(
        self,
        total_claims: int,
        aligned_count: int,
        misaligned_count: int,
        warnings: list[CatastrophicWarning],
        alignment_results: list[AlignmentResult],
    ) -> float:
        """
        Calculate an overall alignment score (0-100).

        Higher score = better alignment.

        Scoring approach:
        - Alignment is calculated only on verifiable claims (aligned + misaligned)
        - Unverified claims don't penalize the score (they indicate code coverage gaps)
        - Misalignments and warnings apply tier-based penalties
        - Critical (tier1) issues cap the maximum score
        """
        if total_claims == 0:
            # Only based on warnings if no claims
            if not warnings:
                return 100.0
            tier1_count = sum(1 for w in warnings if w.tier == IssueTier.RESULTS_INVALID)
            if tier1_count > 0:
                return 25.0  # Still give some score, but capped low
            return max(50, 100 - len(warnings) * 10)

        # Calculate verifiable claims (claims we could actually check)
        verifiable_count = aligned_count + misaligned_count

        # Base score from alignment (60% weight)
        # Only consider claims that could be verified - unverified claims don't hurt the score
        if verifiable_count > 0:
            alignment_ratio = aligned_count / verifiable_count
        else:
            # No claims could be verified - give benefit of doubt with moderate score
            alignment_ratio = 0.7

        alignment_contribution = alignment_ratio * 60

        # Penalty for misalignments based on tier (20% weight)
        # Scale penalty by number of misalignments relative to total verifiable
        misalignment_penalty = 0
        for result in alignment_results:
            if result.status == AlignmentStatus.MISALIGNED:
                tier_penalty = {
                    IssueTier.RESULTS_INVALID: 20,
                    IssueTier.RESULTS_QUESTIONABLE: 12,
                    IssueTier.REPRODUCIBILITY_RISK: 6,
                    IssueTier.MINOR_DISCREPANCY: 2,
                }
                misalignment_penalty += tier_penalty.get(result.tier, 4)

        misalignment_contribution = max(0, 20 - misalignment_penalty)

        # Penalty for catastrophic warnings (20% weight)
        warning_penalty = 0
        for warning in warnings:
            tier_penalty = {
                IssueTier.RESULTS_INVALID: 20,
                IssueTier.RESULTS_QUESTIONABLE: 10,
                IssueTier.REPRODUCIBILITY_RISK: 4,
                IssueTier.MINOR_DISCREPANCY: 1,
            }
            warning_penalty += tier_penalty.get(warning.tier, 2)

        warning_contribution = max(0, 20 - warning_penalty)

        # Total score
        total_score = alignment_contribution + misalignment_contribution + warning_contribution

        # Apply coverage bonus/penalty based on how many claims could be verified
        # This encourages comprehensive code coverage without being too punishing
        if total_claims > 0:
            coverage_ratio = verifiable_count / total_claims
            # Coverage affects a small portion of the score (up to ±10 points)
            coverage_modifier = (coverage_ratio - 0.5) * 20  # Range: -10 to +10
            total_score += coverage_modifier

        # If any tier 1 issues, cap score at 40 (was 30, slightly more lenient)
        has_tier1 = any(w.tier == IssueTier.RESULTS_INVALID for w in warnings)
        has_tier1 = has_tier1 or any(
            r.tier == IssueTier.RESULTS_INVALID
            for r in alignment_results
            if r.status == AlignmentStatus.MISALIGNED
        )

        if has_tier1:
            total_score = min(total_score, 40)

        return round(max(0, min(100, total_score)), 1)

    def _generate_recommendation(
        self,
        overall_score: float,
        critical_issues: int,
        warnings: list[CatastrophicWarning],
        misaligned_count: int,
    ) -> str:
        """Generate a human-readable recommendation."""
        if critical_issues > 0:
            if any(w.category.value == "data_leakage" for w in warnings if w.tier == IssueTier.RESULTS_INVALID):
                return (
                    "CRITICAL: Data leakage detected. Results are likely invalid and should not be trusted. "
                    "Fix the data leakage issues before drawing any conclusions from this code."
                )
            return (
                f"CRITICAL: {critical_issues} critical issue(s) found that likely invalidate results. "
                "Review and fix these issues before trusting any outputs from this code."
            )

        tier2_count = sum(1 for w in warnings if w.tier == IssueTier.RESULTS_QUESTIONABLE)
        if tier2_count > 0 or overall_score < 50:
            return (
                f"WARNING: {tier2_count} significant issue(s) found that may affect results. "
                f"{misaligned_count} claim(s) do not match the code. "
                "Results should be interpreted with caution."
            )

        if overall_score >= 80:
            if misaligned_count == 0 and len(warnings) == 0:
                return "GOOD: Code appears to align well with the description. No major issues detected."
            return (
                f"ACCEPTABLE: Code mostly aligns with description. "
                f"{misaligned_count} minor discrepancy(ies) found. Review flagged items."
            )

        return (
            f"REVIEW NEEDED: Overall alignment score is {overall_score:.0f}/100. "
            f"{misaligned_count} discrepancy(ies) and {len(warnings)} warning(s) found. "
            "Manual review recommended."
        )
