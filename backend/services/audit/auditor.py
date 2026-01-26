"""Main audit orchestrator."""

import uuid
from datetime import datetime
from typing import Optional

from backend.models.audit import (
    AuditReport,
    AuditRequest,
    AuditSummary,
    AlignmentResult,
    AlignmentStatus,
    CatastrophicCategory,
    CatastrophicWarning,
    Claim,
    ClaimType,
    CodeBehavior,
    IssueTier,
)
from backend.models.codebase import Codebase
from backend.models.paper import Paper
from backend.services.audit.claim_extractor import ClaimExtractor
from backend.services.audit.code_analyzer import CodeAnalyzer
from backend.services.audit.catastrophic_detector import CatastrophicDetector
from backend.services.audit.alignment_checker import AlignmentChecker
from backend.services.audit.report_generator import ReportGenerator
from backend.services.audit.false_positive_store import FalsePositiveStore
from backend.services.logging import logger


class Auditor:
    """
    Orchestrates the audit pipeline.

    Pipeline:
    1. Extract claims from description (paper/text)
    2. Analyze code behaviors
    3. Detect catastrophic patterns
    4. Check alignment between claims and code
    5. Generate report
    """

    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.code_analyzer = CodeAnalyzer()
        self.catastrophic_detector = CatastrophicDetector()
        self.alignment_checker = AlignmentChecker()
        self.report_generator = ReportGenerator()
        self.fp_store = FalsePositiveStore()

        # Track running audits
        self._audits: dict[str, AuditReport] = {}

    async def run_audit(
        self,
        paper: Optional[Paper],
        codebase: Codebase,
        request: AuditRequest,
        report: Optional[AuditReport] = None,
    ) -> AuditReport:
        """
        Run a full audit of code against description.

        Args:
            paper: Paper object (if description_source is "paper")
            codebase: Codebase to audit
            request: Audit configuration
            report: Optional existing report to update (for progress tracking)

        Returns:
            Complete AuditReport
        """
        # Use provided report or create a new one
        if report is None:
            audit_id = str(uuid.uuid4())
            report = AuditReport(
                audit_id=audit_id,
                summary=AuditSummary(),
                description_source=request.description_source,
                codebase_name=codebase.name,
                status="running",
                current_step="Initializing audit",
            )

        audit_id = report.audit_id
        self._audits[audit_id] = report

        try:
            # Step 1: Get description content
            report.current_step = "Extracting claims from description"
            report.progress = 0.1
            logger.info(f"Audit {audit_id}: Starting audit on {codebase.name}")

            if request.description_source == "paper" and paper:
                description_text = paper.content
                logger.info(f"Audit {audit_id}: Using paper as description source")
            elif request.text_content:
                description_text = request.text_content
                logger.info(f"Audit {audit_id}: Using text as description source")
            else:
                raise ValueError("No description provided")

            # Step 2: Extract claims
            claims = await self.claim_extractor.extract_claims(
                description_text,
                focus_areas=request.focus_areas,
            )
            logger.info(f"Audit {audit_id}: Extracted {len(claims)} claims")
            report.progress = 0.25

            # Step 3: Analyze code behaviors
            report.current_step = "Analyzing code behaviors"
            behaviors = await self.code_analyzer.analyze_codebase(
                codebase,
                claim_types=[c.claim_type for c in claims],
            )
            logger.info(f"Audit {audit_id}: Analyzed {len(behaviors)} code behaviors")
            report.progress = 0.45

            # Step 4: Detect catastrophic patterns
            report.current_step = "Detecting catastrophic patterns"
            warnings = await self.catastrophic_detector.detect_all(
                codebase,
                categories=request.catastrophic_categories,
            )
            logger.info(f"Audit {audit_id}: Detected {len(warnings)} potential issues")

            # Filter by false positives
            warnings = self.fp_store.filter_warnings(
                warnings,
                audit_id=audit_id,
                codebase_hash=self._hash_codebase(codebase),
            )

            # Filter by minimum tier
            warnings = [
                w for w in warnings
                if self._tier_value(w.tier) <= self._tier_value(request.min_tier)
            ]
            logger.info(f"Audit {audit_id}: {len(warnings)} warnings after filtering")

            report.catastrophic_warnings = warnings
            report.progress = 0.65

            # Step 5: Check alignment
            report.current_step = "Checking alignment"
            alignment_results = await self.alignment_checker.check_all(
                claims=claims,
                behaviors=behaviors,
            )
            report.progress = 0.85

            # Categorize results
            for result in alignment_results:
                if result.status == AlignmentStatus.ALIGNED:
                    report.verified_claims.append(result)
                elif result.status == AlignmentStatus.MISALIGNED:
                    report.misalignments.append(result)
                else:
                    report.unverified_claims.append(result)

            logger.info(
                f"Audit {audit_id}: Alignment results - "
                f"{len(report.verified_claims)} aligned, "
                f"{len(report.misalignments)} misaligned, "
                f"{len(report.unverified_claims)} unverified"
            )

            # Step 6: Generate summary
            report.current_step = "Generating report"
            report.summary = self.report_generator.generate_summary(
                claims=claims,
                alignment_results=alignment_results,
                warnings=warnings,
            )

            report.status = "completed"
            report.progress = 1.0
            report.current_step = "Audit complete"
            logger.info(
                f"Audit {audit_id}: Complete - score {report.summary.overall_score}/100"
            )

        except Exception as e:
            report.status = "failed"
            report.error_message = str(e)
            report.current_step = "Audit failed"
            logger.error(f"Audit {audit_id}: Failed - {e}")

        return report

    def get_audit_status(self, audit_id: str) -> Optional[AuditReport]:
        """Get the current status of an audit."""
        return self._audits.get(audit_id)

    def _hash_codebase(self, codebase: Codebase) -> str:
        """Generate a hash for the codebase for FP matching."""
        import hashlib
        content = f"{codebase.path}:{codebase.name}:{len(codebase.files)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _tier_value(self, tier: IssueTier) -> int:
        """Convert tier to numeric value for comparison."""
        tier_values = {
            IssueTier.RESULTS_INVALID: 1,
            IssueTier.RESULTS_QUESTIONABLE: 2,
            IssueTier.REPRODUCIBILITY_RISK: 3,
            IssueTier.MINOR_DISCREPANCY: 4,
        }
        return tier_values.get(tier, 4)


# Global auditor instance
_auditor: Optional[Auditor] = None


def get_auditor() -> Auditor:
    """Get the global auditor instance."""
    global _auditor
    if _auditor is None:
        _auditor = Auditor()
    return _auditor
