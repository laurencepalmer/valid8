"""API routes for code-description alignment audit."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from backend.models.audit import (
    AuditReport,
    AuditRequest,
    AuditStatusResponse,
    FalsePositiveRequest,
    FalsePositiveScope,
)
from backend.services.audit import get_auditor
from backend.services.audit.false_positive_store import FalsePositiveStore
from backend.services.state import app_state
from backend.services.logging import logger, sanitize_error

router = APIRouter()


class AuditComparisonResponse(BaseModel):
    """Response model for audit comparison."""
    base_audit_id: str
    compare_audit_id: str
    score_change: float
    new_warnings: list[dict]
    resolved_warnings: list[dict]
    new_misalignments: list[dict]
    resolved_misalignments: list[dict]
    summary: str


# Store for tracking audit tasks
_audit_tasks: dict[str, AuditReport] = {}


async def run_audit_task(audit_id: str, request: AuditRequest):
    """Background task to run audit."""
    auditor = get_auditor()

    # Get the shared report object that the frontend polls
    report = _audit_tasks[audit_id]

    try:
        await auditor.run_audit(
            paper=app_state.paper,
            codebase=app_state.codebase,
            request=request,
            report=report,  # Pass the shared report for progress updates
        )
    except Exception as e:
        logger.error(f"Audit failed: {sanitize_error(e)}", exc_info=True)
        report.status = "failed"
        report.error_message = str(e)
        report.current_step = "Audit failed"


@router.post("/run")
async def start_audit(
    request: AuditRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a new audit.

    Requires both a paper and codebase to be loaded.
    """
    # Validate prerequisites
    if request.description_source == "paper" and app_state.paper is None:
        raise HTTPException(
            status_code=400,
            detail="No paper loaded. Upload a paper first or use description_source='text'",
        )

    if request.description_source == "text" and not request.text_content:
        raise HTTPException(
            status_code=400,
            detail="text_content is required when description_source='text'",
        )

    if app_state.codebase is None:
        raise HTTPException(
            status_code=400,
            detail="No codebase loaded. Load a codebase first.",
        )

    # Generate audit ID
    import uuid
    audit_id = str(uuid.uuid4())

    # Initialize placeholder report
    from backend.models.audit import AuditSummary
    _audit_tasks[audit_id] = AuditReport(
        audit_id=audit_id,
        summary=AuditSummary(),
        status="running",
        current_step="Initializing",
        description_source=request.description_source,
        codebase_name=app_state.codebase.name,
    )

    # Start background task
    background_tasks.add_task(run_audit_task, audit_id, request)

    return {
        "audit_id": audit_id,
        "status": "running",
        "message": "Audit started",
    }


@router.get("/{audit_id}/status", response_model=AuditStatusResponse)
async def get_audit_status(audit_id: str):
    """Get the current status of an audit."""
    report = _audit_tasks.get(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit not found")

    return AuditStatusResponse(
        audit_id=audit_id,
        status=report.status,
        progress=report.progress,
        current_step=report.current_step,
        claims_extracted=len(report.verified_claims) + len(report.misalignments) + len(report.unverified_claims),
        behaviors_analyzed=sum(len(r.matched_code) for r in report.verified_claims + report.misalignments),
        warnings_found=len(report.catastrophic_warnings),
    )


@router.get("/{audit_id}/report", response_model=AuditReport)
async def get_audit_report(audit_id: str):
    """Get the full audit report."""
    report = _audit_tasks.get(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit not found")

    if report.status == "running":
        raise HTTPException(
            status_code=202,
            detail="Audit still in progress",
        )

    return report


@router.get("/history")
async def get_audit_history():
    """Get list of previous audits."""
    # Return basic info about all audits
    audits = []
    for audit_id, report in _audit_tasks.items():
        audits.append({
            "audit_id": audit_id,
            "status": report.status,
            "timestamp": report.timestamp.isoformat(),
            "codebase_name": report.codebase_name,
            "description_source": report.description_source,
            "score": report.summary.overall_score if report.summary else None,
        })

    # Sort by timestamp descending
    audits.sort(key=lambda x: x["timestamp"], reverse=True)

    return {"audits": audits}


@router.delete("/{audit_id}")
async def delete_audit(audit_id: str):
    """Delete an audit from history."""
    if audit_id not in _audit_tasks:
        raise HTTPException(status_code=404, detail="Audit not found")

    del _audit_tasks[audit_id]
    return {"success": True, "message": "Audit deleted"}


# --- False Positive Management ---


@router.post("/{audit_id}/false-positive")
async def mark_false_positive(
    audit_id: str,
    request: FalsePositiveRequest,
):
    """Mark a warning as a false positive."""
    report = _audit_tasks.get(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit not found")

    # Find the warning
    warning = None
    for w in report.catastrophic_warnings:
        if w.id == request.warning_id:
            warning = w
            break

    if not warning:
        raise HTTPException(status_code=404, detail="Warning not found")

    # Get codebase hash
    codebase_hash = None
    if app_state.codebase:
        import hashlib
        content = f"{app_state.codebase.path}:{app_state.codebase.name}"
        codebase_hash = hashlib.md5(content.encode()).hexdigest()

    # Add false positive
    fp_store = FalsePositiveStore()
    record = fp_store.add_false_positive(
        warning=warning,
        audit_id=audit_id,
        reason=request.reason,
        scope=request.applies_to,
        codebase_hash=codebase_hash,
    )

    # Mark warning as suppressed in current report
    warning.suppressed = True
    warning.suppression_reason = request.reason

    return {
        "success": True,
        "false_positive_id": record.id,
        "message": f"Warning marked as false positive ({request.applies_to.value})",
    }


@router.get("/false-positives")
async def get_false_positives(codebase_hash: str = None):
    """Get list of false positives."""
    fp_store = FalsePositiveStore()
    records = fp_store.get_false_positives(codebase_hash=codebase_hash)

    return {
        "false_positives": [r.model_dump() for r in records],
    }


@router.delete("/false-positive/{fp_id}")
async def remove_false_positive(fp_id: str):
    """Remove a false positive mark."""
    fp_store = FalsePositiveStore()
    if fp_store.remove_false_positive(fp_id):
        return {"success": True, "message": "False positive removed"}
    raise HTTPException(status_code=404, detail="False positive not found")


# --- Audit Comparison ---


@router.get("/compare/{base_id}/{compare_id}", response_model=AuditComparisonResponse)
async def compare_audits(base_id: str, compare_id: str):
    """
    Compare two audits to see what changed.

    Args:
        base_id: The baseline audit ID (usually older)
        compare_id: The audit to compare against (usually newer)

    Returns:
        Comparison showing new/resolved issues and score change
    """
    base_report = _audit_tasks.get(base_id)
    compare_report = _audit_tasks.get(compare_id)

    if not base_report:
        raise HTTPException(status_code=404, detail=f"Base audit {base_id} not found")
    if not compare_report:
        raise HTTPException(status_code=404, detail=f"Compare audit {compare_id} not found")

    if base_report.status != "completed" or compare_report.status != "completed":
        raise HTTPException(status_code=400, detail="Both audits must be completed")

    # Calculate score change
    score_change = compare_report.summary.overall_score - base_report.summary.overall_score

    # Find new and resolved warnings
    base_warning_keys = {
        (w.pattern_type, w.relative_path, w.line_start)
        for w in base_report.catastrophic_warnings
        if not w.suppressed
    }
    compare_warning_keys = {
        (w.pattern_type, w.relative_path, w.line_start)
        for w in compare_report.catastrophic_warnings
        if not w.suppressed
    }

    new_warning_keys = compare_warning_keys - base_warning_keys
    resolved_warning_keys = base_warning_keys - compare_warning_keys

    new_warnings = [
        {"pattern": w.pattern_type, "file": w.relative_path, "description": w.description}
        for w in compare_report.catastrophic_warnings
        if (w.pattern_type, w.relative_path, w.line_start) in new_warning_keys
    ]

    resolved_warnings = [
        {"pattern": w.pattern_type, "file": w.relative_path, "description": w.description}
        for w in base_report.catastrophic_warnings
        if (w.pattern_type, w.relative_path, w.line_start) in resolved_warning_keys
    ]

    # Find new and resolved misalignments
    base_misalignment_keys = {
        (m.claim.text[:100], m.claim.claim_type)
        for m in base_report.misalignments
    }
    compare_misalignment_keys = {
        (m.claim.text[:100], m.claim.claim_type)
        for m in compare_report.misalignments
    }

    new_misalignment_keys = compare_misalignment_keys - base_misalignment_keys
    resolved_misalignment_keys = base_misalignment_keys - compare_misalignment_keys

    new_misalignments = [
        {"claim": m.claim.text, "explanation": m.explanation}
        for m in compare_report.misalignments
        if (m.claim.text[:100], m.claim.claim_type) in new_misalignment_keys
    ]

    resolved_misalignments = [
        {"claim": m.claim.text, "explanation": m.explanation}
        for m in base_report.misalignments
        if (m.claim.text[:100], m.claim.claim_type) in resolved_misalignment_keys
    ]

    # Generate summary
    if score_change > 5:
        summary = f"Improvement: Score increased by {score_change:.1f} points."
    elif score_change < -5:
        summary = f"Regression: Score decreased by {abs(score_change):.1f} points."
    else:
        summary = f"Minimal change: Score changed by {score_change:.1f} points."

    if new_warnings:
        summary += f" {len(new_warnings)} new warning(s)."
    if resolved_warnings:
        summary += f" {len(resolved_warnings)} warning(s) resolved."

    return AuditComparisonResponse(
        base_audit_id=base_id,
        compare_audit_id=compare_id,
        score_change=score_change,
        new_warnings=new_warnings,
        resolved_warnings=resolved_warnings,
        new_misalignments=new_misalignments,
        resolved_misalignments=resolved_misalignments,
        summary=summary,
    )
