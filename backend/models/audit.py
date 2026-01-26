"""Data models for code-description alignment audit."""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Type of claim extracted from description."""

    DATA_PREPROCESSING = "data_preprocessing"
    DATA_SPLIT = "data_split"
    MODEL_ARCHITECTURE = "model_architecture"
    TRAINING_PROCEDURE = "training_procedure"
    EVALUATION_METRICS = "evaluation_metrics"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    HYPERPARAMETER = "hyperparameter"
    OTHER = "other"


class IssueTier(str, Enum):
    """Priority tier based on impact."""

    RESULTS_INVALID = "tier1"  # Data leakage, wrong evaluation
    RESULTS_QUESTIONABLE = "tier2"  # Statistical issues, missing seeds
    REPRODUCIBILITY_RISK = "tier3"  # Environment issues
    MINOR_DISCREPANCY = "tier4"  # Small differences


class CatastrophicCategory(str, Enum):
    """Category of catastrophic pattern."""

    DATA_LEAKAGE = "data_leakage"
    EVALUATION_ERROR = "evaluation_error"
    TRAINING_ISSUE = "training_issue"
    REPRODUCIBILITY = "reproducibility"
    DATA_INTEGRITY = "data_integrity"
    IMPLEMENTATION_BUG = "implementation_bug"
    STATISTICAL_ERROR = "statistical_error"


class AlignmentStatus(str, Enum):
    """Status of claim alignment."""

    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    UNVERIFIED = "unverified"
    MISSING = "missing"


# --- Claim Models ---


class Claim(BaseModel):
    """A verifiable claim extracted from the description."""

    id: str
    text: str  # Original text from description
    claim_type: ClaimType
    verifiable: bool = True
    keywords: list[str] = Field(default_factory=list)
    source_location: Optional[str] = None  # Page/section in description
    expected_value: Optional[str] = None  # e.g., "5" for "5-fold CV"
    context: Optional[str] = None  # Surrounding context


# --- Code Analysis Models ---


class CodeBehavior(BaseModel):
    """Describes what a piece of code actually does."""

    chunk_id: str
    file_path: str
    relative_path: str
    start_line: int
    end_line: int
    behavior_type: ClaimType
    description: str  # LLM-generated description of what code does
    code_snippet: str
    confidence: float = 1.0
    actual_value: Optional[str] = None  # e.g., "3" for actual 3-fold CV
    extracted_values: dict[str, str] = Field(default_factory=dict)  # Pattern-extracted values
    from_pattern: bool = False  # True if detected via pattern, not LLM


# --- Catastrophic Warning Models ---


class CatastrophicWarning(BaseModel):
    """A critical issue that could invalidate results."""

    id: str
    category: CatastrophicCategory
    pattern_type: str  # e.g., "preprocessing_before_split"
    tier: IssueTier
    file_path: str
    relative_path: str
    line_start: int
    line_end: int
    description: str
    code_snippet: str
    why_catastrophic: str  # Explanation of impact
    recommendation: str
    confidence: float = 1.0
    suppressed: bool = False
    suppression_reason: Optional[str] = None


# --- Alignment Models ---


class AlignmentResult(BaseModel):
    """Result of comparing a claim against code."""

    claim: Claim
    status: AlignmentStatus
    matched_code: list[CodeBehavior] = Field(default_factory=list)
    explanation: str
    confidence: float = 1.0
    tier: IssueTier = IssueTier.MINOR_DISCREPANCY
    specific_issues: list[str] = Field(default_factory=list)


# --- Report Models ---


class AuditSummary(BaseModel):
    """Summary statistics for an audit."""

    total_claims: int = 0
    aligned_count: int = 0
    misaligned_count: int = 0
    unverified_count: int = 0
    missing_count: int = 0
    catastrophic_warnings_by_category: dict[str, int] = Field(default_factory=dict)
    catastrophic_warnings_by_tier: dict[str, int] = Field(default_factory=dict)
    overall_score: float = 0.0  # 0-100 alignment score
    recommendation: str = ""


class AuditReport(BaseModel):
    """Complete audit report."""

    audit_id: str
    summary: AuditSummary
    misalignments: list[AlignmentResult] = Field(default_factory=list)
    catastrophic_warnings: list[CatastrophicWarning] = Field(default_factory=list)
    verified_claims: list[AlignmentResult] = Field(default_factory=list)
    unverified_claims: list[AlignmentResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    description_source: str = ""
    codebase_name: str = ""
    status: Literal["running", "completed", "failed"] = "running"
    error_message: Optional[str] = None
    progress: float = 0.0
    current_step: str = ""


# --- False Positive Models ---


class FalsePositiveScope(str, Enum):
    """Scope of false positive suppression."""

    THIS_AUDIT = "this_audit"
    THIS_CODEBASE = "this_codebase"
    THIS_PATTERN = "this_pattern"


class FalsePositiveRecord(BaseModel):
    """Record of a user-marked false positive."""

    id: str
    audit_id: str
    warning_id: str  # ID of the warning marked as FP
    warning_type: str  # e.g., "leakage.preprocessing_before_split"
    category: CatastrophicCategory
    code_hash: str  # Hash of the flagged code snippet
    file_path: str
    line_start: int
    line_end: int
    user_reason: str
    timestamp: datetime = Field(default_factory=datetime.now)
    applies_to: FalsePositiveScope = FalsePositiveScope.THIS_AUDIT
    codebase_hash: Optional[str] = None


# --- API Request/Response Models ---


class AuditRequest(BaseModel):
    """Request to start an audit."""

    description_source: Literal["paper", "text"] = "paper"
    text_content: Optional[str] = None  # If description_source is "text"
    focus_areas: Optional[list[ClaimType]] = None
    catastrophic_categories: Optional[list[CatastrophicCategory]] = None
    min_tier: IssueTier = IssueTier.MINOR_DISCREPANCY


class AuditStatusResponse(BaseModel):
    """Response for audit status check."""

    audit_id: str
    status: Literal["running", "completed", "failed"]
    progress: float
    current_step: str
    claims_extracted: int = 0
    behaviors_analyzed: int = 0
    warnings_found: int = 0


class FalsePositiveRequest(BaseModel):
    """Request to mark a warning as false positive."""

    warning_id: str
    reason: str
    applies_to: FalsePositiveScope = FalsePositiveScope.THIS_AUDIT
