from pydantic import BaseModel
from typing import Optional


class CodeReference(BaseModel):
    file_path: str
    relative_path: str
    start_line: int
    end_line: int
    content: str
    relevance_score: float
    explanation: str


class HighlightAnalysisRequest(BaseModel):
    highlighted_text: str


class HighlightAnalysisResponse(BaseModel):
    highlighted_text: str
    code_references: list[CodeReference]
    summary: str


class AlignmentCheckRequest(BaseModel):
    summary: str
    file_paths: Optional[list[str]] = None  # If None, check against entire codebase


class AlignmentIssue(BaseModel):
    issue_type: str  # "missing", "incorrect", "extra"
    description: str
    summary_excerpt: Optional[str] = None
    code_reference: Optional[CodeReference] = None


class AlignmentCheckResponse(BaseModel):
    alignment_score: float  # 0.0 to 1.0
    summary: str
    is_aligned: bool
    issues: list[AlignmentIssue]
    suggestions: list[str]


# Code-to-PDF Search Models
class PaperChunk(BaseModel):
    """Represents a searchable chunk of paper content."""

    paper_name: str
    content: str
    start_idx: int
    end_idx: int
    page: Optional[int] = None


class PaperReference(BaseModel):
    """A reference to a section in the paper found by code search."""

    content: str
    page: Optional[int] = None
    start_idx: int
    end_idx: int
    relevance_score: float
    explanation: str


class CodeHighlightAnalysisRequest(BaseModel):
    """Request to analyze highlighted code text."""

    highlighted_code: str
    file_path: Optional[str] = None  # Context about which file the code is from


class CodeHighlightAnalysisResponse(BaseModel):
    """Response containing paper references for highlighted code."""

    highlighted_code: str
    paper_references: list[PaperReference]
    summary: str
