from pydantic import BaseModel
from typing import Optional


class PaperHistoryItem(BaseModel):
    id: str
    name: str
    source_type: str  # "pdf" or "text"
    file_path: Optional[str] = None  # For PDFs
    content: Optional[str] = None  # For text documents
    page_count: Optional[int] = None
    uploaded_at: str


class CodebaseHistoryItem(BaseModel):
    id: str
    name: str
    path: str
    source_type: str  # "local" or "github"
    github_url: Optional[str] = None
    file_count: int
    total_lines: int
    loaded_at: str
