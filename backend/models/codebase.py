from pydantic import BaseModel
from typing import Optional


class CodeFile(BaseModel):
    path: str
    relative_path: str
    content: str
    language: str
    line_count: int


class CodeChunk(BaseModel):
    file_path: str
    relative_path: str
    start_line: int
    end_line: int
    content: str
    language: str


class Codebase(BaseModel):
    path: str
    name: str
    source_type: str  # "local" or "github"
    files: list[CodeFile] = []
    total_lines: int = 0
    github_url: Optional[str] = None


class CodeLoadRequest(BaseModel):
    path: Optional[str] = None
    github_url: Optional[str] = None


class CodeLoadResponse(BaseModel):
    success: bool
    name: str
    source_type: str
    file_count: int
    total_lines: int
    languages: list[str]
