from pydantic import BaseModel
from typing import Optional


class PaperSection(BaseModel):
    text: str
    start_idx: int
    end_idx: int
    page: Optional[int] = None


class Paper(BaseModel):
    name: str
    content: str
    sections: list[PaperSection] = []
    source_type: str  # "pdf" or "text"
    page_count: Optional[int] = None


class PaperUploadResponse(BaseModel):
    success: bool
    name: str
    content_preview: str
    total_length: int
    page_count: Optional[int] = None


class TextUploadRequest(BaseModel):
    name: str
    content: str
