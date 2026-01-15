import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import get_settings
from backend.models.paper import PaperUploadResponse, TextUploadRequest
from backend.services.paper_parser import parse_pdf, parse_text
from backend.services.state import app_state

router = APIRouter()


@router.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext == ".pdf":
        file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}.pdf")
        with open(file_path, "wb") as f:
            f.write(content)

        try:
            paper = parse_pdf(file_path)
            paper.name = file.filename
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    elif file_ext in (".txt", ".md", ".tex"):
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content.decode("latin-1")

        paper = parse_text(text_content, name=file.filename)

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF, TXT, MD, or TEX file.",
        )

    app_state.set_paper(paper)

    return PaperUploadResponse(
        success=True,
        name=paper.name,
        content_preview=paper.content[:500] + "..." if len(paper.content) > 500 else paper.content,
        total_length=len(paper.content),
        page_count=paper.page_count,
    )


@router.post("/text", response_model=PaperUploadResponse)
async def upload_text(request: TextUploadRequest):
    paper = parse_text(request.content, name=request.name)
    app_state.set_paper(paper)

    return PaperUploadResponse(
        success=True,
        name=paper.name,
        content_preview=paper.content[:500] + "..." if len(paper.content) > 500 else paper.content,
        total_length=len(paper.content),
    )


@router.get("/content")
async def get_paper_content():
    if app_state.paper is None:
        raise HTTPException(status_code=404, detail="No paper loaded")

    return {
        "name": app_state.paper.name,
        "content": app_state.paper.content,
        "source_type": app_state.paper.source_type,
        "page_count": app_state.paper.page_count,
    }


@router.delete("/")
async def clear_paper():
    app_state.paper = None
    return {"success": True, "message": "Paper cleared"}
