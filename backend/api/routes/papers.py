import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.models.paper import Paper, PaperUploadResponse, TextUploadRequest
from backend.services.paper_parser import parse_pdf, parse_text
from backend.services.state import app_state
from backend.services.history import history_manager
from backend.services.embeddings import get_embedding_service
from backend.services.logging import logger, get_safe_error_message, sanitize_error

router = APIRouter()


async def index_paper_background():
    """Background task to index paper content for code-to-paper search."""
    if app_state.paper:
        embedding_service = get_embedding_service()
        await embedding_service.index_paper(app_state.paper)


@router.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
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
            logger.error(f"Failed to parse PDF: {sanitize_error(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=get_safe_error_message(e, "PDF parsing")
            )

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
    history_manager.add_paper(paper)

    # Trigger paper indexing in background for code-to-paper search
    background_tasks.add_task(index_paper_background)

    return PaperUploadResponse(
        success=True,
        name=paper.name,
        content_preview=paper.content[:500] + "..." if len(paper.content) > 500 else paper.content,
        total_length=len(paper.content),
        page_count=paper.page_count,
    )


@router.post("/text", response_model=PaperUploadResponse)
async def upload_text(background_tasks: BackgroundTasks, request: TextUploadRequest):
    paper = parse_text(request.content, name=request.name)
    app_state.set_paper(paper)
    history_manager.add_paper(paper)

    # Trigger paper indexing in background for code-to-paper search
    background_tasks.add_task(index_paper_background)

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
        "has_pdf": app_state.paper.source_type == "pdf" and app_state.paper.file_path is not None,
    }


@router.delete("/")
async def clear_paper():
    app_state.paper = None
    return {"success": True, "message": "Paper cleared"}


@router.get("/pdf")
async def get_pdf_file():
    """Serve the raw PDF file for rendering in the frontend."""
    if app_state.paper is None:
        raise HTTPException(status_code=404, detail="No paper loaded")

    if app_state.paper.source_type != "pdf" or not app_state.paper.file_path:
        raise HTTPException(status_code=400, detail="Current document is not a PDF")

    if not os.path.exists(app_state.paper.file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        app_state.paper.file_path,
        media_type="application/pdf",
        filename=app_state.paper.name
    )


@router.get("/history")
async def get_paper_history():
    """Get list of previously uploaded papers."""
    papers = history_manager.get_papers()
    return {"papers": [p.model_dump() for p in papers]}


@router.post("/history/{paper_id}/load")
async def load_paper_from_history(background_tasks: BackgroundTasks, paper_id: str):
    """Load a paper from history."""
    history_item = history_manager.get_paper_by_id(paper_id)
    if not history_item:
        raise HTTPException(status_code=404, detail="Paper not found in history")

    if history_item.source_type == "pdf":
        if not history_item.file_path or not os.path.exists(history_item.file_path):
            raise HTTPException(status_code=404, detail="PDF file no longer exists")

        paper = parse_pdf(history_item.file_path)
        paper.name = history_item.name
    else:
        if not history_item.content:
            raise HTTPException(status_code=404, detail="Text content not found")

        paper = parse_text(history_item.content, name=history_item.name)

    app_state.set_paper(paper)

    # Trigger paper indexing in background for code-to-paper search
    background_tasks.add_task(index_paper_background)

    return {
        "success": True,
        "name": paper.name,
        "total_length": len(paper.content),
        "source_type": paper.source_type,
        "has_pdf": paper.source_type == "pdf" and paper.file_path is not None,
    }


@router.get("/index-status")
async def get_paper_index_status():
    """Check if paper is indexed for code-to-paper search."""
    return {
        "indexed": app_state.is_paper_indexed,
        "paper_loaded": app_state.paper is not None,
        "progress": app_state.paper_index_progress,
        "total": app_state.paper_index_total,
    }


@router.delete("/history/{paper_id}")
async def delete_paper_from_history(paper_id: str):
    """Delete a paper from history."""
    if history_manager.delete_paper(paper_id):
        return {"success": True, "message": "Paper deleted from history"}
    raise HTTPException(status_code=404, detail="Paper not found in history")
