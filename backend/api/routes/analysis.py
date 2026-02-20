from fastapi import APIRouter, HTTPException

from backend.models.analysis import (
    HighlightAnalysisRequest,
    HighlightAnalysisResponse,
    CodeHighlightAnalysisRequest,
    CodeHighlightAnalysisResponse,
)
from backend.services.alignment import analyze_highlight, analyze_code_highlight
from backend.services.precompute import get_precompute_cache
from backend.services.state import app_state
from backend.services.logging import logger, get_safe_error_message, sanitize_error

router = APIRouter()


@router.post("/highlight", response_model=HighlightAnalysisResponse)
async def analyze_highlighted_text(request: HighlightAnalysisRequest):
    if app_state.codebase is None:
        raise HTTPException(status_code=400, detail="No codebase loaded")

    if not app_state.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="Codebase is still being indexed. Please wait and try again.",
        )

    if not request.highlighted_text.strip():
        raise HTTPException(status_code=400, detail="Highlighted text cannot be empty")

    try:
        async with get_precompute_cache().user_query():
            result = await analyze_highlight(request.highlighted_text)
            return result
    except Exception as e:
        logger.error(f"Highlight analysis failed: {sanitize_error(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=get_safe_error_message(e, "Analysis")
        )


@router.get("/precompute-status")
async def get_precompute_status():
    """Return progress of background pre-computation cache."""
    cache = get_precompute_cache()
    return {
        "is_running": cache.is_running,
        "progress": cache.progress,
        "total": cache.total,
        "cached_count": len(cache.cache),
    }


@router.post("/code-highlight", response_model=CodeHighlightAnalysisResponse)
async def analyze_code_highlighted_text(request: CodeHighlightAnalysisRequest):
    """Analyze highlighted code to find relevant paper sections."""
    if app_state.paper is None:
        raise HTTPException(status_code=400, detail="No paper loaded")

    if not app_state.is_paper_indexed:
        raise HTTPException(
            status_code=400,
            detail="Paper is still being indexed. Please wait and try again.",
        )

    if not request.highlighted_code.strip():
        raise HTTPException(status_code=400, detail="Highlighted code cannot be empty")

    try:
        async with get_precompute_cache().user_query():
            result = await analyze_code_highlight(
                request.highlighted_code, file_path=request.file_path
            )
            return result
    except Exception as e:
        logger.error(f"Code highlight analysis failed: {sanitize_error(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=get_safe_error_message(e, "Analysis")
        )
