from fastapi import APIRouter, HTTPException

from backend.models.analysis import (
    HighlightAnalysisRequest,
    HighlightAnalysisResponse,
    AlignmentCheckRequest,
    AlignmentCheckResponse,
)
from backend.services.alignment import analyze_highlight, check_alignment
from backend.services.state import app_state

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
        result = await analyze_highlight(request.highlighted_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/alignment", response_model=AlignmentCheckResponse)
async def check_code_alignment(request: AlignmentCheckRequest):
    if app_state.codebase is None:
        raise HTTPException(status_code=400, detail="No codebase loaded")

    if not app_state.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="Codebase is still being indexed. Please wait and try again.",
        )

    if not request.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty")

    try:
        result = await check_alignment(request.summary, request.file_paths)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alignment check failed: {str(e)}")
