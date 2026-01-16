from fastapi import APIRouter, HTTPException

from backend.models.analysis import (
    HighlightAnalysisRequest,
    HighlightAnalysisResponse,
    CodeHighlightAnalysisRequest,
    CodeHighlightAnalysisResponse,
)
from backend.services.alignment import analyze_highlight, analyze_code_highlight
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
        result = await analyze_code_highlight(
            request.highlighted_code, file_path=request.file_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
