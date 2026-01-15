from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.models.codebase import CodeLoadRequest, CodeLoadResponse
from backend.services.code_loader import load_local_codebase, load_github_codebase
from backend.services.embeddings import get_embedding_service
from backend.services.state import app_state
from backend.services.history import history_manager

router = APIRouter()


async def index_codebase_background():
    if app_state.codebase:
        embedding_service = get_embedding_service()
        await embedding_service.index_codebase(app_state.codebase)
        app_state.set_indexed(True)


@router.post("/load", response_model=CodeLoadResponse)
async def load_codebase(request: CodeLoadRequest, background_tasks: BackgroundTasks):
    if not request.path and not request.github_url:
        raise HTTPException(
            status_code=400,
            detail="Either 'path' or 'github_url' must be provided",
        )

    try:
        if request.github_url:
            codebase = load_github_codebase(request.github_url)
        else:
            codebase = load_local_codebase(request.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load codebase: {str(e)}")

    if not codebase.files:
        raise HTTPException(
            status_code=400,
            detail="No supported code files found in the provided path",
        )

    app_state.set_codebase(codebase)
    history_manager.add_codebase(codebase)

    background_tasks.add_task(index_codebase_background)

    languages = list(set(f.language for f in codebase.files))

    return CodeLoadResponse(
        success=True,
        name=codebase.name,
        source_type=codebase.source_type,
        file_count=len(codebase.files),
        total_lines=codebase.total_lines,
        languages=languages,
    )


@router.get("/files")
async def list_files():
    if app_state.codebase is None:
        raise HTTPException(status_code=404, detail="No codebase loaded")

    return {
        "files": [
            {
                "relative_path": f.relative_path,
                "language": f.language,
                "line_count": f.line_count,
            }
            for f in app_state.codebase.files
        ]
    }


@router.get("/file/{file_path:path}")
async def get_file_content(file_path: str):
    if app_state.codebase is None:
        raise HTTPException(status_code=404, detail="No codebase loaded")

    for f in app_state.codebase.files:
        if f.relative_path == file_path:
            return {
                "relative_path": f.relative_path,
                "language": f.language,
                "content": f.content,
                "line_count": f.line_count,
            }

    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


@router.get("/index-status")
async def get_index_status():
    progress_percent = 0
    if app_state.index_total > 0:
        progress_percent = int((app_state.index_progress / app_state.index_total) * 100)

    return {
        "indexed": app_state.is_indexed,
        "codebase_loaded": app_state.codebase is not None,
        "progress": app_state.index_progress,
        "total": app_state.index_total,
        "progress_percent": progress_percent,
    }


@router.delete("/")
async def clear_codebase():
    app_state.codebase = None
    app_state.is_indexed = False
    return {"success": True, "message": "Codebase cleared"}


@router.get("/history")
async def get_codebase_history():
    """Get list of previously loaded codebases."""
    codebases = history_manager.get_codebases()
    return {"codebases": [c.model_dump() for c in codebases]}


@router.post("/history/{codebase_id}/load")
async def load_codebase_from_history(codebase_id: str, background_tasks: BackgroundTasks):
    """Load a codebase from history."""
    history_item = history_manager.get_codebase_by_id(codebase_id)
    if not history_item:
        raise HTTPException(status_code=404, detail="Codebase not found in history")

    try:
        if history_item.source_type == "github" and history_item.github_url:
            codebase = load_github_codebase(history_item.github_url)
        else:
            codebase = load_local_codebase(history_item.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load codebase: {str(e)}")

    if not codebase.files:
        raise HTTPException(
            status_code=400,
            detail="No supported code files found in the codebase",
        )

    app_state.set_codebase(codebase)
    history_manager.add_codebase(codebase)

    background_tasks.add_task(index_codebase_background)

    languages = list(set(f.language for f in codebase.files))

    return CodeLoadResponse(
        success=True,
        name=codebase.name,
        source_type=codebase.source_type,
        file_count=len(codebase.files),
        total_lines=codebase.total_lines,
        languages=languages,
    )


@router.delete("/history/{codebase_id}")
async def delete_codebase_from_history(codebase_id: str):
    """Delete a codebase from history."""
    if history_manager.delete_codebase(codebase_id):
        return {"success": True, "message": "Codebase deleted from history"}
    raise HTTPException(status_code=404, detail="Codebase not found in history")
