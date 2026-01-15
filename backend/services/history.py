import json
import os
import uuid
from datetime import datetime
from typing import Optional

from backend.config import get_settings
from backend.models.history import PaperHistoryItem, CodebaseHistoryItem
from backend.models.paper import Paper
from backend.models.codebase import Codebase


class HistoryManager:
    def __init__(self):
        settings = get_settings()
        self.history_file = os.path.join(settings.upload_dir, "history.json")
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Ensure the history file and directory exist."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            self._save_history({"papers": [], "codebases": []})

    def _load_history(self) -> dict:
        """Load history from file."""
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"papers": [], "codebases": []}

    def _save_history(self, history: dict):
        """Save history to file."""
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def add_paper(self, paper: Paper) -> PaperHistoryItem:
        """Add a paper to history."""
        history = self._load_history()

        # Check if paper already exists (by name and source_type)
        for existing in history["papers"]:
            if existing["name"] == paper.name and existing["source_type"] == paper.source_type:
                # Update existing entry
                existing["uploaded_at"] = datetime.now().isoformat()
                if paper.file_path:
                    existing["file_path"] = paper.file_path
                if paper.source_type == "text":
                    existing["content"] = paper.content
                existing["page_count"] = paper.page_count
                self._save_history(history)
                return PaperHistoryItem(**existing)

        # Create new entry
        item = PaperHistoryItem(
            id=str(uuid.uuid4()),
            name=paper.name,
            source_type=paper.source_type,
            file_path=paper.file_path if paper.source_type == "pdf" else None,
            content=paper.content if paper.source_type == "text" else None,
            page_count=paper.page_count,
            uploaded_at=datetime.now().isoformat(),
        )

        history["papers"].insert(0, item.model_dump())
        # Keep only last 20 papers
        history["papers"] = history["papers"][:20]
        self._save_history(history)
        return item

    def add_codebase(self, codebase: Codebase) -> CodebaseHistoryItem:
        """Add a codebase to history."""
        history = self._load_history()

        # Check if codebase already exists (by path)
        for existing in history["codebases"]:
            if existing["path"] == codebase.path:
                # Update existing entry
                existing["loaded_at"] = datetime.now().isoformat()
                existing["file_count"] = len(codebase.files)
                existing["total_lines"] = codebase.total_lines
                self._save_history(history)
                return CodebaseHistoryItem(**existing)

        # Create new entry
        item = CodebaseHistoryItem(
            id=str(uuid.uuid4()),
            name=codebase.name,
            path=codebase.path,
            source_type=codebase.source_type,
            github_url=codebase.github_url,
            file_count=len(codebase.files),
            total_lines=codebase.total_lines,
            loaded_at=datetime.now().isoformat(),
        )

        history["codebases"].insert(0, item.model_dump())
        # Keep only last 20 codebases
        history["codebases"] = history["codebases"][:20]
        self._save_history(history)
        return item

    def get_papers(self) -> list[PaperHistoryItem]:
        """Get all papers from history."""
        history = self._load_history()
        return [PaperHistoryItem(**p) for p in history["papers"]]

    def get_codebases(self) -> list[CodebaseHistoryItem]:
        """Get all codebases from history."""
        history = self._load_history()
        return [CodebaseHistoryItem(**c) for c in history["codebases"]]

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperHistoryItem]:
        """Get a paper by ID."""
        history = self._load_history()
        for p in history["papers"]:
            if p["id"] == paper_id:
                return PaperHistoryItem(**p)
        return None

    def get_codebase_by_id(self, codebase_id: str) -> Optional[CodebaseHistoryItem]:
        """Get a codebase by ID."""
        history = self._load_history()
        for c in history["codebases"]:
            if c["id"] == codebase_id:
                return CodebaseHistoryItem(**c)
        return None

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from history."""
        history = self._load_history()
        original_len = len(history["papers"])
        history["papers"] = [p for p in history["papers"] if p["id"] != paper_id]
        if len(history["papers"]) < original_len:
            self._save_history(history)
            return True
        return False

    def delete_codebase(self, codebase_id: str) -> bool:
        """Delete a codebase from history."""
        history = self._load_history()
        original_len = len(history["codebases"])
        history["codebases"] = [c for c in history["codebases"] if c["id"] != codebase_id]
        if len(history["codebases"]) < original_len:
            self._save_history(history)
            return True
        return False


history_manager = HistoryManager()
