from typing import Optional
from backend.models.paper import Paper
from backend.models.codebase import Codebase


class AppState:
    def __init__(self):
        self.paper: Optional[Paper] = None
        self.codebase: Optional[Codebase] = None
        self.is_indexed: bool = False
        self.index_progress: int = 0
        self.index_total: int = 0

    def reset(self):
        self.paper = None
        self.codebase = None
        self.is_indexed = False
        self.index_progress = 0
        self.index_total = 0

    def set_paper(self, paper: Paper):
        self.paper = paper
        self.is_indexed = False

    def set_codebase(self, codebase: Codebase):
        self.codebase = codebase
        self.is_indexed = False
        self.index_progress = 0
        self.index_total = 0

    def set_indexed(self, indexed: bool):
        self.is_indexed = indexed
        if indexed:
            self.index_progress = self.index_total

    def set_index_progress(self, progress: int, total: int):
        self.index_progress = progress
        self.index_total = total


app_state = AppState()
