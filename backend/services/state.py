from typing import Optional
from backend.models.paper import Paper
from backend.models.codebase import Codebase


class AppState:
    def __init__(self):
        self.paper: Optional[Paper] = None
        self.codebase: Optional[Codebase] = None
        self.is_indexed: bool = False

    def reset(self):
        self.paper = None
        self.codebase = None
        self.is_indexed = False

    def set_paper(self, paper: Paper):
        self.paper = paper
        self.is_indexed = False

    def set_codebase(self, codebase: Codebase):
        self.codebase = codebase
        self.is_indexed = False

    def set_indexed(self, indexed: bool):
        self.is_indexed = indexed


app_state = AppState()
