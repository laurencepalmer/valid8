import asyncio
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
        # Paper indexing state for code-to-paper search
        self.is_paper_indexed: bool = False
        self.paper_index_progress: int = 0
        self.paper_index_total: int = 0
        self._precompute_task: Optional[asyncio.Task] = None

    def reset(self):
        self.paper = None
        self.codebase = None
        self.is_indexed = False
        self.index_progress = 0
        self.index_total = 0
        self.is_paper_indexed = False
        self.paper_index_progress = 0
        self.paper_index_total = 0
        if self._precompute_task and not self._precompute_task.done():
            self._precompute_task.cancel()
        self._precompute_task = None

    def set_paper(self, paper: Paper):
        self.paper = paper
        # Reset paper index when new paper is loaded
        self.is_paper_indexed = False
        self.paper_index_progress = 0
        self.paper_index_total = 0

    def set_codebase(self, codebase: Codebase):
        self.codebase = codebase
        self.is_indexed = False
        self.index_progress = 0
        self.index_total = 0

    def set_indexed(self, indexed: bool):
        self.is_indexed = indexed
        if indexed:
            self.index_progress = self.index_total
            self._maybe_precompute()

    def set_index_progress(self, progress: int, total: int):
        self.index_progress = progress
        self.index_total = total

    def set_paper_indexed(self, indexed: bool):
        self.is_paper_indexed = indexed
        if indexed:
            self.paper_index_progress = self.paper_index_total
            self._maybe_precompute()

    def set_paper_index_progress(self, progress: int, total: int):
        self.paper_index_progress = progress
        self.paper_index_total = total

    def _maybe_precompute(self):
        """Trigger background pre-computation when both paper and codebase are indexed."""
        # Disabled — precompute was contending with user queries for resources.
        # Re-enable by uncommenting the block below.
        # if self.is_indexed and self.is_paper_indexed:
        #     from backend.services.precompute import precompute_mappings
        #
        #     # Cancel any previous precompute
        #     if self._precompute_task and not self._precompute_task.done():
        #         self._precompute_task.cancel()
        #     try:
        #         loop = asyncio.get_running_loop()
        #         self._precompute_task = loop.create_task(precompute_mappings())
        #     except RuntimeError:
        #         pass  # No running loop, skip
        pass


app_state = AppState()
