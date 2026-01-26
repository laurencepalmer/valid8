"""Store and manage false positive records."""

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from backend.config import get_settings
from backend.models.audit import (
    CatastrophicWarning,
    FalsePositiveRecord,
    FalsePositiveScope,
)


class FalsePositiveStore:
    """Manage false positive suppressions."""

    def __init__(self):
        settings = get_settings()
        self.base_dir = os.path.join(settings.upload_dir, "..", "audit", "false_positives")
        self.by_codebase_dir = os.path.join(self.base_dir, "by_codebase")
        self.global_file = os.path.join(self.base_dir, "global.json")
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure storage directories exist."""
        os.makedirs(self.by_codebase_dir, exist_ok=True)
        if not os.path.exists(self.global_file):
            self._save_json(self.global_file, {"false_positives": []})

    def _load_json(self, path: str) -> dict:
        """Load JSON from file."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"false_positives": []}

    def _save_json(self, path: str, data: dict):
        """Save JSON to file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _get_codebase_file(self, codebase_hash: str) -> str:
        """Get path to codebase-specific FP file."""
        return os.path.join(self.by_codebase_dir, f"{codebase_hash}.json")

    def _hash_code(self, code: str) -> str:
        """Hash code snippet for matching."""
        # Normalize whitespace for more robust matching
        normalized = " ".join(code.split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def add_false_positive(
        self,
        warning: CatastrophicWarning,
        audit_id: str,
        reason: str,
        scope: FalsePositiveScope,
        codebase_hash: Optional[str] = None,
    ) -> FalsePositiveRecord:
        """
        Add a false positive record.

        Args:
            warning: The warning to mark as FP
            audit_id: ID of the audit
            reason: User's reason for marking as FP
            scope: Scope of the suppression
            codebase_hash: Hash of the codebase (for codebase-scoped FPs)

        Returns:
            The created FalsePositiveRecord
        """
        record = FalsePositiveRecord(
            id=str(uuid.uuid4()),
            audit_id=audit_id,
            warning_id=warning.id,
            warning_type=f"{warning.category.value}.{warning.pattern_type}",
            category=warning.category,
            code_hash=self._hash_code(warning.code_snippet),
            file_path=warning.file_path,
            line_start=warning.line_start,
            line_end=warning.line_end,
            user_reason=reason,
            applies_to=scope,
            codebase_hash=codebase_hash,
        )

        # Store based on scope
        if scope == FalsePositiveScope.THIS_PATTERN:
            # Global suppression
            data = self._load_json(self.global_file)
            data["false_positives"].append(record.model_dump())
            self._save_json(self.global_file, data)

        elif scope == FalsePositiveScope.THIS_CODEBASE and codebase_hash:
            # Codebase-specific suppression
            fp_file = self._get_codebase_file(codebase_hash)
            data = self._load_json(fp_file)
            data["false_positives"].append(record.model_dump())
            self._save_json(fp_file, data)

        # For THIS_AUDIT scope, we don't persist (handled in memory during audit)

        return record

    def is_false_positive(
        self,
        warning: CatastrophicWarning,
        audit_id: str,
        codebase_hash: Optional[str] = None,
    ) -> bool:
        """
        Check if a warning is a known false positive.

        Args:
            warning: The warning to check
            audit_id: Current audit ID
            codebase_hash: Hash of the codebase

        Returns:
            True if this is a known false positive
        """
        code_hash = self._hash_code(warning.code_snippet)
        warning_type = f"{warning.category.value}.{warning.pattern_type}"

        # Check global FPs
        global_data = self._load_json(self.global_file)
        for fp in global_data.get("false_positives", []):
            if fp.get("warning_type") == warning_type:
                # For pattern-level suppression, match by pattern type
                return True
            if fp.get("code_hash") == code_hash:
                return True

        # Check codebase-specific FPs
        if codebase_hash:
            cb_file = self._get_codebase_file(codebase_hash)
            cb_data = self._load_json(cb_file)
            for fp in cb_data.get("false_positives", []):
                if fp.get("code_hash") == code_hash:
                    return True
                if (fp.get("file_path") == warning.file_path and
                    fp.get("line_start") == warning.line_start and
                    fp.get("warning_type") == warning_type):
                    return True

        return False

    def get_suppression_reason(
        self,
        warning: CatastrophicWarning,
        codebase_hash: Optional[str] = None,
    ) -> Optional[str]:
        """Get the reason a warning was suppressed."""
        code_hash = self._hash_code(warning.code_snippet)
        warning_type = f"{warning.category.value}.{warning.pattern_type}"

        # Check global FPs
        global_data = self._load_json(self.global_file)
        for fp in global_data.get("false_positives", []):
            if fp.get("warning_type") == warning_type or fp.get("code_hash") == code_hash:
                return fp.get("user_reason")

        # Check codebase-specific FPs
        if codebase_hash:
            cb_file = self._get_codebase_file(codebase_hash)
            cb_data = self._load_json(cb_file)
            for fp in cb_data.get("false_positives", []):
                if fp.get("code_hash") == code_hash:
                    return fp.get("user_reason")

        return None

    def filter_warnings(
        self,
        warnings: list[CatastrophicWarning],
        audit_id: str,
        codebase_hash: Optional[str] = None,
    ) -> list[CatastrophicWarning]:
        """
        Filter out known false positives from warnings.

        Args:
            warnings: List of warnings to filter
            audit_id: Current audit ID
            codebase_hash: Hash of the codebase

        Returns:
            Filtered list with FPs removed (or marked as suppressed)
        """
        filtered = []
        for warning in warnings:
            if self.is_false_positive(warning, audit_id, codebase_hash):
                # Mark as suppressed but keep for transparency
                warning.suppressed = True
                warning.suppression_reason = self.get_suppression_reason(warning, codebase_hash)
            filtered.append(warning)

        return filtered

    def get_false_positives(
        self,
        codebase_hash: Optional[str] = None,
    ) -> list[FalsePositiveRecord]:
        """Get all false positive records."""
        records = []

        # Global FPs
        global_data = self._load_json(self.global_file)
        for fp in global_data.get("false_positives", []):
            try:
                records.append(FalsePositiveRecord(**fp))
            except Exception:
                pass

        # Codebase-specific FPs
        if codebase_hash:
            cb_file = self._get_codebase_file(codebase_hash)
            cb_data = self._load_json(cb_file)
            for fp in cb_data.get("false_positives", []):
                try:
                    records.append(FalsePositiveRecord(**fp))
                except Exception:
                    pass

        return records

    def remove_false_positive(self, fp_id: str) -> bool:
        """
        Remove a false positive record.

        Args:
            fp_id: ID of the false positive to remove

        Returns:
            True if removed, False if not found
        """
        # Check global file
        global_data = self._load_json(self.global_file)
        original_len = len(global_data.get("false_positives", []))
        global_data["false_positives"] = [
            fp for fp in global_data.get("false_positives", [])
            if fp.get("id") != fp_id
        ]
        if len(global_data["false_positives"]) < original_len:
            self._save_json(self.global_file, global_data)
            return True

        # Check codebase files
        for filename in os.listdir(self.by_codebase_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.by_codebase_dir, filename)
                data = self._load_json(filepath)
                original_len = len(data.get("false_positives", []))
                data["false_positives"] = [
                    fp for fp in data.get("false_positives", [])
                    if fp.get("id") != fp_id
                ]
                if len(data["false_positives"]) < original_len:
                    self._save_json(filepath, data)
                    return True

        return False
