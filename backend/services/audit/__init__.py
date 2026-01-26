"""Audit service for code-description alignment checking."""

from backend.services.audit.auditor import Auditor, get_auditor

__all__ = ["Auditor", "get_auditor"]
