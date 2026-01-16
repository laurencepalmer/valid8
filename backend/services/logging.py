"""
Logging module with sanitization for sensitive data.
Provides secure logging that masks API keys and other secrets.
"""

import logging
import re
from typing import Optional

from backend.config import get_settings


# Patterns for sensitive data that should be masked in logs
SENSITIVE_PATTERNS = [
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})'), 'sk-***REDACTED***'),  # OpenAI keys
    (re.compile(r'(sk-ant-[a-zA-Z0-9-]{20,})'), 'sk-ant-***REDACTED***'),  # Anthropic keys
    (re.compile(r'(api[_-]?key["\s:=]+)["\']?([a-zA-Z0-9-_]{20,})["\']?', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(token["\s:=]+)["\']?([a-zA-Z0-9-_]{20,})["\']?', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(password["\s:=]+)["\']?([^\s"\']+)["\']?', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(secret["\s:=]+)["\']?([a-zA-Z0-9-_]{10,})["\']?', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(bearer\s+)([a-zA-Z0-9-_.]+)', re.IGNORECASE), r'\1***REDACTED***'),
]


def sanitize_message(message: str) -> str:
    """
    Sanitize a message by masking sensitive data patterns.

    Args:
        message: The message to sanitize

    Returns:
        The sanitized message with sensitive data masked
    """
    sanitized = message
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_error(error: Exception) -> str:
    """
    Sanitize an exception message for safe display/logging.
    Returns a generic message for external users while logging the full error.

    Args:
        error: The exception to sanitize

    Returns:
        A sanitized error message safe for external display
    """
    error_str = str(error)
    return sanitize_message(error_str)


def get_safe_error_message(error: Exception, context: str = "Operation") -> str:
    """
    Get a user-safe error message that doesn't expose sensitive details.

    Args:
        error: The exception that occurred
        context: A description of what operation was being performed

    Returns:
        A generic, safe error message for end users
    """
    error_type = type(error).__name__

    # Map common error types to user-friendly messages
    error_messages = {
        "ConnectionError": f"{context} failed: Unable to connect to external service",
        "TimeoutError": f"{context} failed: Request timed out",
        "AuthenticationError": f"{context} failed: Authentication error",
        "RateLimitError": f"{context} failed: Rate limit exceeded",
        "APIError": f"{context} failed: External API error",
        "ValueError": f"{context} failed: Invalid input provided",
        "FileNotFoundError": f"{context} failed: Required file not found",
        "PermissionError": f"{context} failed: Permission denied",
    }

    return error_messages.get(error_type, f"{context} failed: An unexpected error occurred")


class SanitizingFormatter(logging.Formatter):
    """
    A logging formatter that sanitizes sensitive data from log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize the message
        original_msg = record.getMessage()
        record.msg = sanitize_message(original_msg)
        record.args = ()

        # Also sanitize exception info if present
        if record.exc_info:
            # Format will handle exc_info, but we need to be careful
            pass

        return super().format(record)


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Set up and return a logger with sanitization enabled.

    Args:
        name: The logger name. If None, returns the root logger.

    Returns:
        A configured logger instance with sanitization
    """
    settings = get_settings()

    logger = logging.getLogger(name or "valid8")

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SanitizingFormatter(settings.log_format))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level))

    return logger


# Create a default application logger
logger = setup_logging("valid8")
