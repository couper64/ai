"""Logging configuration for the image classifier pipeline service."""

import logging
import os
import sys


class ColorFormatter(logging.Formatter):
    """Apply ANSI colors to log level names for terminal readability."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    LEVEL_COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colored timestamp and level name."""
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelname, "")
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return f"{self.DIM}{formatted[:23]}{self.RESET}{formatted[23:]}"


def should_color_logs(stream) -> bool:  # type: ignore[no-untyped-def]
    """Return whether logs should use ANSI colors."""
    color_setting = os.environ.get("LOG_COLOR", "auto").lower()
    if color_setting in {"1", "true", "yes", "on"}:
        return True
    if color_setting in {"0", "false", "no", "off"}:
        return False
    return stream.isatty()


def configure_logging() -> None:
    """Configure application-wide logging from the LOG_LEVEL environment variable."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    stream = sys.stderr
    formatter_class = ColorFormatter if should_color_logs(stream) else logging.Formatter
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter_class("%(asctime)s %(levelname)s %(name)s %(message)s"))

    package_logger = logging.getLogger("image_classifier_pipeline")
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
