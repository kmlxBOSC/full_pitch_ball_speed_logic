import logging
import os
from logging.handlers import RotatingFileHandler
from ball_tracking.core.config import (
    ROOT_LOGGER_NAME, LOG_DIR, LOG_FILE,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    LOG_FILE_FMT, LOG_CONSOLE_FMT, LOG_DATE_FMT,
)

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_CONSOLE_FMT, datefmt=LOG_DATE_FMT))
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FILE_FMT, datefmt=LOG_DATE_FMT))
    root.addHandler(file_handler)

    _configured = True

def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
