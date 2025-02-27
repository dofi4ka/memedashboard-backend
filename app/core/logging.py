import logging
import os

from app.core.envloader import Environment


def configure_logging():
    logging_level = logging.getLevelName(Environment.LOG_LEVEL)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=logging_level, format=log_format, datefmt=date_format)

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    os.makedirs("logs", exist_ok=True)

    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
