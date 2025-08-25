import logging
import sys

Formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"
)

logger = logging.getLogger("soil zoning")

if not logger.handlers:
    logger.setLevel(logging.INFO)  # Set logger level
    logger_handler = logging.StreamHandler(sys.stdout)
    logger_handler.setLevel(logging.INFO)  # Using logging.INFO instead of string "INFO"
    logger_handler.setFormatter(Formatter)
    logger.addHandler(logger_handler)