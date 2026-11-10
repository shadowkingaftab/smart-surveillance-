"""Shared logging setup for edge and backend components."""

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
