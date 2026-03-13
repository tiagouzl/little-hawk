#!/usr/bin/env python3
"""Utilidades compartilhadas (cores ANSI)."""

def _c(code: int) -> str:
    return f"\033[{code}m"

RESET = _c(0)
BOLD = _c(1)
DIM = _c(2)
CYAN = _c(36)
GREEN = _c(32)
YELLOW = _c(33)
MAGENTA = _c(35)
RED = _c(31)
WHITE = _c(97)

__all__ = [
    "RESET",
    "BOLD",
    "DIM",
    "CYAN",
    "GREEN",
    "YELLOW",
    "MAGENTA",
    "RED",
    "WHITE",
]
