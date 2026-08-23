"""Cores ANSI compartilhadas pela UI do Little Hawk."""

import os

try:
    import colorama

    colorama.init()
except ImportError:
    pass


def _c(code):
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


def _hdr(title):
    w = 60
    print(f"\n{DIM}╔{'═' * w}╗{RESET}\n{DIM}║{RESET}{BOLD}{title:^{w}}{RESET}{DIM}║{RESET}\n{DIM}╚{'═' * w}╝{RESET}\n")


__all__ = [
    "BOLD",
    "CYAN",
    "DIM",
    "GREEN",
    "MAGENTA",
    "RED",
    "RESET",
    "WHITE",
    "YELLOW",
    "_c",
    "_hdr",
]
