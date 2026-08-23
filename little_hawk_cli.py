#!/usr/bin/env python3
"""
Little Hawk CLI - Wrapper for the modular CLI.

This script maintains backward compatibility by delegating to the new modular CLI.
"""

import sys

from cli.main import main


def _legacy_flags(argv):
    """Traduz invocação legada por flags (v2) para o subcomando `infer`.

    Ex.: little_hawk_cli.py --weights x.npz --prompt "..."  →  infer --weights ...
    """
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        return ["infer"] + argv
    return argv


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + _legacy_flags(sys.argv[1:])
    sys.exit(main())
