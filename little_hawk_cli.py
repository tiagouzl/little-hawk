#!/usr/bin/env python3
"""
Little Hawk CLI - Wrapper for the modular CLI.

This script maintains backward compatibility by delegating to the new modular CLI.
"""

import sys
from cli.main import main

if __name__ == "__main__":
    sys.exit(main())
