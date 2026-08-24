#!/usr/bin/env python3
"""
little_hawk_transplant_qwen.py — shim de compatibilidade.

A implementação vive em transplants/qwen.py. Este arquivo mantém o ponto
de entrada histórico funcionando:
    python little_hawk_transplant_qwen.py [--layers 24]
"""

import sys

from transplants.qwen import main

if __name__ == "__main__":
    sys.exit(main())
