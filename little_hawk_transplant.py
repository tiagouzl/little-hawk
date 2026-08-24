#!/usr/bin/env python3
"""
little_hawk_transplant.py — shim de compatibilidade.

A implementação vive em transplants/smollm.py. Este arquivo mantém o ponto
de entrada histórico funcionando:
    python little_hawk_transplant.py --layers 30
"""

import sys

from transplants.smollm import main

if __name__ == "__main__":
    sys.exit(main())
