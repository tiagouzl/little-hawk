"""Compatibilidade: little_hawk_cli re-exporta a API do pacote little_hawk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_reexports_match_package():
    import little_hawk_cli as cli
    import little_hawk as pkg

    for name in ("BPETokenizer", "CORPUS", "LittleHawkInference", "MultiLayerEngine", "StreamDecoder"):
        assert getattr(cli, name) is getattr(pkg, name), f"re-export divergente: {name}"
