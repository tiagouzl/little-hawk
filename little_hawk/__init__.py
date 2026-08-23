"""Little Hawk — motor de inferência LLM streaming em NumPy puro.

Módulos:
    tokenizer  → BPE (demo + doador) e StreamDecoder byte-safe
    engine     → LlamaLayer + MultiLayerEngine com StreamingKVCache O(1)
    inference  → loop autoregressivo, amostragem e telemetria
"""

from ._ansi import _hdr
from .engine import LlamaLayer, MultiLayerEngine
from .inference import LittleHawkInference
from .tokenizer import BPETokenizer, CORPUS, StreamDecoder

__version__ = "0.3.0"

__all__ = [
    "CORPUS",
    "BPETokenizer",
    "LittleHawkInference",
    "LlamaLayer",
    "MultiLayerEngine",
    "StreamDecoder",
    "__version__",
    "_hdr",
]
