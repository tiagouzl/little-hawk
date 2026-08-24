# engine/__init__.py
import os

from .engine import MultiLayerEngine

try:
    from .onnx_engine import OnnxEngine, is_onnx_enabled

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

    def is_onnx_enabled():
        return False


def get_engine(*args, **kwargs):
    """Factory: ONNX é POC de bench (6.6×) — inferência ainda usa NumPy para
    garantir StreamingKVCache O(1) + position freeze corretos.
    Use `scripts/onnx_export.py --bench` para medir teto ONNX."""
    # TODO v0.6: ativar OnnxEngine quando cache linear vs circular for validado
    # if HAS_ONNX and is_onnx_enabled():
    #     npz_path = kwargs.get("npz_path", "little_hawk_weights.npz")
    #     return OnnxEngine(npz_path=npz_path)
    return MultiLayerEngine(*args, **kwargs)


__all__ = ["MultiLayerEngine", "OnnxEngine", "get_engine", "is_onnx_enabled"]