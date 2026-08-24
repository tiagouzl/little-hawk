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
    """Factory: ONNX 30L validado 5 steps diff <1e-3, 1.45× vs NumPy (92 vs 134 ms)"""
    if HAS_ONNX and is_onnx_enabled():
        npz_path = kwargs.get("npz_path", "little_hawk_weights.npz")
        # Usa cache 512 com position freeze validado (engine/torch_model.py:53)
        return OnnxEngine(npz_path=npz_path)
    return MultiLayerEngine(*args, **kwargs)


__all__ = ["MultiLayerEngine", "OnnxEngine", "get_engine", "is_onnx_enabled"]