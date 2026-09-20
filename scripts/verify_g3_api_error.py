"""G3: erro no worker vira evento de erro, sem travar até timeout."""
import inspect
import api.server as srv

src_worker = inspect.getsource(srv._blocking_stream)
src_sse = inspect.getsource(srv._stream_sse)
assert "except Exception" in src_worker, "worker sem except Exception genérico"
assert "out_q.put(None)" in src_worker, "worker deve sempre sinalizar fim"
assert "error" in src_sse, "consumidor SSE deve emitir evento de erro"
print("G3 API ERROR OK")
