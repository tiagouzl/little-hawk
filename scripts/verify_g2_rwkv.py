"""G2: RWKV sem `n_layer and n_embd` e com eos distinguível."""
from engine.rwkv_engine import Rwkv7Engine

e = Rwkv7Engine(n_layer=12, n_embd=768)
assert e.n_embd == 768, f"n_embd={e.n_embd}"
# Edge que quebra hoje: n_layer=0 -> `0 and 32 == 0`
e0 = Rwkv7Engine(n_layer=0, n_embd=32)
assert e0.n_embd == 32, f"edge n_layer=0 quebrou: n_embd={e0.n_embd}"
assert e.bos_id != e.eos_id or e.eos_id is None, f"bos==eos=={e.bos_id}: EOS indistinguível"
print("G2 RWKV IDS OK")
