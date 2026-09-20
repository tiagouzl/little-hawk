"""G1: generate(speculative_k>0) deve retornar str, igual ao caminho normal."""
import inspect
from runtime.inference import LittleHawkInference

src = inspect.getsource(LittleHawkInference.generate)
# Falha honesta hoje: existe `return self._generate_speculative(` direto (tupla).
if "return self._generate_speculative(" in src:
    raise SystemExit("FAIL: generate retorna tupla do _generate_speculative sem desembrulhar")
# Após o fix, generate deve desembrulhar (texto, stats) ou _generate_speculative retornar str.
print("G1 SPECULATIVE RETURN OK")
