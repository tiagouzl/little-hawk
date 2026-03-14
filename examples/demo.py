# Exemplo de uso do Little Hawk

from runtime.inference import LittleHawkInference, SamplingConfig, ConsoleTelemetry
from runtime.tokenizer import BPETokenizer
from engine.engine import MultiLayerEngine

prompt = "atenção e memória são os fundamentos"
tok = BPETokenizer(); tok.train(prompt, vocab_size=128, verbose=False)
engine = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=128)
hawk = LittleHawkInference(tokenizer=tok, engine=engine)
cfg = SamplingConfig(max_tokens=20)
print("Prompt:", prompt)
print("Saída:")
hawk.generate(prompt, sampling_config=cfg, telemetry=ConsoleTelemetry())
