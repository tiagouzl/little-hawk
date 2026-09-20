"""G4: chat default demo + no-panel respeita speculative + sem sys.argv hack."""
import inspect
import cli.main as m

p = m.setup_argparse()
# encontra default do chat --weights
chat_weights_default = None
for a in p._actions:
    if a.dest == "command":
        for name, sub in a.choices.items():
            if name == "chat":
                for act in sub._actions:
                    if act.dest == "weights":
                        chat_weights_default = act.default
assert chat_weights_default is None, f"chat --weights default={chat_weights_default!r}, esperado None (demo)"

src_infer = inspect.getsource(m.handle_infer)
assert "speculative" in src_infer, "handle_infer no-panel ignora --speculative"
src_chat = inspect.getsource(m.handle_chat)
assert "speculative" in src_chat or "speculative_k" in src_chat, "handle_chat ignora --speculative"

src_all = inspect.getsource(m)
assert "sys.argv =" not in src_all and "sys.argv=" not in src_all, "transplant ainda muta sys.argv"
print("G4 CLI OK")
