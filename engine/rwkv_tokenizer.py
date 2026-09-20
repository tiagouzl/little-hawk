"""Tokenizer RWKV World (rwkv_vocab_v20230424) — espelho byte-level do
TRIE_TOKENIZER oficial (rwkv pip, Apache-2.0), sem torch.

Tokens são bytes; encode = longest-match sobre UTF-8; decode = concat +
utf-8 (� em caso de corte). Comportamento idêntico ao PIPELINE por
construção; teste compara diretamente contra ele.
"""
import ast


class RwkvTokenizer:
    def __init__(self, vocab_path):
        self.idx2token = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx = int(line[:line.index(" ")])
                tok = ast.literal_eval(line[line.index(" "):line.rindex(" ")])
                b = tok.encode("utf-8") if isinstance(tok, str) else bytes(tok)
                self.idx2token[idx] = b
        # longest-match: ordena por comprimento decrescente (empate: menor id)
        self._by_len = sorted(self.idx2token.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        self._tok2idx = {v: k for k, v in self.idx2token.items()}

    def __len__(self):
        return len(self.idx2token)

    def encode(self, text):
        src = text.replace("\r\n", "\n").encode("utf-8")
        ids = []
        i = 0
        while i < len(src):
            for idx, tok in self._by_len:
                if src.startswith(tok, i):
                    ids.append(idx)
                    i += len(tok)
                    break
        return ids

    def decode(self, ids):
        try:
            return b"".join(self.idx2token[i] for i in ids).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return "�"
