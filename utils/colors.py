"""
utils/colors.py — Constantes de cores para output
"""


def _c(code):
    return f"\033[{code}m"


RESET = _c(0)
BOLD = _c(1)
DIM = _c(2)
CYAN = _c(36)
GREEN = _c(32)
YELLOW = _c(33)
MAGENTA = _c(35)
RED = _c(31)
WHITE = _c(97)


def ok(s):
    return f"  {GREEN}✓{RESET} {s}"


def err(s):
    return f"  {RED}✗{RESET} {s}"


def inf(s):
    return f"  {CYAN}·{RESET} {s}"


def warn(s):
    return f"  {YELLOW}⚠{RESET} {s}"


def hdr(s):
    print(f"\n{BOLD}{s}{RESET}\n{DIM}{'─' * 58}{RESET}")


# Banner ASCII
BANNER = """
  ·  ʟɪᴛᴛʟᴇ  ·
  ██╗  ██╗ █████╗ ██╗    ██╗██╗  ██╗
  ██║  ██║██╔══██╗██║    ██║██║ ██╔╝
  ███████║███████║██║ █╗ ██║█████╔╝
  ██╔══██║██╔══██║██║███╗██║██╔═██╗
  ██║  ██║██║  ██║╚███╔███╔╝██║  ██╗
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝
  H A W K  v2 — Atenção + MLP SwiGLU Empilhados
  StreamingKVCache O(1) · RoPE · multi-layer
"""
