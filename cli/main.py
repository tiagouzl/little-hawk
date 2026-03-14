#!/usr/bin/env python3
"""
cli/main.py — CLI principal com subcomandos
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import BANNER, GREEN, RESET, YELLOW, RED
from runtime.tokenizer import BPETokenizer, CORPUS
from engine.engine import MultiLayerEngine
from runtime.inference import LittleHawkInference
from utils.config import DEFAULT_MODEL_CONFIG, DEFAULT_INFERENCE_CONFIG, load_config_from_env
from utils.helpers import validate_weights_file

def setup_argparse():
    """Configura parser principal com subcomandos"""
    parser = argparse.ArgumentParser(
        description="Little Hawk CLI v2 — multi-layer Atenção + MLP SwiGLU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modo demo
  python -m cli.main infer --prompt "hello world"

  # Com pesos
  python -m cli.main transplant --model smollm-135m --layers 4
  python -m cli.main infer --weights little_hawk_weights.npz --prompt "attention is all you need"

  # API
  python -m cli.main api --weights little_hawk_weights.npz
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')

    # Subcomando infer
    infer_parser = subparsers.add_parser('infer', help='Executa inferência')
    infer_parser.add_argument('--weights', type=str, help='Caminho para arquivo .npz')
    infer_parser.add_argument('--prompt', type=str, required=True, help='Texto de entrada')
    infer_parser.add_argument('--max-tokens', type=int, default=DEFAULT_INFERENCE_CONFIG['max_tokens'])
    infer_parser.add_argument('--temperature', type=float, default=DEFAULT_INFERENCE_CONFIG['temperature'])
    infer_parser.add_argument('--top-k', type=int, default=DEFAULT_INFERENCE_CONFIG['top_k'])
    infer_parser.add_argument('--top-p', type=float, default=DEFAULT_INFERENCE_CONFIG['top_p'])
    infer_parser.add_argument('--rep-penalty', type=float, default=DEFAULT_INFERENCE_CONFIG['rep_penalty'])
    infer_parser.add_argument('--no-panel', action='store_true', help='Sem painel de telemetria')

    # Subcomando transplant
    transplant_parser = subparsers.add_parser('transplant', help='Transplanta pesos de modelo HF')
    transplant_parser.add_argument('--model', type=str, required=True,
                                   choices=['smollm-135m', 'qwen2.5-0.5b'], help='Modelo a transplantar')
    transplant_parser.add_argument('--layers', type=int, default=4, help='Número de camadas')

    # Subcomando api
    api_parser = subparsers.add_parser('api', help='Inicia servidor API')
    api_parser.add_argument('--weights', type=str, help='Caminho para arquivo .npz')
    api_parser.add_argument('--host', type=str, default='0.0.0.0')
    api_parser.add_argument('--port', type=int, default=8000)

    return parser

def handle_infer(args):
    """Processa comando infer"""
    print(BANNER)

    tok, engine = build_tokenizer_and_engine(args.weights)
    hawk = LittleHawkInference(tokenizer=tok, engine=engine)

    hawk.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        stream=True,
        panel=not args.no_panel
    )

def handle_transplant(args):
    """Processa comando transplant"""
    print(BANNER)
    print(f"  {GREEN}Transplantando modelo: {args.model}{RESET}")

    if args.model == 'smollm-135m':
        from little_hawk_transplant import main as transplant_main
        # Simular argumentos
        import sys
        sys.argv = ['little_hawk_transplant.py', '--layers', str(args.layers)]
        transplant_main()
    elif args.model == 'qwen2.5-0.5b':
        from little_hawk_transplant_qwen import main as transplant_qwen_main
        sys.argv = ['little_hawk_transplant_qwen.py']
        transplant_qwen_main()

def handle_api(args):
    """Processa comando api"""
    print(BANNER)
    print(f"  {GREEN}Iniciando API em {args.host}:{args.port}{RESET}")

    # Configurar ambiente
    if args.weights:
        os.environ['LITTLE_HAWK_WEIGHTS'] = args.weights

    # Importar e executar API
    from api import app
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

def build_tokenizer_and_engine(weights_path):
    """Constrói tokenizer e engine baseado nos argumentos"""
    tok = BPETokenizer()

    if weights_path:
        valid, msg = validate_weights_file(weights_path)
        if not valid:
            print(f"  {RED}Erro: {msg}{RESET}")
            sys.exit(1)

        meta_path = weights_path.replace('.npz', '_meta.json')
        tok.load_donor_vocab(meta_path)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        engine = MultiLayerEngine(
            d_model=int(meta.get("d_model", DEFAULT_MODEL_CONFIG["d_model"])),
            n_heads=int(meta.get("n_heads", DEFAULT_MODEL_CONFIG["n_heads"])),
            n_layers=int(meta.get("n_layers", DEFAULT_MODEL_CONFIG["n_layers"])),
            sink_size=4, window_size=508,
            vocab_size=int(meta.get("vocab_size", DEFAULT_MODEL_CONFIG["vocab_size"]))
        )
        print(f"  {GREEN}✓ Carregando pesos...{RESET}")
        engine.load_weights(weights_path)
    else:
        print(f"  {YELLOW}Modo demo (pesos aleatórios){RESET}")
        tok.train(CORPUS, vocab_size=DEFAULT_MODEL_CONFIG["vocab_size"], verbose=True)
        engine = MultiLayerEngine(**DEFAULT_MODEL_CONFIG)

    return tok, engine

def main():
    """Função principal"""
    parser = setup_argparse()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Carregar configurações do ambiente
    env_config = load_config_from_env()
    if env_config:
        print(f"  {GREEN}Configurações do ambiente carregadas{RESET}")

    # Executar subcomando
    if args.command == 'infer':
        handle_infer(args)
    elif args.command == 'transplant':
        handle_transplant(args)
    elif args.command == 'api':
        handle_api(args)

if __name__ == '__main__':
    main()