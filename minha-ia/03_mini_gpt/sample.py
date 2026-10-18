"""
Gera texto com o modelo treinado.

    python sample.py --prompt "def " --tokens 400 --temperatura 0.8
Temperatura baixa = previsivel e repetitivo; alta = criativo e errático.
"""
import argparse

import numpy as np

from gpt import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelo.npz")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperatura", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--semente", type=int, default=None)
    a = ap.parse_args()

    model, chars = GPT.carregar(a.modelo)
    tabela = {c: i for i, c in enumerate(chars)}
    prompt = [tabela[c] for c in a.prompt if c in tabela] or [0]
    rng = np.random.default_rng(a.semente)
    saida = model.gerar(prompt, a.tokens, a.temperatura, a.top_k, rng)
    print("".join(chars[i] for i in saida))


if __name__ == "__main__":
    main()
