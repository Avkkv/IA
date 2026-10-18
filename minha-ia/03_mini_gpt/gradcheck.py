"""
Teste de gradiente: compara o backward() escrito a mao com derivadas numericas.

    dL/dtheta  ~=  ( L(theta + h) - L(theta - h) ) / (2h)

Se o erro relativo ficar abaixo de ~1e-5 em float64, a matematica do backprop esta certa.
Rode:  python gradcheck.py
"""
import sys

import numpy as np

from gpt import GPT


def main():
    rng = np.random.default_rng(0)
    m = GPT(vocab=11, bloco=6, dim=8, cabecas=2, camadas=2, seed=1, dtype=np.float64)
    # parametros maiores que o init real, para os gradientes nao ficarem minusculos
    for k, v in m.p.items():
        if v.ndim == 2:
            m.p[k] = v + rng.normal(0, 0.15, v.shape)
        elif k.endswith(".b"):
            m.p[k] = v + rng.normal(0, 0.1, v.shape)
        elif k.endswith(".g"):
            m.p[k] = v + rng.normal(0, 0.1, v.shape)

    idx = rng.integers(0, 11, (3, 6))
    alvos = rng.integers(0, 11, (3, 6))

    _, perda = m.forward(idx, alvos)
    G = m.backward()

    h = 1e-5
    pior = 0.0
    print(f"{'parametro':14s} {'analitico':>13s} {'numerico':>13s} {'erro rel':>10s}")
    for nome in m.p:
        w = m.p[nome]
        for rep in range(4):
            pos = tuple(int(rng.integers(0, s)) for s in w.shape)
            orig = w[pos]
            w[pos] = orig + h
            _, lp = m.forward(idx, alvos)
            w[pos] = orig - h
            _, lm = m.forward(idx, alvos)
            w[pos] = orig
            num = (lp - lm) / (2 * h)
            ana = G[nome][pos]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            pior = max(pior, rel)
            if rep == 0:
                print(f"{nome:14s} {ana:13.8f} {num:13.8f} {rel:10.2e}")
    print(f"\npior erro relativo: {pior:.2e}")
    if pior > 1e-5:
        print("FALHOU: algum gradiente nao confere")
        sys.exit(1)
    print("OK: o backprop escrito a mao bate com a derivada numerica")


if __name__ == "__main__":
    main()
