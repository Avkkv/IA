"""
Treina o mini-GPT em NumPy sobre o texto/codigo que voce escolher.

Exemplos:
    python train.py --dados ~/meus_projetos --passos 3000
    python train.py --dados ../01_autograd.py --passos 300 --dim 48 --camadas 2

Aprendeu com o seu proprio codigo? Gere amostras com:  python sample.py --prompt "def "
"""
import argparse
import math
import time

import numpy as np

from dados import carregar_texto, tokenizar
from gpt import GPT


class Adam:
    """Adam com decaimento de peso desacoplado (AdamW). Equacoes no README."""

    def __init__(self, params, b1=0.9, b2=0.95, eps=1e-8, wd=0.01):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2, self.eps, self.wd, self.t = b1, b2, eps, wd, 0

    def passo(self, params, grads, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, p in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            update = (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if p.ndim >= 2:                      # decaimento so nas matrizes
                p -= lr * self.wd * p
            p -= lr * update


def lr_em(passo, total, lr_max, aquecimento):
    """Aquecimento linear, depois cosseno ate 10% do maximo."""
    if passo < aquecimento:
        return lr_max * (passo + 1) / aquecimento
    prog = (passo - aquecimento) / max(1, total - aquecimento)
    return lr_max * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def lote(ids, B, T, rng):
    ini = rng.integers(0, len(ids) - T - 1, B)
    x = np.stack([ids[i:i + T] for i in ini])
    y = np.stack([ids[i + 1:i + T + 1] for i in ini])      # alvo = proximo caractere
    return x, y


def avaliar(model, ids, B, T, rng, n=5):
    perdas = []
    for _ in range(n):
        x, y = lote(ids, B, T, rng)
        perdas.append(model.forward(x, y)[1])
    return float(np.mean(perdas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True, help="arquivo ou pasta de codigo/texto")
    ap.add_argument("--passos", type=int, default=2000)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--bloco", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cabecas", type=int, default=4)
    ap.add_argument("--camadas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--saida", default="modelo.npz")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--avaliar-a-cada", type=int, default=100)
    a = ap.parse_args()

    texto = carregar_texto(a.dados)
    chars, ids = tokenizar(texto)
    corte = int(len(ids) * 0.9)
    treino, val = ids[:corte], ids[corte:]
    print(f"texto: {len(texto):,} caracteres | vocabulario: {len(chars)} | treino {len(treino):,} / validacao {len(val):,}")
    if len(val) < a.bloco + 2:
        raise SystemExit("dados demais curtos; use mais texto ou um --bloco menor")

    model = GPT(len(chars), a.bloco, a.dim, a.cabecas, a.camadas, seed=a.semente)
    print(f"parametros: {model.n_parametros():,}")
    print(f"perda inicial esperada ~ ln(V) = {math.log(len(chars)):.3f}\n")
    opt = Adam(model.p)
    rng = np.random.default_rng(a.semente)
    aquec = max(1, a.passos // 20)
    t0 = time.perf_counter()
    for passo in range(a.passos):
        x, y = lote(treino, a.lote, a.bloco, rng)
        _, perda = model.forward(x, y)
        G = model.backward()
        norma = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
        if norma > 1.0:                          # recorte de gradiente: evita explosoes
            for g in G.values():
                g *= 1.0 / norma
        opt.passo(model.p, G, lr_em(passo, a.passos, a.lr, aquec))
        if passo % a.avaliar_a_cada == 0 or passo == a.passos - 1:
            pv = avaliar(model, val, a.lote, a.bloco, rng)
            dt = time.perf_counter() - t0
            tps = (passo + 1) * a.lote * a.bloco / dt
            print(f"passo {passo:5d} | treino {perda:.3f} | validacao {pv:.3f} (ppl {math.exp(pv):6.2f}) | {tps:,.0f} tokens/s")
    model.salvar(a.saida, chars)
    print(f"\nmodelo salvo em {a.saida}")


if __name__ == "__main__":
    main()
