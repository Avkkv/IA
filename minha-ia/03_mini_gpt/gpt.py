"""
03 - Um GPT (transformer decoder-only) em NumPy, com backpropagation escrito a mao.

Nada de PyTorch: cada gradiente aqui foi derivado no papel (veja o README, secao de
matematica) e conferido numericamente pelo gradcheck.py. Mesma arquitetura, em
miniatura, dos modelos de linguagem modernos:

  tokens -> embedding + posicao -> [LayerNorm -> Atencao causal -> +residual
                                    LayerNorm -> MLP(GELU)      -> +residual] x L
         -> LayerNorm -> logits (pesos do embedding reutilizados) -> softmax

Convencao de formas:  B = lote, T = tamanho do contexto, D = dimensao do modelo,
H = numero de cabecas, V = tamanho do vocabulario.
"""
import math

import numpy as np


# ----------------------------------------------------------------------------
# Blocos elementares: cada um tem um "fwd" (guarda o que precisa) e um "bwd".
# ----------------------------------------------------------------------------
def lin_fwd(x, w, b):
    """y = x W + b"""
    return x @ w + b, (x, w)


def lin_bwd(dy, cache):
    """dx = dy W^T ;  dW = x^T dy ;  db = soma de dy sobre o lote."""
    x, w = cache
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    return dy @ w.T, x2.T @ dy2, dy2.sum(0)


def ln_fwd(x, g, b, eps=1e-5):
    """LayerNorm: normaliza cada vetor (ultimo eixo) e aplica escala g e deslocamento b."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def ln_bwd(dy, cache):
    xh, inv, g = cache
    eixos = tuple(range(dy.ndim - 1))
    dg = (dy * xh).sum(axis=eixos)
    db = dy.sum(axis=eixos)
    dxh = dy * g
    # dx = (1/sigma) * ( dxh - media(dxh) - xh * media(dxh * xh) )
    dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, dg, db


_C = math.sqrt(2.0 / math.pi)


def gelu_fwd(x):
    u = _C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_bwd(dy, cache):
    x, t = cache
    du = _C * (1.0 + 3 * 0.044715 * x ** 2)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du)


def attn_fwd(x, P, pre, H):
    """Auto-atencao causal multi-cabeca."""
    B, T, D = x.shape
    hd = D // H
    qkv, c1 = lin_fwd(x, P[pre + "qkv.w"], P[pre + "qkv.b"])
    q, k, v = np.split(qkv, 3, axis=-1)

    def cabecas(t):                                   # (B,T,D) -> (B,H,T,hd)
        return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

    q, k, v = cabecas(q), cabecas(k), cabecas(v)
    s = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)  # (B,H,T,T): quanto cada posicao olha para cada outra
    mascara = np.triu(np.full((T, T), -1e9, dtype=s.dtype), 1)  # ninguem enxerga o futuro
    s = s + mascara
    s = s - s.max(-1, keepdims=True)                   # estabilidade numerica
    e = np.exp(s)
    a = e / e.sum(-1, keepdims=True)                   # softmax
    o = a @ v                                          # media ponderada dos valores
    om = o.transpose(0, 2, 1, 3).reshape(B, T, D)
    y, c2 = lin_fwd(om, P[pre + "proj.w"], P[pre + "proj.b"])
    return y, (c1, c2, q, k, v, a, hd, B, T, D, H)


def attn_bwd(dy, cache, G, pre):
    c1, c2, q, k, v, a, hd, B, T, D, H = cache
    dom, dwp, dbp = lin_bwd(dy, c2)
    G[pre + "proj.w"] += dwp
    G[pre + "proj.b"] += dbp
    do = dom.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    da = do @ v.transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ do
    # Jacobiano do softmax aplicado: ds = a * (da - sum(da * a))
    ds = a * (da - (da * a).sum(-1, keepdims=True))
    ds = ds / math.sqrt(hd)
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q

    def juntar(t):
        return t.transpose(0, 2, 1, 3).reshape(B, T, D)

    dqkv = np.concatenate([juntar(dq), juntar(dk), juntar(dv)], axis=-1)
    dx, dw, db = lin_bwd(dqkv, c1)
    G[pre + "qkv.w"] += dw
    G[pre + "qkv.b"] += db
    return dx


# ----------------------------------------------------------------------------
# O modelo
# ----------------------------------------------------------------------------
class GPT:
    def __init__(self, vocab, bloco=64, dim=64, cabecas=4, camadas=2, seed=0, dtype=np.float32):
        assert dim % cabecas == 0
        self.cfg = dict(vocab=vocab, bloco=bloco, dim=dim, cabecas=cabecas, camadas=camadas)
        self.V, self.T, self.D, self.H, self.L = vocab, bloco, dim, cabecas, camadas
        rng = np.random.default_rng(seed)
        std = 0.02
        std_res = std / math.sqrt(2 * camadas)    # camadas que escrevem no fluxo residual: escala menor
        P = {}
        P["wte"] = rng.normal(0, std, (vocab, dim))
        P["wpe"] = rng.normal(0, std, (bloco, dim))
        for l in range(camadas):
            p = f"{l}."
            P[p + "ln1.g"] = np.ones(dim); P[p + "ln1.b"] = np.zeros(dim)
            P[p + "qkv.w"] = rng.normal(0, std, (dim, 3 * dim)); P[p + "qkv.b"] = np.zeros(3 * dim)
            P[p + "proj.w"] = rng.normal(0, std_res, (dim, dim)); P[p + "proj.b"] = np.zeros(dim)
            P[p + "ln2.g"] = np.ones(dim); P[p + "ln2.b"] = np.zeros(dim)
            P[p + "fc.w"] = rng.normal(0, std, (dim, 4 * dim)); P[p + "fc.b"] = np.zeros(4 * dim)
            P[p + "out.w"] = rng.normal(0, std_res, (4 * dim, dim)); P[p + "out.b"] = np.zeros(dim)
        P["lnf.g"] = np.ones(dim); P["lnf.b"] = np.zeros(dim)
        self.p = {k: v.astype(dtype) for k, v in P.items()}
        self._cache = None

    def n_parametros(self):
        return sum(v.size for v in self.p.values())

    # --- passada para frente ---------------------------------------------------
    def forward(self, idx, alvos=None):
        P, H = self.p, self.H
        B, T = idx.shape
        assert T <= self.T
        h = P["wte"][idx] + P["wpe"][:T]
        caches = []
        for l in range(self.L):
            p = f"{l}."
            n1, cl1 = ln_fwd(h, P[p + "ln1.g"], P[p + "ln1.b"])
            at, ca = attn_fwd(n1, P, p, H)
            h = h + at                                         # conexao residual
            n2, cl2 = ln_fwd(h, P[p + "ln2.g"], P[p + "ln2.b"])
            f1, cf1 = lin_fwd(n2, P[p + "fc.w"], P[p + "fc.b"])
            g, cg = gelu_fwd(f1)
            f2, cf2 = lin_fwd(g, P[p + "out.w"], P[p + "out.b"])
            h = h + f2                                         # conexao residual
            caches.append((cl1, ca, cl2, cf1, cg, cf2))
        nf, clf = ln_fwd(h, P["lnf.g"], P["lnf.b"])
        logits = nf @ P["wte"].T                               # pesos do embedding reutilizados
        if alvos is None:
            self._cache = None
            return logits, None
        z = logits - logits.max(-1, keepdims=True)
        lse = np.log(np.exp(z).sum(-1, keepdims=True))
        logp = z - lse
        probs = np.exp(logp)
        perda = -np.take_along_axis(logp, alvos[..., None], axis=-1).mean()   # entropia cruzada media
        self._cache = (idx, alvos, caches, nf, clf, probs)
        return logits, float(perda)

    # --- passada para tras -------------------------------------------------------
    def backward(self):
        P = self.p
        idx, alvos, caches, nf, clf, probs = self._cache
        B, T = idx.shape
        G = {k: np.zeros_like(v) for k, v in P.items()}

        # d(perda)/d(logits) = (softmax - onehot) / (B*T)
        dlog = probs.copy()
        np.put_along_axis(dlog, alvos[..., None], np.take_along_axis(dlog, alvos[..., None], -1) - 1.0, -1)
        dlog /= (B * T)

        dnf = dlog @ P["wte"]
        G["wte"] += dlog.reshape(-1, self.V).T @ nf.reshape(-1, self.D)
        dh, G["lnf.g"], G["lnf.b"] = ln_bwd(dnf, clf)

        for l in reversed(range(self.L)):
            p = f"{l}."
            cl1, ca, cl2, cf1, cg, cf2 = caches[l]
            # bloco MLP (a conexao residual deixa dh passar direto e tambem entrar no ramo)
            dg, dw, db = lin_bwd(dh, cf2)
            G[p + "out.w"] += dw; G[p + "out.b"] += db
            df1 = gelu_bwd(dg, cg)
            dn2, dw, db = lin_bwd(df1, cf1)
            G[p + "fc.w"] += dw; G[p + "fc.b"] += db
            dx, dgam, dbet = ln_bwd(dn2, cl2)
            G[p + "ln2.g"] += dgam; G[p + "ln2.b"] += dbet
            dh = dh + dx
            # bloco de atencao
            dn1 = attn_bwd(dh, ca, G, p)
            dx, dgam, dbet = ln_bwd(dn1, cl1)
            G[p + "ln1.g"] += dgam; G[p + "ln1.b"] += dbet
            dh = dh + dx

        np.add.at(G["wte"], idx, dh)          # o embedding recebe gradiente de duas fontes
        G["wpe"][:T] += dh.sum(0)
        return G

    # --- geracao -----------------------------------------------------------------
    def gerar(self, idx, n_novos, temperatura=0.8, top_k=40, rng=None):
        rng = rng or np.random.default_rng()
        idx = list(idx)
        for _ in range(n_novos):
            ctx = np.array(idx[-self.T:], dtype=np.int64)[None, :]
            logits, _ = self.forward(ctx)
            l = logits[0, -1].astype(np.float64) / max(temperatura, 1e-6)
            if top_k and top_k < l.size:
                corte = np.partition(l, -top_k)[-top_k]
                l = np.where(l < corte, -np.inf, l)
            pr = np.exp(l - l.max())
            pr /= pr.sum()
            idx.append(int(rng.choice(l.size, p=pr)))
        return idx

    # --- salvar / carregar ------------------------------------------------------
    def salvar(self, caminho, chars):
        np.savez(caminho, **{"p/" + k: v for k, v in self.p.items()},
                 cfg=np.array([self.cfg[k] for k in ("vocab", "bloco", "dim", "cabecas", "camadas")]),
                 chars=np.array(list(chars)))

    @staticmethod
    def carregar(caminho):
        d = np.load(caminho, allow_pickle=False)
        vocab, bloco, dim, cabecas, camadas = [int(x) for x in d["cfg"]]
        m = GPT(vocab, bloco, dim, cabecas, camadas)
        for k in m.p:
            m.p[k] = d["p/" + k].astype(m.p[k].dtype)
        return m, "".join(d["chars"].tolist())
