"""
01 - Autograd escalar do zero (estilo micrograd).

Objetivo: entender de verdade o que e o backpropagation.
Cada operacao guarda como propagar o gradiente para tras (regra da cadeia).
Nenhuma biblioteca: so Python puro.

Rode:  python 01_autograd.py
"""
import math
import random


class Valor:
    """Um numero que lembra de onde veio, para podermos derivar tudo por tras."""

    __slots__ = ("dado", "grad", "_filhos", "_voltar")

    def __init__(self, dado, filhos=()):
        self.dado = float(dado)
        self.grad = 0.0          # dL/d(este valor), preenchido no backward
        self._filhos = filhos
        self._voltar = lambda: None

    # --- operacoes: cada uma define a derivada local -------------------------
    def __add__(self, o):
        o = o if isinstance(o, Valor) else Valor(o)
        saida = Valor(self.dado + o.dado, (self, o))

        def voltar():                       # d(a+b)/da = 1 ; d(a+b)/db = 1
            self.grad += saida.grad
            o.grad += saida.grad
        saida._voltar = voltar
        return saida

    def __mul__(self, o):
        o = o if isinstance(o, Valor) else Valor(o)
        saida = Valor(self.dado * o.dado, (self, o))

        def voltar():                       # d(a*b)/da = b ; d(a*b)/db = a
            self.grad += o.dado * saida.grad
            o.grad += self.dado * saida.grad
        saida._voltar = voltar
        return saida

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        saida = Valor(self.dado ** k, (self,))

        def voltar():                       # d(a^k)/da = k a^(k-1)
            self.grad += k * (self.dado ** (k - 1)) * saida.grad
        saida._voltar = voltar
        return saida

    def tanh(self):
        t = math.tanh(self.dado)
        saida = Valor(t, (self,))

        def voltar():                       # d tanh(a)/da = 1 - tanh^2(a)
            self.grad += (1 - t * t) * saida.grad
        saida._voltar = voltar
        return saida

    def exp(self):
        e = math.exp(self.dado)
        saida = Valor(e, (self,))

        def voltar():                       # d e^a/da = e^a
            self.grad += e * saida.grad
        saida._voltar = voltar
        return saida

    def log(self):
        saida = Valor(math.log(self.dado), (self,))

        def voltar():                       # d ln(a)/da = 1/a
            self.grad += (1.0 / self.dado) * saida.grad
        saida._voltar = voltar
        return saida

    def relu(self):
        saida = Valor(max(0.0, self.dado), (self,))

        def voltar():
            self.grad += (1.0 if self.dado > 0 else 0.0) * saida.grad
        saida._voltar = voltar
        return saida

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o if isinstance(o, Valor) else -o)
    def __rsub__(self, o): return (-self) + o
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Valor) else 1.0 / o)
    def __rtruediv__(self, o): return (self ** -1) * o
    def __repr__(self): return f"Valor(dado={self.dado:.4f}, grad={self.grad:.4f})"

    def backward(self):
        """Ordena o grafo topologicamente e aplica a regra da cadeia de tras pra frente."""
        ordem, visto = [], set()
        pilha = [(self, False)]
        while pilha:                         # DFS iterativa (evita estouro de recursao)
            no, expandido = pilha.pop()
            if expandido:
                ordem.append(no)
                continue
            if id(no) in visto:
                continue
            visto.add(id(no))
            pilha.append((no, True))
            for f in no._filhos:
                pilha.append((f, False))
        self.grad = 1.0                      # dL/dL = 1
        for no in reversed(ordem):
            no._voltar()


# --- uma rede neural pequena feita com Valor ---------------------------------
class Neuronio:
    def __init__(self, n_entradas, linear=False):
        self.w = [Valor(random.uniform(-1, 1)) for _ in range(n_entradas)]
        self.b = Valor(0.0)
        self.linear = linear

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # z = w.x + b
        return z if self.linear else z.tanh()

    def parametros(self):
        return self.w + [self.b]


class Camada:
    def __init__(self, n_in, n_out, **kw):
        self.neuronios = [Neuronio(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        saidas = [n(x) for n in self.neuronios]
        return saidas[0] if len(saidas) == 1 else saidas

    def parametros(self):
        return [p for n in self.neuronios for p in n.parametros()]


class MLP:
    def __init__(self, n_in, tamanhos):
        dims = [n_in] + tamanhos
        self.camadas = [Camada(dims[i], dims[i + 1], linear=(i == len(tamanhos) - 1))
                        for i in range(len(tamanhos))]

    def __call__(self, x):
        for c in self.camadas:
            x = c(x)
        return x

    def parametros(self):
        return [p for c in self.camadas for p in c.parametros()]


def verificar_gradiente():
    """Compara o gradiente do autograd com a derivada numerica (diferencas centrais)."""
    def f(a, b):
        return ((a * b + b ** 3).tanh() + a.exp() * 0.1 - (b + 2.0).log()) / (a * a + 1.5)

    a0, b0, h = 0.7, 1.3, 1e-6
    a, b = Valor(a0), Valor(b0)
    f(a, b).backward()
    num_a = (f(Valor(a0 + h), Valor(b0)).dado - f(Valor(a0 - h), Valor(b0)).dado) / (2 * h)
    num_b = (f(Valor(a0), Valor(b0 + h)).dado - f(Valor(a0), Valor(b0 - h)).dado) / (2 * h)
    print(f"dL/da  autograd={a.grad:+.8f}  numerico={num_a:+.8f}")
    print(f"dL/db  autograd={b.grad:+.8f}  numerico={num_b:+.8f}")
    assert abs(a.grad - num_a) < 1e-6 and abs(b.grad - num_b) < 1e-6
    print("gradientes conferem\n")


def demo_treino():
    """Ensina uma MLP a resolver XOR, o problema que um neuronio unico nao resolve."""
    random.seed(1)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [-1, 1, 1, -1]
    rede = MLP(2, [6, 6, 1])
    lr = 0.1
    for passo in range(301):
        previsoes = [rede(x) for x in X]
        perda = sum((p - t) ** 2 for p, t in zip(previsoes, y)) * (1 / len(X))   # MSE
        for p in rede.parametros():
            p.grad = 0.0
        perda.backward()
        for p in rede.parametros():
            p.dado -= lr * p.grad                  # descida do gradiente
        if passo % 50 == 0:
            print(f"passo {passo:3d}  perda {perda.dado:.5f}")
    print("\nentrada -> previsao (alvo)")
    for x, t in zip(X, y):
        print(f"{x} -> {rede(x).dado:+.3f} ({t:+d})")


if __name__ == "__main__":
    verificar_gradiente()
    demo_treino()
