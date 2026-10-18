"""
Compila matmul.c, confere o resultado contra o NumPy e compara os tempos.

Rode:  python bench.py          (n=512 por padrao)
       python bench.py 1024
Precisa de gcc (no Linux Mint: sudo apt install build-essential).
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FONTE = os.path.join(AQUI, "matmul.c")
LIB = os.path.join(AQUI, "libmatmul.so")


def compilar():
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", FONTE, "-o", LIB]
    print("compilando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(LIB)


def cronometrar(fn, repeticoes=3):
    melhor = float("inf")
    for _ in range(repeticoes):
        t = time.perf_counter()
        fn()
        melhor = min(melhor, time.perf_counter() - t)
    return melhor


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    lib = compilar()
    fptr = ctypes.POINTER(ctypes.c_float)
    for nome in ("matmul_ingenua", "matmul_ikj"):
        getattr(lib, nome).argtypes = [fptr, fptr, fptr, ctypes.c_int]
    for nome in ("matmul_blocada", "matmul_paralela"):
        getattr(lib, nome).argtypes = [fptr, fptr, fptr, ctypes.c_int, ctypes.c_int]

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n), dtype=np.float32)
    B = rng.standard_normal((n, n), dtype=np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    ref = A @ B
    pa, pb, pc = (x.ctypes.data_as(fptr) for x in (A, B, C))

    versoes = [
        ("C v1 ingenua (i,j,k)", lambda: lib.matmul_ingenua(pa, pb, pc, n)),
        ("C v2 ordem (i,k,j)", lambda: lib.matmul_ikj(pa, pb, pc, n)),
        ("C v3 blocada (tiling)", lambda: lib.matmul_blocada(pa, pb, pc, n, 64)),
        ("C v4 blocada + threads", lambda: lib.matmul_paralela(pa, pb, pc, n, 64)),
        ("NumPy (BLAS)", lambda: np.matmul(A, B)),
    ]
    flops = 2.0 * n ** 3
    print(f"\nn = {n}   (cada multiplicacao faz {flops / 1e9:.2f} GFLOP)\n")
    print(f"{'versao':26s} {'tempo':>10s} {'GFLOPS':>9s} {'erro max':>10s}")
    for nome, fn in versoes:
        C[:] = 0
        t = cronometrar(fn)
        erro = float(np.abs(C - ref).max()) if "NumPy" not in nome else 0.0
        print(f"{nome:26s} {t * 1000:8.1f}ms {flops / t / 1e9:9.2f} {erro:10.2e}")
    print("\nO erro max pequeno (~1e-4) vem da ordem diferente das somas em float32.")
    print("Repare como o BLAS (usado pelo NumPy) ainda ganha: ele combina tiling, SIMD,")
    print("registradores e threads de forma muito mais agressiva. Em treino real, use-o (ou GPU).")


if __name__ == "__main__":
    main()
