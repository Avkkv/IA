"""Carrega texto (um arquivo ou uma pasta inteira de codigo) e transforma em ids de caracteres."""
import os

import numpy as np

EXTENSOES = {".py", ".c", ".h", ".cpp", ".hpp", ".js", ".ts", ".java", ".go", ".rs",
             ".sh", ".md", ".txt", ".sql", ".html", ".css"}
IGNORAR = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}


def carregar_texto(caminho, max_bytes=5_000_000):
    """Se for pasta, junta todos os arquivos de codigo; se for arquivo, le o arquivo."""
    if os.path.isfile(caminho):
        arquivos = [caminho]
    else:
        arquivos = []
        for raiz, pastas, nomes in os.walk(caminho):
            pastas[:] = sorted(p for p in pastas if p not in IGNORAR and not p.startswith("."))
            for n in sorted(nomes):
                if os.path.splitext(n)[1].lower() in EXTENSOES:
                    arquivos.append(os.path.join(raiz, n))
    partes, total = [], 0
    for a in arquivos:
        try:
            with open(a, "r", encoding="utf-8") as f:
                t = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        partes.append(t)
        total += len(t)
        if total >= max_bytes:
            break
    if not partes:
        raise SystemExit(f"nenhum arquivo de texto/codigo encontrado em {caminho!r}")
    return "\n\n".join(partes)[:max_bytes]


def tokenizar(texto, chars=None):
    """Tokenizacao por caractere: simples e boa para estudar. (LLMs reais usam BPE.)"""
    chars = chars or "".join(sorted(set(texto)))
    tabela = {c: i for i, c in enumerate(chars)}
    ids = np.array([tabela[c] for c in texto if c in tabela], dtype=np.int64)
    return chars, ids
