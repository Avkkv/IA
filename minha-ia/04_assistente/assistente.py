#!/usr/bin/env python3
"""
04 - Assistente local de estudo de codigo (o embriao do seu "Jarvis").

Pecas:
  1. Indice BM25 sobre os seus arquivos (RAG: Retrieval-Augmented Generation).
  2. Um LLM local rodando no Ollama (voce troca o modelo quando quiser).
  3. Memoria: toda pergunta/resposta vai para memoria.jsonl; voce marca o que
     foi bom com "+" e isso vira dataset de fine-tuning (--exportar-sft).

So usa a biblioteca padrao do Python.

Uso:
  python assistente.py --dir ~/meu_projeto                    # conversa interativa
  python assistente.py --dir . "o que faz a funcao ordenar?"  # uma pergunta
  python assistente.py --dir . --sem-llm "lista encadeada"    # so a busca, sem modelo
  python assistente.py --exportar-sft sft.jsonl               # dataset das respostas aprovadas

Modelo: variavel MODELO_OLLAMA ou --modelo (padrao qwen2.5-coder:7b).
Instalar o Ollama: https://ollama.com  ->  ollama pull qwen2.5-coder:7b
"""
import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter

EXTENSOES = {".py", ".c", ".h", ".cpp", ".hpp", ".js", ".ts", ".java", ".go", ".rs",
             ".sh", ".md", ".txt", ".sql", ".html", ".css", ".json", ".toml", ".yaml", ".yml"}
IGNORAR = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}
PARADAS = set(
    "a o e de da do das dos em um uma que para com por os as no na nos nas se ao ou mas "
    "como funciona funcionam faz fazer qual quais quando onde porque pq oque isso isto esta este "
    "ele ela eu voce me meu minha sao ser e foi the of to in is and for it how does what why "
    "explique explica explicar mostre".split()
)

SISTEMA = (
    "Voce e um assistente de estudos de Ciencia da Computacao. Responda em portugues, "
    "de forma didatica e precisa. Use os trechos de codigo fornecidos como fonte principal "
    "e cite sempre arquivo:linhas ao se referir a eles. Se os trechos nao bastarem, diga o "
    "que falta em vez de inventar."
)


def tokens(texto):
    """Quebra identificadores (snake_case e camelCase) e mantem o identificador inteiro."""
    saida = []
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")   # tira acentos: atenção == atencao
    for palavra in re.findall(r"[A-Za-z0-9_]+", texto):
        baixo = palavra.lower()
        partes = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", palavra).replace("_", " ").lower().split()
        if len(partes) > 1:
            saida.append(baixo)
        saida.extend(partes)
    return [t for t in saida if t not in PARADAS and len(t) > 1]


class Indice:
    def __init__(self, pasta, linhas=40, sobreposicao=10):
        self.pasta = pasta
        self.chunks = []                      # (caminho, linha_ini, linha_fim, texto)
        for raiz, pastas, nomes in os.walk(pasta):
            pastas[:] = sorted(p for p in pastas if p not in IGNORAR and not p.startswith("."))
            for nome in sorted(nomes):
                if os.path.splitext(nome)[1].lower() not in EXTENSOES:
                    continue
                caminho = os.path.join(raiz, nome)
                try:
                    with open(caminho, "r", encoding="utf-8") as f:
                        L = f.read().splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                passo = max(1, linhas - sobreposicao)
                for ini in range(0, max(1, len(L)), passo):
                    if ini > 0 and len(L) - ini <= sobreposicao:
                        break                  # sobra so o que ja estava no trecho anterior
                    trecho = "\n".join(L[ini:ini + linhas])
                    if trecho.strip():
                        rel = os.path.relpath(caminho, pasta)
                        self.chunks.append((rel, ini + 1, min(ini + linhas, len(L)), trecho))
        self.tf = [Counter(tokens(c[0] + " " + c[3])) for c in self.chunks]   # o caminho tambem conta
        self.comp = [sum(t.values()) for t in self.tf]
        self.media = (sum(self.comp) / len(self.comp)) if self.comp else 1.0
        self.df = Counter()
        for t in self.tf:
            self.df.update(t.keys())

    def buscar(self, consulta, k=5, k1=1.5, b=0.75):
        """BM25: soma, para cada termo da consulta, idf * saturacao(tf) com normalizacao de tamanho."""
        N = len(self.chunks)
        termos = tokens(consulta)
        pontos = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in termos:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.comp[i] / self.media))
            pontos.append((s, i))
        pontos.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in pontos[:k] if s > 0]


def montar_contexto(achados):
    partes = []
    for _, (caminho, a, b, texto) in achados:
        partes.append(f"### {caminho}:{a}-{b}\n```\n{texto}\n```")
    return "\n\n".join(partes)


def chamar_llm(mensagens, modelo, url):
    corpo = json.dumps({"model": modelo, "messages": mensagens, "stream": False,
                        "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=corpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


class Memoria:
    def __init__(self, caminho):
        self.caminho = caminho

    def _anexar(self, obj):
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def registrar(self, pergunta, contexto, resposta):
        ts = time.time()
        self._anexar({"tipo": "troca", "ts": ts, "pergunta": pergunta, "contexto": contexto, "resposta": resposta})
        return ts

    def avaliar(self, ts, nota):
        self._anexar({"tipo": "nota", "ref": ts, "nota": nota})

    def exportar_sft(self, saida):
        """Pares pergunta/resposta aprovados com '+', no formato de chat usado por ferramentas de fine-tuning."""
        trocas, notas = {}, {}
        if os.path.exists(self.caminho):
            with open(self.caminho, encoding="utf-8") as f:
                for linha in f:
                    d = json.loads(linha)
                    if d["tipo"] == "troca":
                        trocas[d["ts"]] = d
                    else:
                        notas[d["ref"]] = d["nota"]
        n = 0
        with open(saida, "w", encoding="utf-8") as f:
            for ts, d in trocas.items():
                if notas.get(ts) == "+":
                    ex = {"messages": [{"role": "system", "content": SISTEMA},
                                       {"role": "user", "content": d["pergunta"]},
                                       {"role": "assistant", "content": d["resposta"]}]}
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n += 1
        return n


def responder(pergunta, indice, historico, args, memoria):
    achados = indice.buscar(pergunta, k=args.trechos)
    if not achados:
        print("(nenhum trecho relevante encontrado nos arquivos)")
    if args.sem_llm:
        print(montar_contexto(achados) or "(sem resultados)")
        return None
    contexto = montar_contexto(achados)
    msgs = [{"role": "system", "content": SISTEMA}] + historico[-6:]
    msgs.append({"role": "user", "content": f"Trechos dos meus arquivos:\n\n{contexto}\n\nPergunta: {pergunta}"})
    try:
        resp = chamar_llm(msgs, args.modelo, args.url)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        print(f"\n[nao consegui falar com o Ollama em {args.url}: {e}]")
        print("Instale e inicie: https://ollama.com  e depois  ollama pull " + args.modelo)
        print("Enquanto isso, estes sao os trechos mais relevantes:\n")
        print(contexto or "(sem resultados)")
        return None
    print("\n" + resp + "\n")
    historico.append({"role": "user", "content": pergunta})
    historico.append({"role": "assistant", "content": resp})
    return memoria.registrar(pergunta, [f"{c[0]}:{c[1]}-{c[2]}" for _, c in achados], resp)


def main():
    ap = argparse.ArgumentParser(description="Assistente local de estudo de codigo")
    ap.add_argument("pergunta", nargs="?", help="se omitida, abre o modo interativo")
    ap.add_argument("--dir", default=".", help="pasta com o codigo a indexar")
    ap.add_argument("--modelo", default=os.environ.get("MODELO_OLLAMA", "qwen2.5-coder:7b"))
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--trechos", type=int, default=5, help="quantos trechos enviar ao modelo")
    ap.add_argument("--sem-llm", action="store_true", help="so mostra a busca")
    ap.add_argument("--memoria", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoria.jsonl"))
    ap.add_argument("--exportar-sft", metavar="ARQUIVO", help="exporta respostas aprovadas (+) para fine-tuning")
    args = ap.parse_args()

    memoria = Memoria(args.memoria)
    if args.exportar_sft:
        print(f"{memoria.exportar_sft(args.exportar_sft)} exemplos aprovados exportados para {args.exportar_sft}")
        return

    pasta = os.path.abspath(os.path.expanduser(args.dir))
    indice = Indice(pasta)
    print(f"indexados {len(indice.chunks)} trechos de {pasta}")
    historico = []

    if args.pergunta:
        responder(args.pergunta, indice, historico, args, memoria)
        return

    print("Digite sua pergunta. Comandos: /reindexar  /sair  e, apos uma resposta, + (boa) ou - (ruim).\n")
    ultimo = None
    while True:
        try:
            linha = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not linha:
            continue
        if linha == "/sair":
            break
        if linha == "/reindexar":
            indice = Indice(pasta)
            print(f"indexados {len(indice.chunks)} trechos")
        elif linha in ("+", "-"):
            if ultimo:
                memoria.avaliar(ultimo, linha)
                print("anotado." if linha == "+" else "anotado (nao entra no dataset).")
            else:
                print("ainda nao ha resposta para avaliar.")
        else:
            ultimo = responder(linha, indice, historico, args, memoria) or ultimo


if __name__ == "__main__":
    sys.exit(main())
