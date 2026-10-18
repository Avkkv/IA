# Minha IA: roteiro de estudo, código e matemática

Este projeto tem duas metas:

1. **Entender de verdade** como um modelo de linguagem funciona, escrevendo as peças você mesmo.
2. **Ter hoje um assistente local** que lê o seu código, ajuda nos estudos e guarda memória do que deu certo.

> A matemática está em `$...$`. No VS Code, o preview do Markdown (`Ctrl+Shift+V`) renderiza as fórmulas.

---

## 0. Uma conversa honesta sobre expectativas

**Treinar do zero algo comparável ao Claude não é possível em um computador pessoal.** Não é questão de otimizar em C. Modelos de fronteira são treinados com bilhões a centenas de bilhões de parâmetros, trilhões de tokens, milhares de GPUs por semanas ou meses, e custam milhões de dólares só de computação. Depois ainda passam por etapas de ajuste fino e de preferências humanas, que também exigem dados e pessoas.

Para ter uma noção da escala, a regra aproximada é que treinar custa $6 \cdot N \cdot D$ operações de ponto flutuante ($N$ = parâmetros, $D$ = tokens). Um modelo de 7 bilhões de parâmetros em 1 trilhão de tokens dá $6 \cdot 7\times10^{9} \cdot 10^{12} = 4{,}2\times10^{22}$ FLOPs. Uma GPU doméstica sustentando cerca de $1{,}5\times10^{13}$ FLOP/s levaria $2{,}8\times10^{9}$ s, ou **uns 90 anos**.

**O que é viável, e muito bom para estudar**, em degraus:

| Degrau | O que você faz | Onde está neste projeto |
|---|---|---|
| 1. Entender | Escrever autograd, otimizador e um GPT do zero | `01_`, `02_`, `03_` |
| 2. Usar | Rodar um modelo aberto local (offline) sobre o seu código | `04_assistente` |
| 3. Especializar | Ajustar esse modelo com os seus dados (LoRA) | seção 5 |
| 4. Jarvis | Voz, ferramentas, memória, automação | seção 6 |

**Sobre o seu cenário hipotético (IAs como eu inacessíveis):** modelos de pesos abertos, uma vez baixados, rodam 100% offline e ninguém pode tirá-los de você. Guarde uma cópia do instalador do Ollama e dos arquivos de modelo (ficam em `~/.ollama/models`) em um HD externo. Essa é a sua "apólice de seguro" mais barata.

---

## 1. Como rodar (Linux Mint)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
cd minha-ia
python3 -m venv .venv && source .venv/bin/activate
pip install numpy

python 01_autograd.py                      # autograd + XOR
cd 02_otimizacao_c && python bench.py      # C vs NumPy
cd ../03_mini_gpt
python gradcheck.py                        # prova que o backprop está certo
python train.py --dados ~/seus_projetos --passos 3000
python sample.py --prompt "def "
cd ../04_assistente
python assistente.py --dir ~/seus_projetos # precisa do Ollama (seção 4)
```

**Estrutura**

```
01_autograd.py        derivadas automáticas em Python puro
02_otimizacao_c/      multiplicação de matrizes em C: 4 versões + benchmark
03_mini_gpt/          GPT em NumPy com backprop escrito à mão
    gpt.py  train.py  sample.py  gradcheck.py  dados.py
04_assistente/        RAG sobre seu código + LLM local + memória
```

**Resultados que medi ao testar** (máquina de 2 núcleos, a sua deve variar):

- `gradcheck.py`: erro relativo máximo de ~3e-6 entre o gradiente escrito à mão e a derivada numérica.
- `train.py` com 110 mil parâmetros: perda cai de 4,4 para ~2,7 em 150 passos, a ~14 mil tokens/s.
- `bench.py` (n=512): 1,9 GFLOPS (ingênua) → 20 (ordem `ikj`) → 28 (com threads) → 270 (NumPy/BLAS).
  Com n=2048 a ingênua cai para 0,5 GFLOPS (o cache não aguenta o acesso em saltos), enquanto `ikj` mantém 14 e o BLAS passa de 300.
  Nesta máquina a versão blocada (tiling) **não** ganhou da `ikj`, nem em n=2048. O compilador já vetoriza bem a `ikj`, e o tiling só compensa quando as matrizes estouram o cache da sua CPU. Descobrir por quê no seu hardware é um bom exercício.

Com poucos minutos de treino o modelo gera texto com cara de código, mas sem sentido. Isso é normal. Ele aprende primeiro a frequência de caracteres, depois palavras, depois indentação e sintaxe. Experimente `--dim 128 --camadas 4 --passos 5000` com mais código seu.

---

## 2. Roteiro de estudo

**Fase 0: base (1 a 2 semanas).** Álgebra linear (vetores, produto matricial, autovalores só por cima) e cálculo (derivada, gradiente, regra da cadeia). Veja a série *Essence of Linear Algebra* e a de redes neurais do 3Blue1Brown, e o livro gratuito *Mathematics for Machine Learning* (Deisenroth et al.).

**Fase 1: autograd.** Rode `01_autograd.py`. Exercícios: adicione `sigmoid` e `sin`; treine a rede para separar dois círculos; troque MSE por entropia cruzada.

**Fase 2: C e desempenho.** Estude `matmul.c`. Exercícios: teste `bs` = 16, 32, 128, 256 e veja se a blocagem chega a ganhar da `ikj` na sua CPU (use `lscpu` para ver os tamanhos de cache L1/L2 e relacione com `bs`); escreva uma versão com intrínsecos AVX2; leia o projeto `llm.c` do Andrej Karpathy, que treina um GPT-2 inteiro em C.

**Fase 3: o GPT.** Leia `gpt.py` ao lado da seção 3 deste README. Exercícios, em ordem de dificuldade:
1. Treine com o seu próprio código e compare perplexidade com `--dim` e `--camadas` diferentes.
2. Adicione dropout (com o backward).
3. Troque a posição aprendida por RoPE.
4. Implemente BPE no lugar do tokenizador por caractere.
5. Implemente KV-cache na geração.

**Fase 4: PyTorch.** Reescreva o mini-GPT em PyTorch (≈100 linhas) e compare com o seu. Depois leia o `nanoGPT`. Você já vai saber o que o `loss.backward()` faz por dentro.

**Fase 5: modelos reais.** Seções 4 a 6.

**Referências:** *Attention Is All You Need* (Vaswani et al., 2017); *Deep Learning* (Goodfellow, Bengio, Courville); *Dive into Deep Learning* (d2l.ai); série *Neural Networks: Zero to Hero* (Karpathy); cursos Stanford CS224n e CS231n.

---

## 3. A matemática, ligada ao código

### 3.1 Aprender é minimizar uma perda

Temos dados $(x_i, y_i)$, um modelo $f_\theta$ com parâmetros $\theta$ e uma perda por exemplo $\ell$. Queremos minimizar

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i),\, y_i\big).$$

**Descida do gradiente.** Para um passo pequeno $\delta$, $L(\theta+\delta) \approx L(\theta) + \nabla L(\theta)\cdot\delta$. Essa aproximação diminui o máximo quando $\delta$ aponta contra o gradiente, então

$$\theta \leftarrow \theta - \eta\,\nabla L(\theta),$$

em que $\eta$ é a taxa de aprendizado. Todo o resto do campo é sobre calcular $\nabla L$ de forma barata e dar passos bons.

### 3.2 Álgebra linear: o que a GPU faz

Uma camada linear é $y = xW + b$, com $x\in\mathbb{R}^{d}$, $W\in\mathbb{R}^{d\times k}$. Em lote, $Y = XW + b$ com $X\in\mathbb{R}^{N\times d}$. O custo é $2Ndk$ operações, e é isso que `matmul.c` otimiza.

O produto escalar mede alinhamento: $a\cdot b = \|a\|\,\|b\|\cos\theta$. A atenção é, no fundo, "quanto cada posição se parece com cada outra".

### 3.3 Regra da cadeia e backpropagation

Para $L = \ell\big(f_3(f_2(f_1(x)))\big)$:

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial f_3}\cdot\frac{\partial f_3}{\partial f_2}\cdot\frac{\partial f_2}{\partial f_1}\cdot\frac{\partial f_1}{\partial \theta_1}.$$

Backprop calcula isso de trás para frente, reaproveitando os produtos parciais. Por isso o custo do backward é só ~2× o do forward, e não proporcional ao número de parâmetros. É o que `Valor.backward()` faz no `01_autograd.py`, com um valor de cada vez.

**Camada linear** (`lin_bwd` em `gpt.py`). Em componentes, $y_{ij} = \sum_k x_{ik}\,w_{kj} + b_j$. Logo

$$\frac{\partial L}{\partial w_{kj}} = \sum_i x_{ik}\,\frac{\partial L}{\partial y_{ij}}.$$

Em forma matricial, com $dY = \partial L/\partial Y$:

$$dW = X^{\top} dY,\qquad dX = dY\,W^{\top},\qquad db = \sum_i dY_{i,:}.$$

### 3.4 Probabilidade: softmax e entropia cruzada

O modelo emite *logits* $z\in\mathbb{R}^V$. O softmax os transforma em probabilidades:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.$$

A perda é a **entropia cruzada** do token correto $y$:

$$\ell = -\log p_y = -z_y + \log\sum_j e^{z_j}.$$

Minimizá-la equivale a maximizar a verossimilhança dos dados. A derivada é limpa:

$$\frac{\partial \ell}{\partial z_i} = -\mathbb{1}[i=y] + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - \mathbb{1}[i=y].$$

É a linha `dlog = probs - onehot` em `backward()`, dividida por $B\cdot T$ por causa da média.

Dois fatos úteis: a perda inicial de um modelo sem treino é $\ln V$ (o `train.py` imprime isso), e a **perplexidade** é $e^{L}$, o número efetivo de opções entre as quais o modelo "hesita". Para estabilidade numérica, subtraímos $\max_j z_j$ antes de exponenciar (não muda o resultado).

**Jacobiano do softmax.** $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$. Aplicado a um gradiente $dp$ que chega de cima:

$$dz = p \odot \big(dp - \textstyle\sum_j dp_j\,p_j\big).$$

Essa é a linha `ds = a * (da - (da*a).sum(...))` na atenção.

### 3.5 Otimizadores

**SGD:** $\theta \leftarrow \theta - \eta g$. **Adam** (usado em `train.py`) mantém médias móveis do gradiente e do seu quadrado:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,$$

$$\hat m_t = \frac{m_t}{1-\beta_1^{t}},\quad \hat v_t = \frac{v_t}{1-\beta_2^{t}},\qquad \theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.$$

A divisão por $\sqrt{\hat v_t}$ dá a cada parâmetro um passo adaptado à sua escala, e as correções $1-\beta^t$ compensam o início em zero. **AdamW** aplica o decaimento de peso separado do gradiente. Também usamos *warmup* linear seguido de queda em cosseno, e *recorte de gradiente* (se $\|g\|>1$, reescala para $\|g\|=1$) para evitar explosões.

### 3.6 Normalização, residuais e ativação

**LayerNorm** normaliza cada vetor:

$$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}},\qquad y = \gamma\odot\hat x + \beta.$$

O backward (`ln_bwd`), com $d\hat x = dy\odot\gamma$ e médias sobre a última dimensão:

$$dx = \frac{1}{\sigma}\Big(d\hat x - \overline{d\hat x} - \hat x\,\overline{d\hat x\odot\hat x}\Big).$$

**Conexão residual** $y = x + F(x)$ tem derivada $\partial y/\partial x = I + \partial F/\partial x$. O termo $I$ é um "atalho" que deixa o gradiente atravessar dezenas de camadas sem sumir. No código, `dh = dh + dx`.

**GELU** (a não linearidade): $\text{GELU}(x)\approx \tfrac12 x\big(1+\tanh(\sqrt{2/\pi}\,(x+0{,}044715x^3))\big)$. Sem não linearidade, empilhar camadas lineares seria o mesmo que uma só camada linear.

### 3.7 Modelagem de linguagem

Um modelo autorregressivo fatora a probabilidade de um texto como

$$P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t}).$$

Treinamos para prever o próximo token em **todas as posições ao mesmo tempo**, e a máscara causal garante que a posição $t$ não enxerga o futuro. O texto é convertido em ids (`dados.py`; modelos reais usam BPE, aqui é por caractere) e cada id vira um vetor por busca numa tabela $E\in\mathbb{R}^{V\times D}$. Os logits finais reaproveitam a mesma tabela, $z = h\,E^{\top}$ (*weight tying*), então "o vetor de uma palavra" serve tanto para entrada quanto para saída.

### 3.8 Atenção, o núcleo do transformer

Cada posição gera uma consulta $q$, uma chave $k$ e um valor $v$ ($Q=XW_Q$, $K=XW_K$, $V=XW_V$):

$$\text{Atenção}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}} + M\Big)V,$$

em que $M_{ij}=-\infty$ para $j>i$ (máscara causal).

**Por que dividir por $\sqrt{d_k}$?** Se as componentes de $q$ e $k$ são independentes com média 0 e variância 1, então $q\cdot k$ tem variância $d_k$. Sem a divisão, os escores ficam grandes, o softmax satura em quase one-hot e o gradiente some. Dividir devolve variância 1.

**Multi-cabeça:** a dimensão $D$ é dividida em $H$ grupos, cada um com sua própria atenção, e depois concatenados. Cada cabeça pode aprender uma relação diferente (por exemplo, "fecha o parêntese aberto antes").

**Backward** (`attn_bwd`), com $O=AV$, $S=QK^{\top}/\sqrt{d_k}$ e $dO$ vindo de cima:

$$dV = A^{\top}dO,\quad dA = dO\,V^{\top},\quad dS = A\odot\big(dA - \textstyle\sum dA\odot A\big),$$

$$dQ = \frac{dS\,K}{\sqrt{d_k}},\qquad dK = \frac{dS^{\top}Q}{\sqrt{d_k}}.$$

Os gradientes de $W_Q, W_K, W_V$ saem da regra da camada linear (3.3). Custo da atenção: $O(T^{2}D)$, e é por isso que contextos longos são caros.

**Bloco completo** (repete $L$ vezes): $h \leftarrow h + \text{Atenção}(\text{LN}(h))$ e depois $h \leftarrow h + \text{MLP}(\text{LN}(h))$, com MLP $= W_2\,\text{GELU}(W_1 x)$ e largura interna $4D$.

**Contagem de parâmetros.** Por camada: $4D^2$ (atenção) $+ 8D^2$ (MLP) $= 12D^2$, mais $VD$ do embedding. Confira com os padrões do `train.py` ($D=64$, $L=2$, $V=92$): $12\cdot 64^2\cdot 2 = 98\,304$, mais $92\cdot 64 + 64\cdot 64$ do embedding e posição, mais biases e LayerNorms, o que dá os 110 080 impressos.

### 3.9 Escala

A regra aproximada de custo está na seção 0. Os estudos de *leis de escala* mostram que a perda cai de forma previsível (lei de potência) com mais parâmetros, dados e computação, e que, para um orçamento fixo, o ideal é crescer modelo e dados juntos (algo em torno de dezenas de tokens por parâmetro). É por isso que modelo grande sozinho não basta: sem dados e computação na mesma proporção, ele fica subtreinado.

### 3.10 De "prever texto" a "assistente"

O treino por próximo token (pré-treino) cria um modelo que completa texto. Para virar um assistente há etapas posteriores:

1. **Ajuste supervisionado (SFT):** treinar em pares pergunta/resposta bons. O seu `assistente.py` gera exatamente esse tipo de dado quando você marca `+`.
2. **Aprendizado por preferências (RLHF, DPO e variantes):** o modelo aprende, comparando respostas, qual é preferida.
3. **Uso de ferramentas:** o modelo aprende a pedir uma busca, rodar código, ler um arquivo, e usar o resultado.

---

## 4. Passo 2: o assistente local (já funciona)

1. Instale o Ollama (https://ollama.com) e baixe um modelo de código que caiba na sua máquina (tabela abaixo):
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
2. Aponte o assistente para uma pasta de código:
   ```bash
   cd 04_assistente
   python assistente.py --dir ~/meus_projetos
   ```
3. Pergunte ("explique a função X de main.c", "por que esse laço é O(n²)?"). Ele busca os trechos mais relevantes por **BM25**, manda ao modelo e cita `arquivo:linhas`. Depois de uma resposta boa, digite `+`; se ruim, `-`.
4. Exporte o que você aprovou para treinar depois: `python assistente.py --exportar-sft sft.jsonl`.

Se o Ollama não estiver rodando, o assistente mostra só os trechos encontrados (você pode testar a busca com `--sem-llm`).

**BM25 em uma linha:** para cada termo da pergunta, soma $\text{idf}(t)\cdot\frac{f\,(k_1+1)}{f + k_1(1-b+b\,|d|/\overline{|d|})}$, em que $f$ é a frequência do termo no trecho. Termos raros pesam mais (idf), repetições têm retorno decrescente, e trechos longos são penalizados. É a base do RAG; o próximo upgrade é busca por **embeddings** (vetores de significado) em vez de palavras exatas.

**Que modelo rodar.** A regra de bolso é que a memória necessária é $\approx N_{\text{params}} \times$ bytes por parâmetro: 16 bits = 2 bytes, 4 bits (quantização Q4) ≈ 0,5 byte, mais alguns GB para o contexto. Segundo a lista do [Morph](https://www.morphllm.com/best-ollama-models), com tamanhos aproximados em Q4:

| Memória disponível | Exemplos de modelos de código | Tamanho aprox. |
|---|---|---|
| ~8 GB (VRAM) | `qwen2.5-coder` 7B | ~5 GB |
| ~12 GB | `deepseek-coder-v2` 16B | ~9 GB |
| 16 GB (VRAM ou RAM) | `qwen2.5-coder` 14B | ~9 GB |
| ~24 GB | `devstral` 24B, `qwen3-coder` 30B | ~14 a 19 GB |

Esses modelos mudam rápido. Confira a lista atual em https://ollama.com/library antes de baixar. Sem GPU, rodam só na CPU e RAM (mais lento, mas funciona com modelos de 3B a 7B).

---

## 5. Passo 3: especializar o modelo com LoRA

Em vez de atualizar todos os pesos $W\in\mathbb{R}^{d\times k}$, o **LoRA** congela $W$ e treina duas matrizes pequenas:

$$W' = W + \frac{\alpha}{r}\,B A,\qquad A\in\mathbb{R}^{r\times k},\; B\in\mathbb{R}^{d\times r},\; r \ll \min(d,k).$$

A intuição é que a mudança necessária para uma tarefa nova tem **posto baixo**. Parâmetros treinados: $r(d+k)$ em vez de $dk$. Para $d=k=4096$ e $r=16$: $131\,072$ contra $16\,777\,216$ (0,78%). Isso cabe em uma GPU doméstica, e combinado com pesos quantizados em 4 bits (QLoRA) permite ajustar modelos de 7B em placas de 12 a 16 GB.

Fluxo prático: acumule algumas centenas de pares aprovados com o `assistente.py`, exporte com `--exportar-sft`, e treine com as bibliotecas `transformers` + `peft` (ou ferramentas como Unsloth e Axolotl). As APIs mudam, então siga a documentação atual delas. Antes de mexer em ajuste fino, estude a Fase 4 (PyTorch) para entender o que está acontecendo.

---

## 6. Passo 4: rumo ao "Jarvis"

Um Jarvis não é um modelo único, é um **sistema** com peças que você pode montar uma de cada vez:

```
 microfone ─► reconhecimento de voz (Whisper local) ─┐
                                                     ▼
 memória (SQLite + embeddings) ◄──────────►  LLM local (Ollama)  ◄──► ferramentas
                                                     │              (ler arquivos, rodar
                                                     ▼               testes, agenda, shell)
 alto-falante ◄─ síntese de voz (Piper) ◄────── resposta
```

Ordem sugerida: (1) este assistente por texto; (2) acrescentar **ferramentas** (o modelo pede "ler `main.c`", seu código executa e devolve o resultado); (3) memória por **embeddings** em SQLite; (4) voz de entrada e saída; (5) integração com o sistema (atalhos, scripts, agenda). Tudo isso roda local no Linux. Dê só as permissões necessárias às ferramentas (por exemplo, só leitura de uma pasta, e nada de `rm` livre), porque um modelo pequeno comete erros.

---

## 7. O que fazer a seguir

- Rode os quatro módulos na ordem e altere cada um.
- Faça o `gradcheck.py` falhar de propósito (apague um termo do backward) para ver o teste pegar o erro. É o melhor jeito de sentir por que a derivada importa.
- Treine o mini-GPT no seu código e leia o que ele gera.
- Instale o Ollama e use o assistente nas suas matérias por duas semanas, aprovando as boas respostas.

Quando tiver dúvidas sobre qualquer fórmula ou trecho de código, me mande e eu explico passo a passo.
