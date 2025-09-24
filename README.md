# Análise e Resolução — Lista de Exercícios 1 (Aprendizado de Máquina)

**Autor:** Fábio Linhares
**Resumo:** Resolução completa, questão a questão, com código, resultados e material de suporte (prints dos cálculos manuais). Este README descreve a estrutura do repositório, como reproduzir os experimentos, as decisões metodológicas, resultados-chave e observações críticas.

---

## Índice

1. [Estrutura do repositório](#estrutura-do-repositório)
2. [Como replicar (quickstart)](#como-replicar-quickstart)
3. [Dados (`clientes.csv`)](#dados-clientescsv)
4. [Questão 1 — Expansão da base e construção manual de árvores](#questão-1---expansão-da-base-e-construção-manual-de-árvores)
5. [Questão 2 — Implementação com bibliotecas (Scikit-learn) e código próprio](#questão-2---implementação-com-bibliotecas-scikit-learn-e-código-próprio)
6. [Questão 3 — Dataset do Kaggle (Heart Disease) e regras diretas (RIPPER)](#questão-3---dataset-do-kaggle-heart-disease-e-regras-diretas-ripper)
7. [Questões 4 & 5 — Overfitting, Underfitting e o papel do C4.5](#questões-4--5---overfitting-underfitting-e-o-papel-do-c45)
8. [Questão 6 — k-Nearest Neighbors (kNN): explicação, exemplo numérico e limitações](#questão-6---k%E2%80%8Bnearest-neighbors-knn-explicação-exemplo-numérico-e-limitações)
9. [Reprodutibilidade, limitações e notas metodológicas](#reprodutibilidade-limitações-e-notas-metodológicas)
10. [Contribuições, contato e licença](#contribuições-contato-e-licença)

---

## Estrutura do repositório

```
.
├── sodeusnacausa/      # Prints/screenshots com os cálculos manuais no caderno (Questão 1)
├── app.py              # Aplicação web mínima para interação (opcional)
├── class_cart.py       # Implementação didática do CART (do zero)
├── class_c45.py        # Implementação didática do C4.5 (do zero - versão pedagógica)
├── class_id3.py        # Implementação didática do ID3 (do zero)
├── clientes.csv        # Base de dados expandida (30 instâncias) — fonte principal
├── optimized_tree.py   # Implementação usando scikit-learn + exportação de regras
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

---

## Como replicar (quickstart)

**Requisitos mínimos**

* Python 3.10+ (testado em 3.10 / 3.11)
* `pip`

**Instalação (recomendado: ambiente virtual)**

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Conteúdo sugerido de `requirements.txt`**

```
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.8.4
wittgenstein==0.10.1      # para JRip / RIPPER (Questão 3)
```

**Exemplos de execução**

* Treinar árvore (Scikit-learn) com o dataset `clientes.csv`:

```bash
python optimized_tree.py --dataset clientes.csv --criterion gini
```

* Rodar a aplicação web (se aplicável):

```bash
python app.py
# acessar http://localhost:5000 (ou porta configurada)
```

*(Obs.: os scripts contêm instruções de uso/argumentos — abra o cabeçalho dos scripts para opções adicionais.)*

---

## Dados (`clientes.csv`)

**Descrição resumida**

* 30 instâncias
* 6 atributos (ex.: Renda, História de Crédito, Garantia, Idade, Tipo de Emprego, etc.)
* Target/Classe: `Risco` com três níveis: `Alto`, `Moderado`, `Baixo`

**Distribuição de classes**

* Alto Risco: 10
* Moderado Risco: 8
* Baixo Risco: 12
* Total: 30

> Observação: o arquivo `clientes.csv` é a versão expandida usada em todas as análises deste exercício.

---

## Questão 1 — Expansão da base e construção manual de árvores

### Objetivos

1. Expandir a base original para 30 exemplos.
2. Construir manualmente três árvores: **ID3**, **C4.5** e **CART**, mostrando cálculos.
3. Extrair regras `SE ... ENTÃO`.
4. Comparar e recomendar a melhor base de regras.

### Metodologia e fórmulas utilizadas

* **Entropia:** `Entropy(S) = - Σ p_i log2(p_i)`
* **Ganho de Informação (ID3/C4.5):** `Gain(S, A) = Entropy(S) - Σ (|S_v|/|S|) Entropy(S_v)`
* **Índice Gini (CART):** `Gini(S) = 1 - Σ p_i^2`

### Cálculo do nó raiz (dataset completo)

* `p_Alto = 10/30`, `p_Moderado = 8/30`, `p_Baixo = 12/30`
* **Entropia (raiz, ID3/C4.5):**

  ```
  Entropy(S) = -[ (10/30)log2(10/30) + (8/30)log2(8/30) + (12/30)log2(12/30) ] ≈ 1.565
  ```
* **Gini (raiz, CART):**

  ```
  Gini(S) = 1 - [ (10/30)^2 + (8/30)^2 + (12/30)^2 ] ≈ 0.658
  ```

> Os cálculos completos (passo a passo, por nó) foram feitos em papel; os prints estão em `sodeusnacausa/` (fotografias dos cadernos).

### Árvores finais (resumo em ASCII)

**ID3** (exemplo resumido)

```
         [Renda]
        /   |    \
  $0-15k $15-35k  >$35k
   |       |        \
[HistCred] [HistCred] [Baixo]
```

**C4.5** (semelhante ao ID3, mas com cortes contínuos — exemplo)

```
         [Renda]
        /   |    \
  $0-15k $15-35k  >$35k
   |       |        \
 [Alto] [Idade <= 32.5] [Baixo]
```

**CART** (sempre binária)

```
        [Renda == '$0-$15k'?]
         /               \
       Sim               Nao
       |                   |
     [Alto]        [Renda == '> $35k'?]
                   /               \
                 Sim               Nao
                 |                  |
               [Baixo]          [Moderado]
```

### Regras extraídas (exemplos)

* **ID3:** (8 regras; mais específicas — risco de overfitting)
  `SE Renda = '$0 a $15k' E Historia de Credito = 'Ruim' ENTÃO Risco = 'Alto'`
* **C4.5:** (regras mais podadas/generais)
  `SE Renda = 'Acima de $35k' ENTÃO Risco = 'Baixo'`
* **CART:** (conjunto enxuto, binário, fácil interpretação)
  `SE Renda != '$0 a $15k' E Renda = 'Acima de $35k' ENTÃO Risco = 'Baixo'`

### Comparação (resumo)

| Critério           |                     ID3 |                   C4.5 |              CART |
| ------------------ | ----------------------: | ---------------------: | ----------------: |
| Simplicidade       |                   Baixa |                  Média |          **Alta** |
| Acurácia (treino)  | Alta (overfit possível) | Alta, mais generalista | Boa e equilibrada |
| Interpretabilidade |                Moderada |                    Boa |     **Excelente** |

**Recomendação (opinião do autor):** para sistemas de apoio à decisão em que interpretabilidade/clareza são críticas, **CART** foi a escolha mais adequada.
**Nota (consenso na literatura):** C4.5 (e suas variantes) também é altamente recomendada devido à poda estatística; escolha depende do trade-off desejado entre precisão e generalização.

---

## Questão 2 — Implementação com bibliotecas (Scikit-learn) e código próprio

### Arquivos relevantes

* `class_id3.py`, `class_c45.py`, `class_cart.py` — implementações didáticas (do zero) para fins educacionais.
* `optimized_tree.py` — utilização do `scikit-learn` (`DecisionTreeClassifier`) com exportação de regras via `export_text`.

### Pré-processamento padrão

* Conversão de variáveis categóricas → **Label Encoding** (para demonstrar comportamento do `DecisionTreeClassifier`).
* Normalmente, para interpretabilidade, recomenda-se `OneHotEncoder` em alguns casos; aqui usamos LabelEncoding para manter coerência com os exemplos didáticos.

### Exemplo de saída (criterio `gini`)

**Acurácia no conjunto de treino:** `93.33%`
**Regras (`export_text`) — exemplo**

```text
|--- Renda <= 1.50
|   |--- Historia de Crédito <= 0.50
|   |   |--- class: Moderado
|   |--- Historia de Crédito > 0.50
|   |   |--- class: Alto
|--- Renda > 1.50
|   |--- class: Baixo
```

*(os limiares como `0.50` e `1.50` provêm do Label Encoding; mapeamentos estão no script)*

### Observações técnicas

* `scikit-learn` produz árvores binárias por padrão (CART) e faz otimizações que reduzem overfitting aparente (ex.: parâmetros `min_samples_split`, `max_depth`, etc.).
* Implementações “do zero” são pedagógicas e não substituem `sklearn` em produção, mas são excelentes para entender internals (ganho, entropia, Gini, seleção de corte contínuo).

---

## Questão 3 — Dataset do Kaggle (Heart Disease UCI) e regras diretas (RIPPER)

### Dataset escolhido

* **Heart Disease UCI** — conjunto clássico, alvo binário (`target`), mistura de variáveis contínuas e categóricas. (Ex.: `cp`, `thal`, `ca`, `exang`, etc.)

### Procedimento

1. Baixar dataset (Kaggle).
2. Pré-processar (tratamento de nulos, normalização/encoding conforme necessário).
3. Treinar árvore de decisão (`scikit-learn`) e gerar métricas (80/20 train/test).
4. Treinar JRip / RIPPER (via `wittgenstein`) para gerar regras diretamente.

### Resultados (resumo)

* **Acurácia (80/20):** `85.25%`
* **Matriz de confusão (exemplo):**

  ```
  Prev.0 | Prev.1
  Real 0 | 24 | 5
  Real 1 |  4 | 28
  ```
* **Relatório de classificação (exemplo):**

  ```
  Classe 0 — precision 0.86 | recall 0.83 | f1 0.84 | support 29
  Classe 1 — precision 0.85 | recall 0.88 | f1 0.86 | support 32
  accuracy = 0.85 (n=61)
  ```
* **Regras (RIPPER, exemplo simplificado):**

  ```
  IF (cp = 0) AND (ca = 0) THEN target = 0
  IF (thal = 2) AND (exang = 0) THEN target = 0
  IF (ca = 0) AND (slope = 1) THEN target = 0
  ELSE target = 1
  ```

  *(regras simplificadas para ilustração — ver `optimized_tree.py` / saída do `wittgenstein` para conjunto completo)*

### Observação comparativa

* Árvores: cobrem o espaço completo (mapeamento exaustivo, mutuamente exclusivo).
* RIPPER/JRip: produz uma lista ordenada de regras — a primeira aplicável determina o rótulo; tendem a ser curtas e focadas.

---

## Questões 4 & 5 — Overfitting, Underfitting e o papel do C4.5

### Definições (concisas)

* **Overfitting (sobreajuste):** modelo que ajusta o ruído dos dados de treino. Alto desempenho no treino; baixa generalização no teste.
  *Analogia didática:* aluno que decora respostas específicas da lista, mas não resolve problemas novos.
* **Underfitting (subajuste):** modelo demasiado simples; não captura a estrutura dos dados. Performance ruim em treino e teste.
  *Analogia:* aluno que não estudou o conteúdo.

### Como C4.5 ajuda

* **Poda pós-crescimento (pessimistic pruning):** cresce árvore até pureza e depois poda nós que não reduzem o erro esperado — troca complexidade por generalização.
* **Melhor tratamento de atributos contínuos:** ponto de corte único e estatisticamente justificado.
* **Critérios de parada / mínimos:** evita divisões em nós com poucos exemplos (reduz regras espúrias).

**Opinião do autor:** essas características tornam C4.5 (e variantes) robusto para uso didático e em casos onde controle de overfitting é necessário, sem sacrificar interpretabilidade.

---

## Questão 6 — k-Nearest Neighbors (kNN)

### (a) Exemplo numérico (k = 1, 3, 7)

**Dataset (x (renda em x1000), y (idade)):**

```
P1 (50,30) — A
P2 (80,40) — B
P3 (90,35) — B
P4 (40,25) — A
P5 (85,45) — B
P6 (60,35) — A
P? (70,30) — ? (a classificar)
```

**Distâncias euclidianas (cálculo passo a passo):**

* d(P?,P1) = sqrt((70−50)² + (30−30)²) = sqrt(400+0) = 20.00
* d(P?,P2) = sqrt((70−80)² + (30−40)²) = sqrt(100+100) ≈ 14.142 → 14.14
* d(P?,P3) = sqrt((70−90)² + (30−35)²) = sqrt(400+25) ≈ 20.616 → 20.61
* d(P?,P4) = sqrt((70−40)² + (30−25)²) = sqrt(900+25) ≈ 30.414 → 30.41
* d(P?,P5) = sqrt((70−85)² + (30−45)²) = sqrt(225+225) ≈ 21.213 → 21.21
* d(P?,P6) = sqrt((70−60)² + (30−35)²) = sqrt(100+25) ≈ 11.180 → 11.18

**Ordenação (mais próximo → mais distante):** P6, P2, P1, P3, P5, P4

* `k=1` → vizinho: P6 (A) → **A**
* `k=3` → vizinhos: P6 (A), P2 (B), P1 (A) → votação (A:2, B:1) → **A**
* `k=7` (aqui n=6 → k=6): empate A:3 vs B:3 → estratégias: desempate por soma/ média de distâncias ou reduzir k (ex.: k ímpar)

### (b) Como escolher k

* *Regra-de-polegar:* k ≈ √N (N = nº amostras).
* *Melhor prática:* validação cruzada para testar vários k e escolher o que maximiza a métrica de interesse.

### (c) Falhas da distância Euclidiana e alternativa (Gower)

* **Problema:** atributos com escalas diferentes e dados mistos (numérico + categórico).
* **Alternativa recomendada para dados mistos:** *Distância de Gower* — normaliza numéricos (por range) e usa 0/1 para categóricos; soma ponderada dá a distância final.

Exemplo Gower (Renda em R\$):

* `|50000 − 51000| / range = 1000 / 100000 = 0.01`
* Estado civil diferente → distância categórica = 1
* Gower média = (0.01 + 1) / 2 = 0.505

### (d) Cenários onde kNN é ineficaz

* Alta dimensionalidade (a vizinhança perde sentido).
* Conjuntos de treino muito grandes (previsões custosas).
* Classes fortemente desbalanceadas.
* Dados mistos sem uso de distâncias apropriadas.

### (e) Lazy vs Eager

* **kNN:** *lazy* — quase nenhum treino, custo na predição.
* **Árvores/SVM:** *eager* — treino caro, predição rápida.

---

## Reprodutibilidade, limitações e notas metodológicas

* **Cálculos manuais:** realizados em papel; os scans/fotos estão em `sodeusnacausa/`. 
* **Pré-processamento:** LabelEncoding foi usado para simplicidade didática. Em aplicações reais, possivelmente utilizaríamos `OneHotEncoding` e normalização, que é importante.



