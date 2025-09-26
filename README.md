
<p align="center">
  <img src="https://ufal.br/ufal/comunicacao/identidade-visual/brasao/ods/ufal_ods1.png" alt="Logo UFAL" width="320"/>
</p>



<h1 align="center">
  Lista de Exercícios — Aprendizagem de Máquina<br>
  Mestrado em Informática — UFAL
</h1>

<p align="center">
  <b>Disciplina:</b> Aprendizagem de Máquina<br>
  <b>Professor:</b> Evandro de Barros Costa (<a href="mailto:evandro@ic.ufal.br">evandro@ic.ufal.br</a>)<br>
  <b>Departamento:</b> Instituto de Computação — UFAL<br>
  <b>Lattes:</b> <a href="http://lattes.cnpq.br/5760364940162939">http://lattes.cnpq.br/5760364940162939</a>
</p>

<p align="center">
  <b>Autor:</b> Fábio Linhares<br>
  <b>Lattes:</b> <a href="http://lattes.cnpq.br/7908261028551208">http://lattes.cnpq.br/7908261028551208</a>
</p>

<p align="center">
  <i>Este trabalho integra as atividades da disciplina de Aprendizagem de Máquina do Programa de Pós-Graduação em Informática (Mestrado), Universidade Federal de Alagoas.</i>
</p>

<div style="text-align: justify;">

# Lista de Exercícios 1

**Resumo:** Resolução completa, questão a questão, com código, resultados e material de suporte (prints dos cálculos manuais). Este README descreve a estrutura do repositório, como reproduzir os experimentos, as decisões metodológicas, resultados-chave e observações críticas.

</div>

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
├── app.py              # Aplicação web mínima para interação
├── class_cart.py       # Implementação didática do CART
├── class_c45.py        # Implementação didática do C4.5
├── class_id3.py        # Implementação didática do ID3
├── clientes.csv        # Base de dados expandida (30 instâncias) — fonte principal
├── optimized_tree.py   # Implementação usando scikit-learn + exportação de regras
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

---

<div style="text-align: justify;">

## Instalação e Execução

Recomendamos a criação de um ambiente virtual para isolar as dependências do projeto.
Ainda assim, você tem total liberdade para adotar o procedimento que considerar mais conveniente. Nossa orientação contempla duas opções bastante utilizadas no mercado: **venv** – ferramenta nativa e padrão do Python para criação de ambientes virtuais, simples e leve ou **Miniconda** – uma versão reduzida do Anaconda, que inclui o gerenciador de pacotes Conda sem trazer, por padrão, o conjunto extenso de bibliotecas pré-instaladas.

### 1. Criação do Ambiente Virtual
**Usando venv** : cria um ambiente isolado em uma pasta chamada `venv`:

```bash
python -m venv venv
```

**Usando Miniconda**: cria um ambiente isolado chamado `lista1_ml` com uma versão específica do Python:

```bash
conda create --name lista1_ml python=3.12
```

---

### 2. Ativação e Desativação do Ambiente

É crucial ativar o ambiente antes de instalar pacotes ou executar a aplicação.

**Com venv**

Para ativar:

```bash
# No Linux ou macOS
source venv/bin/activate

# No Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

Para desativar:

```bash
deactivate
```

**Com Miniconda**

Para ativar:

```bash
conda activate lista1_ml
```

Para desativar:

```bash
conda deactivate
```

---

### 3. Instalação das Dependências

Com o ambiente devidamente ativado, instale todos os pacotes necessários:

```bash
pip install -r requirements.txt
```

---

### 4. Executando a Aplicação

Para iniciar o servidor do Streamlit e visualizar a aplicação, execute:

```bash
streamlit run app.py 
# acesse http://localhost:8501
```
---

## Questão 1 — Expansão da base e construção manual de árvores

### Objetivos

1. Expandir a base original para 30 exemplos.
2. Construir manualmente três árvores: **ID3**, **C4.5** e **CART**, mostrando cálculos.
3. Extrair regras `SE ... ENTÃO`.
4. Comparar e recomendar a melhor base de regras.

## Base de Dados

**Descrição**


A primeira questão da lista de exercícios solicitava que, a partir de uma base fornecida pelo “gerente do banco” fosse realizada uma ampliação para que contivesse 6 atributos e 30 exemplos divididos entre as classes de Risco Baixo, Moderado e Alto. Como a referida base continha apenas 14 instâncias, fizemos a adição de 16, distribuídas entre as classes solicitadas, mantendo a coerência com os dados originais. A base final, disponível no arquivo `clientes.csv`, possui as seguintes características:

* 30 instâncias
* 6 atributos (Renda, História de Crédito, Garantia, Idade, Tipo de Emprego e Risco)
* Target/Classe: `Risco` com três níveis: `Baixo`, `Moderado` e `Alto`.

**Distribuição de classes**

* Baixo Risco: 12
* Moderado Risco: 8
* Alto Risco: 10
* Total: 30

> Observação: o arquivo `clientes.csv` foi usado em todas as análises deste exercício.

---


### Metodologia e fórmulas utilizadas

* **Entropia:** `Entropy(S) = - Σ p_i log2(p_i)`
* **Ganho de Informação (ID3/C4.5):** `Gain(S, A) = Entropy(S) - Σ (|S_v|/|S|) Entropy(S_v)`
* **Índice Gini (CART):** `Gini(S) = 1 - Σ p_i^2`

### Cálculo do nó raiz

* `p_Baixo = 12/30`, `p_Moderado = 8/30`, `p_Alto = 10/30`

* **Entropia (raiz, ID3/C4.5):**

  ```
  Entropy(S) = -[ (12/30)log2(12/30) + (8/30)log2(8/30) + (10/30)log2(10/30) ] ≈ 1.565
  ```
* **Gini (raiz, CART):**

  ```
  Gini(S) = 1 - [ (12/30)^2 + (8/30)^2 + (10/30)^2 ] ≈ 0.658
  ```

> Os cálculos completos (passo a passo) foram feitos em papel e os prints estão em `sodeusnacausa/` .

### Árvores finais

**ID3**

```
  [ Renda ]
     /    |    \
 ($0-$15k) ($15-$35k) (> $35k)
    |         |          \
 [Hist. Cred] [Hist. Cred] [ Baixo ]
  /  |   \       /   |   \
(R) (D)  (B)   (R) (D) (B)
 |   |    |     |   |   |
[A] [A] [Garantia] [A] [M] [M]
    /    \
  (N)    (A)
   |      |
  [A]    [M]
```

Legenda: R=Ruim, D=Desconhecida, B=Boa, N=Nenhuma, A=Adequada  
    A=Alto, M=Moderado
```

**C4.5** (semelhante ao ID3, mas com cortes contínuos)

```
         [Renda]
        /   |    \
  $0-15k $15-35k  >$35k
   |       |        \
 [Alto] [Idade <= 32.5] [Baixo]
```

**CART**

```
  [ Renda = '$0-$15k'? ]
  /                   \
    Sim                    Nao
     |                      |
   [ Alto ]        [ Renda = '> $35k'? ]
       /                  \
         Sim                   Nao
      |                     |
        [ Baixo ]      [ Hist. Cred = 'Ruim'? ]
            /                     \
          Sim                      Nao
           |                        |
             [ Alto ]                [ Moderado ]
```
```

### Regras extraídas

> Observação: em aula eu comentei que não recordava a ocorrencia de over e underfitting em árvores de decisão. Após a realização destes exercícios percebi que o overfitting esta mais relacionados às árvores não podadas (como ID3 puro). Já o underfitting, que é menos comum, e talvez isso justifique minha falta de memória, pode ocorrer se a árvore for excessivamente simplificada ou se os dados forem muito ruidosos.

#### **Regras da Árvore ID3**
- **R1:** SE Renda = '$0 a $15k' E História de Crédito = 'Ruim' ENTÃO Risco = 'Alto' (Suporte=3, Acurácia=100%)
- **R2:** SE Renda = '$0 a $15k' E História de Crédito = 'Desconhecida' ENTÃO Risco = 'Alto' (Suporte=3, Acurácia=100%)
- **R3:** SE Renda = '$0 a $15k' E História de Crédito = 'Boa' E Garantia = 'Nenhuma' ENTÃO Risco = 'Alto' (Suporte=1, Acurácia=100%)
- **R4:** SE Renda = '$0 a $15k' E História de Crédito = 'Boa' E Garantia = 'Adequada' ENTÃO Risco = 'Moderado' (Suporte=1, Acurácia=100%)
- **R5:** SE Renda = '$15 a $35k' E História de Crédito = 'Ruim' ENTÃO Risco = 'Alto' (Suporte=2, Acurácia=100%)
- **R6:** SE Renda = '$15 a $35k' E História de Crédito = 'Desconhecida' ENTÃO Risco = 'Moderado' (Suporte=1, Acurácia=100%)
- **R7:** SE Renda = '$15 a $35k' E História de Crédito = 'Boa' ENTÃO Risco = 'Moderado' (Suporte=4, Acurácia=50%)
- **R8:** SE Renda = 'Acima de $35k' ENTÃO Risco = 'Baixo' (Suporte=13, Acurácia=76,9%)

#### **Regras da Árvore C4.5**
- **R1:** SE Renda = '$0 a $15k' ENTÃO Risco = 'Alto' (Suporte=8, Acurácia=87,5%)
- **R2:** SE Renda = '$15 a $35k' E Idade <= 30.5 E História de Crédito = 'Desconhecida' ENTÃO Risco = 'Moderado' (Suporte=1, Acurácia=100%)
- *(Demais ramos seguem lógica semelhante, com regras mais gerais devido à poda.)*

#### **Regras da Árvore CART**
- **R1:** SE Renda = '$0 a $15k' ENTÃO Risco = 'Alto' (Suporte=8, Acurácia=87,5%)
- **R2:** SE Renda != '$0 a $15k' E Renda = 'Acima de $35k' ENTÃO Risco = 'Baixo' (Suporte=13, Acurácia=76,9%)
- **R3:** SE Renda != '$0 a $15k' E Renda != 'Acima de $35k' E História de Crédito = 'Ruim' ENTÃO Risco = 'Alto' (Suporte=2, Acurácia=100%)
- **R4:** SE Renda != '$0 a $15k' E Renda != 'Acima de $35k' E História de Crédito != 'Ruim' ENTÃO Risco = 'Moderado' (Suporte=7, Acurácia=71,4%)

---

### (iv) Comparação e Seleção da Base de Regras

| Critério           | ID3                                         | C4.5                                  | CART                                 |
|--------------------|---------------------------------------------|---------------------------------------|--------------------------------------|
| **Simplicidade**   | Muitas regras, algumas muito específicas    | Menos regras, mais gerais (poda)      | Poucas regras, estrutura binária     |
| **Exatidão (Treino)** | Alta (tende a overfit)                  | Alta, controlada por poda             | Boa, balanceada                      |
| **Interpretabilidade** | Moderada, granularidade pode confundir | Boa                                   | Excelente, lógica binária clara      |
| **Robustez**       | Baixa (sensível a ruído)                   | Média (poda ajuda)                    | Média                                |

**Reflexão:**

A base de regras do CART é a mais concisa (4 regras) e possui estrutura binária de fácil interpretação. As regras cobrem todos os casos de forma mutuamente exclusiva. O ID3 gera muitas regras muito específicas, sugerindo overfitting. Sinceramene, preferimos o CART pela simplicidade e clareza, **principalmente porque tivemos que fazer a mão**, embora tais característias sejam em essencia, fundamentais em sistemas de apoio à decisão. A regra "SE Renda é alta, ENTÃO Risco é Baixo" é intuitiva. A leve perda de acurácia é compensada pela interpretabilidade. De modo geral, pareceu-nos que modelos mais simples (C4.5 podado ou CART) tendem a generalizar melhor e, por conta disso, serem preferíveis em domínios onde interpretabilidade é crítica, como na área de finanças, por exemplo.


---

## Questão 2 — Implementação com bibliotecas (Scikit-learn) e código próprio

### Arquivos relevantes

* `class_id3.py`, `class_c45.py`, `class_cart.py` — implementações, digamos, customizadas, já que diferem um pouco do clássico.
* `optimized_tree.py` — criado para abordar limitações observadas nas implementações didáticas dos algoritmos de árvore de decisão (ID3, C4.5, CART), especialmente em relação à eficiência, robustez e clareza das regras extraídas. Com o `scikit-learn`, buscamos aproveitar algoritmos otimizados, controles de complexidade (como profundidade máxima e número mínimo de amostras por nó) e métodos de exportação de regras mais legíveis. Um ponto importante foi evitar problemas de recursão excessiva ou loops infinitos, comuns em implementações manuais, especialmente quando os critérios de parada não são bem definidos ou quando há dados inconsistentes. O script também facilita a comparação entre árvores geradas automaticamente e as construídas manualmente, permitindo avaliar ganhos em desempenho, interpretabilidade e generalização.

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
* Implementações “do zero”, como estas, são pedagógicas e não substituem `sklearn` em produção, mas são excelentes para entender os processos, os conceitos fundamentais como ganho, entropia, Gini, seleção de corte contínuo, etc.

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
* **Matriz de confusão:**

  ```
  Prev.0 | Prev.1
  Real 0 | 24 | 5
  Real 1 |  4 | 28
  ```
* **Relatório de classificação:**

  ```
  Classe 0 — precision 0.86 | recall 0.83 | f1 0.84 | support 29
  Classe 1 — precision 0.85 | recall 0.88 | f1 0.86 | support 32
  accuracy = 0.85 (n=61)
  ```
* **Regras (RIPPER simplificado):**

  ```
  IF (cp = 0) AND (ca = 0) THEN target = 0
  IF (thal = 2) AND (exang = 0) THEN target = 0
  IF (ca = 0) AND (slope = 1) THEN target = 0
  ELSE target = 1
  ```

  *(regras simplificadas para apresentação aqui (cansado de escrever...) Veja `optimized_tree.py` ou execute a aplicação para visualizar os dados completo)*

### Observação comparativa

* Árvores: cobrem o espaço completo (mapeamento exaustivo, mutuamente exclusivo).
* RIPPER/JRip: produz uma lista ordenada de regras — a primeira aplicável determina o rótulo; tendem a ser curtas e focadas.

---

## Questões 4 & 5 — Overfitting, Underfitting e o papel do C4.5

### Definições

* **Overfitting (sobreajuste):** modelo que ajusta o ruído dos dados de treino. Alto desempenho no treino; baixa generalização no teste.

  *Analogia didática:* aluno que decora respostas específicas da lista, mas não resolve problemas novos.

* **Underfitting (subajuste):** modelo demasiado simples; não captura a estrutura dos dados. Performance ruim em treino e teste.

  *Analogia:* aluno que não estudou o conteúdo.

### Como C4.5 ajuda?

* **Poda pós-crescimento (pessimistic pruning):** cresce árvore até pureza e depois poda nós que não reduzem o erro esperado — troca complexidade por generalização.

* **Melhor tratamento de atributos contínuos:** ponto de corte único e estatisticamente justificado.

* **Critérios de parada / mínimos:** evita divisões em nós com poucos exemplos (reduz regras espúrias).

Ao fim e ao cabo, essas características parecem tornar o C4.5 (e variantes) robusto para uso didático e em casos onde controle de overfitting é necessário, sem sacrificar interpretabilidade.

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

</div>  