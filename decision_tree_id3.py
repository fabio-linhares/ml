import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dados import dados_credito
from class_id3 import ID3DecisionTree
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPLEMENTAÇÃO DO ALGORITMO DE ÁRVORE DE DECISÃO - ID3")
print("="*80)

dados = dados_credito
df = pd.DataFrame(dados)

print(f"Base de dados carregada: {len(df)} exemplos")
print(f"Atributos: {list(df.columns[1:-1])}")  # sem id e alvo
print(f"Distribuição da classe alvo (Risco): {df['Risco'].value_counts().to_dict()}")

feature_columns = ['Historia_Credito', 'Divida', 'Garantia', 'Renda', "Idade", 'Tipo_Emprego']
X = df[feature_columns].copy()
y = df['Risco'].copy()

print(f"Features selecionadas: {feature_columns}")
print(f"Target: {y.name}")
print(f"Distribuição do target: {y.value_counts().to_dict()}")

print("\n" + "="*50)
print("TREINAMENTO DO ALGORITMO ID3")
print("="*50)

# discretizando a idade. Na resposta a mão preferi remover a coluno para não ter esse trabalho, 
# mas aqui vou fazer a discretização
print("Discretizando idade para ID3:")
age_bins = [0, 30, 45, 100]
age_labels = ['Jovem', 'Adulto', 'Idoso']
X_id3 = X.copy()
X_id3['Idade'] = pd.cut(X_id3['Idade'], bins=age_bins, labels=age_labels, include_lowest=True)
print(f"Idade discretizada: {X_id3['Idade'].value_counts().to_dict()}")

id3_model = ID3DecisionTree()
id3_model.fit(X_id3, y)

print("\nÁRVORE ID3 FINAL:")
id3_model.print_tree()

print("\n" + "="*80)
print("AVALIAÇÃO DO MODELO ID3")
print("="*80)

predictions_id3 = id3_model.predict(X_id3)
accuracy_id3 = accuracy_score(y, predictions_id3)
report = classification_report(y, predictions_id3, output_dict=True)

print(f"Acurácia ID3: {accuracy_id3:.3f} ({accuracy_id3*100:.1f}%)")
print(f"Precisão média: {report['weighted avg']['precision']:.3f}")
print(f"Recall médio: {report['weighted avg']['recall']:.3f}")
print(f"F1-score médio: {report['weighted avg']['f1-score']:.3f}")

print("\n" + "="*80)
print("JUSTIFICATIVAS TÉCNICAS DA IMPLEMENTAÇÃO ID3")
print("="*80)

justificativas = """
🔹 ID3 - DECISÕES DE IMPLEMENTAÇÃO:

1. ENTROPIA COMO MEDIDA DE IMPUREZA:
   - Fórmula: H(S) = -∑(pi * log2(pi))
   - JUSTIFICATIVA: Mede desordem/incerteza no conjunto
   - VANTAGEM: Matematicamente bem fundamentada
   - DESVANTAGEM: Computacionalmente cara (logaritmos)

2. INFORMATION GAIN COMO CRITÉRIO:
   - Fórmula: Gain(S,A) = H(S) - ∑((|Sv|/|S|) * H(Sv))
   - JUSTIFICATIVA: Redução da entropia após divisão
   - PROBLEMA: Favorece atributos com muitos valores únicos

3. DIVISÕES MÚLTIPLAS:
   - Cada valor do atributo gera um ramo
   - VANTAGEM: Natural para categóricos
   - DESVANTAGEM: Árvores muito largas

4. SEM TRATAMENTO DE CONTÍNUOS:
   - Requer discretização prévia
   - IMPLEMENTAÇÃO: Usamos pd.cut() para idade
   - LIMITAÇÃO: Perda de informação na discretização

5. EDGE CASES:
   - Conjuntos vazios: Retornar classe majoritária
   - Valores não vistos: Default conservador ("Baixo")
   - Divisões inválidas: Verificar tamanhos mínimos
   - JUSTIFICATIVA: Robustez em produção
"""
print(justificativas)

print("\n" + "="*60)
print("EXEMPLO PRÁTICO DE CLASSIFICAÇÃO COM ID3")
print("="*60)

novo_exemplo = {
    'Historia_Credito': 'Boa',
    'Divida': 'Baixa', 
    'Garantia': 'Adequada',
    'Renda': 'Acima de $35k',
    'Idade': 35,
    'Tipo_Emprego': 'Estável'
}

print("NOVO EXEMPLO PARA CLASSIFICAR:")
for k, v in novo_exemplo.items():
    print(f"  {k}: {v}")

exemplo_df = pd.DataFrame([novo_exemplo])
exemplo_id3 = exemplo_df.copy()
exemplo_id3['Idade'] = pd.cut(exemplo_id3['Idade'], bins=[0, 30, 45, 100], 
                             labels=['Jovem', 'Adulto', 'Idoso'], include_lowest=True)

pred_id3 = id3_model.predict(exemplo_id3)[0]
print(f"\nPREDIÇÃO ID3: {pred_id3}")

print("\n" + "="*80)
print("IMPLEMENTAÇÃO ID3 FINALIZADA!")
print("="*80)
