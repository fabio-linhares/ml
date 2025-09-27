import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from dados import dados_credito

# --- Importação dos Modelos ---
from class_id3 import ID3DecisionTree
from class_c45 import C45DecisionTree
from class_cart import CARTDecisionTree

# --- Configurações ---
OUTPUT_DIR = "resultados_img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funções Auxiliares ---
def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Cria, salva e exibe um gráfico da matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'Matriz de Confusão - {model_name}', fontsize=16)
    ax.set_xlabel('Valores Preditos', fontsize=12)
    ax.set_ylabel('Valores Reais', fontsize=12)
    
    # Salvar a figura
    filepath = os.path.join(OUTPUT_DIR, f"matriz_confusao_{model_name.lower()}.png")
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {filepath}")
    return filepath

def save_tree_visualization(model, class_names, model_name):
    """Gera e salva a visualização da árvore de decisão."""
    try:
        dot = graphviz.Digraph(comment=f'Decision Tree - {model_name}', graph_attr={'rankdir': 'TB'})
        model.model._get_tree_graph(dot, model.get_tree_structure(), class_names=class_names)
        
        # Salvar o gráfico
        filepath = os.path.join(OUTPUT_DIR, f"arvore_{model_name.lower()}")
        dot.render(filepath, format='png', cleanup=True)
        print(f"Visualização da árvore salva em: {filepath}.png")
        return f"{filepath}.png"
    except Exception as e:
        print(f"Não foi possível gerar o gráfico da árvore para {model_name}: {e}")
        return None

# --- Carregamento de Dados ---
df = pd.DataFrame(dados_credito)
feature_columns = ['Historia_Credito', 'Divida', 'Garantia', 'Renda', 'Idade', 'Tipo_Emprego']
target_column = 'Risco'

X = df[feature_columns]
y = df[target_column]

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
class_names = sorted(y.unique())

# --- Modelos a serem avaliados ---
models = {
    "ID3": ID3DecisionTree(max_depth=5),
    "C4.5": C45DecisionTree(max_depth=5),
    "CART": CARTDecisionTree(max_depth=5)
}

results = {}

# --- Loop de Treinamento e Avaliação ---
for name, model in models.items():
    print(f"\n{'='*20} Avaliando Modelo: {name} {'='*20}")
    
    # Pré-processamento específico do ID3
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    if name == "ID3":
        # ID3 requer atributos categóricos, então discretizamos a idade
        age_bins = [0, 30, 45, 100]
        age_labels = ['Jovem', 'Adulto', 'Idoso']
        X_train_processed['Idade'] = pd.cut(X_train_processed['Idade'], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)
        X_test_processed['Idade'] = pd.cut(X_test_processed['Idade'], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)

    # Treinamento
    model.fit(X_train_processed, y_train)
    
    # Predição
    y_pred = model.predict(X_test_processed)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"\nRelatório de Classificação - {name}:\n{report}")
    print(f"Acurácia - {name}: {accuracy:.2%}")
    
    # Geração de Gráficos
    cm_path = plot_confusion_matrix(y_test, y_pred, class_names, name)
    tree_path = save_tree_visualization(model, class_names, name)
    
    results[name] = {
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "accuracy": accuracy,
        "cm_path": cm_path,
        "tree_path": tree_path
    }

print(f"\n{'='*50}\nAVALIAÇÃO CONCLUÍDA\n{'='*50}")
