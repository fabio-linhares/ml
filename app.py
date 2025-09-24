import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import warnings

# --- Configurações Iniciais ---
warnings.filterwarnings("ignore", category=UserWarning, message="Signature.*")
st.set_page_config(layout="wide", page_title="Construtor de Árvores de Decisão")

# --- Importação dos Modelos ---
from class_id3 import ID3DecisionTree
# from class_cart import CARTDecisionTree # Próximo passo

# --- Funções Auxiliares ---
def preprocess_for_id3(df_features: pd.DataFrame) -> pd.DataFrame:
    """Discretiza colunas comprovadamente numéricas para o ID3."""
    df_processed = df_features.copy()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols:
        st.sidebar.info(f"Colunas numéricas discretizadas: {', '.join(numeric_cols)}")
        for col in numeric_cols:
            df_processed[col] = pd.qcut(df_processed[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop').astype(str)
    
    return df_processed

@st.cache_resource
def get_model(algorithm):
    """Carrega a classe do modelo com base na seleção."""
    if algorithm == 'ID3':
        return ID3DecisionTree()
    # Adicione CART aqui depois
    return None

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Cria e exibe um gráfico da matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Matriz de Confusão', fontsize=16)
    ax.set_xlabel('Valores Preditos', fontsize=12)
    ax.set_ylabel('Valores Reais', fontsize=12)
    return fig

# --- Interface Principal ---
st.title("🌳 Construtor Interativo de Árvores de Decisão")
st.markdown("Faça o upload de um CSV, configure os parâmetros e avalie o desempenho do seu modelo.")

# --- Sidebar (Controles) ---
with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_file = st.file_uploader("1. Carregue seu arquivo CSV", type=["csv"])
    
    # Se um arquivo for carregado, mostre as outras opções
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        algorithm = st.selectbox("2. Escolha o algoritmo:", ('ID3', 'CART (em breve)'))
        
        all_columns = df.columns.tolist()
        target_column = st.selectbox("3. Selecione a coluna Alvo:", all_columns, index=len(all_columns)-1)
        
        feature_columns = st.multiselect("4. Selecione os Atributos:",
                                         [col for col in all_columns if col != target_column],
                                         default=[col for col in all_columns if col != target_column])

        st.header("🔬 Validação")
        test_size = st.slider("5. Percentual para Teste:", 0.1, 0.5, 0.25, 0.05)
        
        st.header("🔧 Hiperparâmetros")
        max_depth = st.slider("6. Profundidade Máxima da Árvore:", 2, 20, 5)

        # O botão de treino agora aciona diretamente a lógica principal
        if st.button("🚀 Treinar e Avaliar Modelo", use_container_width=True, type="primary"):
            st.session_state.run_training = True # Usamos o session_state para "lembrar" do clique
            # Armazenamos todas as configurações para usar na área principal
            st.session_state.df = df
            st.session_state.algorithm = algorithm
            st.session_state.target_column = target_column
            st.session_state.feature_columns = feature_columns
            st.session_state.test_size = test_size
            st.session_state.max_depth = max_depth

# --- Lógica principal movida para fora da sidebar ---
# Só executa se o botão foi clicado
if 'run_training' in st.session_state and st.session_state.run_training:
    
    # Recupera as configurações
    df = st.session_state.df
    algorithm = st.session_state.algorithm
    target_column = st.session_state.target_column
    feature_columns = st.session_state.feature_columns
    test_size = st.session_state.test_size
    max_depth = st.session_state.max_depth

    model = get_model(algorithm)
    model.max_depth = max_depth

    if model and feature_columns and target_column:
        X_raw = df[feature_columns]
        y_raw = df[target_column]

        if algorithm == 'ID3':
            X_processed = preprocess_for_id3(X_raw)
        else:
            X_processed = X_raw

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_raw, test_size=test_size, random_state=42, stratify=y_raw)
        
        with st.expander("👁️ Visualizar Dados de Treino (Pré-processados)"):
            st.dataframe(X_train.head(10))

        with st.spinner(f"Treinando modelo {algorithm}..."):
            model.fit(X_train, y_train)
        
        st.success("Modelo treinado e avaliado com sucesso!")
        st.header("📈 Resultados da Avaliação")

        y_pred = model.predict(X_test)
        class_names = sorted(y_raw.unique())
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("📊 Métricas de Desempenho (em dados de teste)")
        # ... (código das métricas e gráficos permanece o mesmo)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Acurácia", f"{accuracy:.2%}")
        col2.metric("Precisão", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")

        st.subheader("🔍 Análise Detalhada")
        col_cm, col_tree = st.columns([1, 2])
        with col_cm:
            st.write("**Matriz de Confusão**")
            fig_cm = plot_confusion_matrix(y_test, y_pred, class_names)
            st.pyplot(fig_cm)
        with col_tree:
            st.write("**Visualização da Árvore de Decisão**")
            try:
                dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': 'LR'})
                model.model._get_tree_graph(dot, model.get_tree_structure(), class_names=class_names)
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"Não foi possível gerar o gráfico da árvore: {e}")
                st.json(model.get_tree_structure())
        
    st.session_state.run_training = False # Reseta o estado para a próxima execução

elif not uploaded_file:
    st.info("Aguarde o carregamento de um arquivo CSV na barra lateral para começar.")