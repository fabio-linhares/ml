import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import altair as alt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Configurações Iniciais ---
warnings.filterwarnings("ignore", category=UserWarning, message=".*Signature.*")
st.set_page_config(layout="wide", page_title="Análise de Algoritmos de ML")

# --- Importação dos Modelos e Análises ---
from class_id3 import ID3DecisionTree
from class_c45 import C45DecisionTree
from class_cart import CARTDecisionTree
from heart_disease_ripper import analyze_heart_disease
from knn_classifier import KNNClassifier

# --- Funções Auxiliares ---
def preprocess_for_id3(df_features: pd.DataFrame) -> pd.DataFrame:
    """Discretiza colunas comprovadamente numéricas para o ID3."""
    df_processed = df_features.copy()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols:
        st.sidebar.info(f"Colunas numéricas discretizadas para ID3: {', '.join(numeric_cols)}")
        for col in numeric_cols:
            try:
                df_processed[col] = pd.qcut(df_processed[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop').astype(str)
            except ValueError: # Se não conseguir dividir em 4 quantis
                df_processed[col] = pd.cut(df_processed[col], bins=4, labels=['B1', 'B2', 'B3', 'B4'], duplicates='drop').astype(str)
    
    return df_processed

@st.cache_resource
def get_model(algorithm):
    """Carrega a classe do modelo com base na seleção."""
    if algorithm == 'ID3':
        return ID3DecisionTree(max_depth=st.session_state.get('max_depth', 5))
    elif algorithm == 'C4.5':
        return C45DecisionTree(max_depth=st.session_state.get('max_depth', 5))
    elif algorithm == 'CART':
        return CARTDecisionTree(max_depth=st.session_state.get('max_depth', 5))
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
st.title("🌳 Análise Interativa de Algoritmos de Machine Learning")
st.markdown("Selecione uma análise na barra lateral, configure os parâmetros e avalie os resultados.")

# --- Sidebar (Controles) ---
with st.sidebar:
    st.header("⚙️ Configurações de Análise")
    
    analysis_type = st.radio(
        "1. Escolha o tipo de análise:",
        ('Árvores de Decisão (Questão 1 & 2)', 'Análise de Doenças Cardíacas (Questão 3)')
    )

    if analysis_type == 'Árvores de Decisão (Questão 1 & 2)':
        uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"], key="tree_uploader")
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            st.session_state.df = df
            algorithm = st.selectbox("2. Escolha o algoritmo:", ('ID3', 'C4.5', 'CART'))
            all_columns = df.columns.tolist()
            target_column = st.selectbox("3. Selecione a coluna Alvo:", all_columns, index=len(all_columns)-1)
            feature_columns = st.multiselect("4. Selecione os Atributos:",
                                             [col for col in all_columns if col != target_column],
                                             default=[col for col in all_columns if col != target_column])
            st.header("🔬 Validação")
            test_size = st.slider("5. Percentual para Teste:", 0.1, 0.5, 0.25, 0.05)
            st.header("🔧 Hiperparâmetros")
            max_depth = st.slider("6. Profundidade Máxima da Árvore:", 2, 20, 5)
            
            if st.button("🚀 Treinar e Avaliar Modelo", use_container_width=True, type="primary"):
                st.session_state.run_training = 'tree'
                st.session_state.algorithm = algorithm
                st.session_state.target_column = target_column
                st.session_state.feature_columns = feature_columns
                st.session_state.test_size = test_size
                st.session_state.max_depth = max_depth

    elif analysis_type == 'Análise de Doenças Cardíacas (Questão 3)':
        st.info("Esta análise usa o dataset 'Heart Disease UCI'. Certifique-se de que o arquivo `heart.csv` está no diretório do projeto.")
        if st.button("🔍 Executar Análise (RIPPER)", use_container_width=True, type="primary"):
            st.session_state.run_training = 'ripper'

# --- Lógica para Árvores de Decisão ---
if st.session_state.get('run_training') == 'tree':
    df = st.session_state.df
    algorithm = st.session_state.algorithm
    target_column = st.session_state.target_column
    feature_columns = st.session_state.feature_columns
    test_size = st.session_state.test_size
    max_depth = st.session_state.max_depth

    model = get_model(algorithm)

    if model and feature_columns and target_column:
        X_raw = df[feature_columns]
        y_raw = df[target_column]

        X_processed = preprocess_for_id3(X_raw) if algorithm == 'ID3' else X_raw

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_raw, test_size=test_size, random_state=42, stratify=y_raw)
        
        with st.expander("👁️ Visualizar Dados de Treino (Pré-processados)"):
            st.dataframe(X_train.head(10))

        with st.spinner(f"Treinando modelo {algorithm}..."):
            model.fit(X_train, y_train)
        
        st.success("Modelo treinado e avaliado com sucesso!")
        st.header("📈 Resultados da Avaliação (Árvore de Decisão)")

        y_pred = model.predict(X_test)
        class_names = sorted(y_raw.unique())
        
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Acurácia", f"{accuracy:.2%}")
            st.write("**Relatório de Classificação**")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
            st.dataframe(report_df)
        with col2:
            st.write("**Matriz de Confusão**")
            fig_cm = plot_confusion_matrix(y_test, y_pred, class_names)
            st.pyplot(fig_cm)
            
        st.write("**Visualização da Árvore de Decisão**")
        try:
            dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': 'TB'})
            model.model._get_tree_graph(dot, model.get_tree_structure(), class_names=class_names)
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"Não foi possível gerar o gráfico da árvore: {e}")
            st.json(model.get_tree_structure())
        
    st.session_state.run_training = None

# --- Lógica para Análise de Doenças Cardíacas (RIPPER) ---
if st.session_state.get('run_training') == 'ripper':
    st.header("🩺 Resultados da Análise de Doenças Cardíacas (Questão 3)")
    with st.spinner("Analisando dataset com Árvore de Decisão e RIPPER..."):
        results, message = analyze_heart_disease()
    
    if results:
        st.success(message)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌳 Árvore de Decisão (Scikit-learn)")
            st.metric("Acurácia", f"{results['decision_tree']['accuracy']:.2%}")
            st.write("**Relatório de Classificação**")
            report_df = pd.DataFrame(results['decision_tree']['report']).transpose()
            st.dataframe(report_df)

        with col2:
            st.subheader("📜 Regras Diretas (RIPPER)")
            st.metric("Acurácia", f"{results['ripper']['accuracy']:.2%}")
            st.write("**Conjunto de Regras Gerado**")
            st.code(results['ripper']['ruleset'], language='text')
            
    else:
        st.error(message)
    
    st.session_state.run_training = None


# --- NOVA SEÇÃO: Análise de Overfitting/Underfitting com kNN (Questões 5 & 6) ---
with st.expander("🔬 Análise de Overfitting/Underfitting com kNN (Questões 5 & 6)"):
    st.markdown("""
    Esta seção permite analisar como o valor de **k** no kNN afeta o desempenho do modelo,
    ilustrando os conceitos de **overfitting** e **underfitting**.
    - **k pequeno (ex: 1):** O modelo pode se tornar muito sensível ao ruído nos dados de treino (**overfitting**).
    - **k grande:** O modelo pode se tornar muito genérico, perdendo a capacidade de capturar a estrutura dos dados (**underfitting**).
    
    Usaremos o dataset de doenças cardíacas para esta análise.
    """)

    max_k = st.slider("Selecione o valor máximo de k para testar:", 5, 50, 25, key="max_k_slider")
    
    if st.button("Executar Análise de Variação do k", use_container_width=True):
        try:
            df_heart = pd.read_csv('heart.csv')
            # Label Encoding para todas as colunas, pois RIPPER e Árvores lidam bem com isso
            le = LabelEncoder()
            df_encoded = df_heart.apply(le.fit_transform)
            X = df_encoded.drop('target', axis=1)
            y = df_encoded['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            accuracies = []
            k_values = range(1, max_k + 1)
            
            with st.spinner("Testando diferentes valores de k..."):
                for k in k_values:
                    knn = KNNClassifier(k=k)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)

            # Encontrar o melhor k
            best_k = k_values[np.argmax(accuracies)]
            best_acc = max(accuracies)

            st.success(f"Análise concluída! O melhor desempenho foi com **k={best_k}**, atingindo **{best_acc:.2%}** de acurácia.")

            # Criar gráfico de acurácia vs. k
            chart_data = pd.DataFrame({
                'k': k_values,
                'Acurácia': accuracies
            })
            
            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X('k', title='Valor de k'),
                y=alt.Y('Acurácia', scale=alt.Scale(zero=False)),
                tooltip=['k', 'Acurácia']
            ).properties(
                title='Desempenho do kNN vs. Valor de k'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)

        except FileNotFoundError:
            st.error("Arquivo 'heart.csv' não encontrado. Execute a análise da Questão 3 primeiro ou use o script 'download_heart_dataset.py'.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")


# --- Seção kNN (Exemplo Didático Original) ---
st.markdown("---")
st.header("🐾 Explorador k-Nearest Neighbors (kNN - Exemplo Didático)")

# Dados de exemplo para o kNN
X_train_knn = np.array([[50, 30], [80, 40], [90, 35], [40, 25], [85, 45], [60, 35]])
y_train_knn = np.array(['A', 'B', 'B', 'A', 'B', 'A'])
df_knn = pd.DataFrame(X_train_knn, columns=['Renda (k$)', 'Idade'])
df_knn['Classe'] = y_train_knn

col_knn1, col_knn2 = st.columns([1, 2])

with col_knn1:
    st.subheader("Configuração do Teste")
    k_val = st.slider("Selecione o valor de k:", 1, 5, 3, 1)
    
    st.write("**Novo Ponto para Classificar:**")
    new_x = st.number_input("Renda (k$):", min_value=0, max_value=150, value=70)
    new_y = st.number_input("Idade:", min_value=18, max_value=70, value=30)
    new_point = np.array([new_x, new_y])

    # Classificar o novo ponto
    knn = KNNClassifier(k=k_val)
    knn.fit(X_train_knn, y_train_knn)
    prediction = knn.predict([new_point])[0]
    neighbors = knn.get_neighbors(new_point)
    
    st.success(f"**Classe Prevista para o ponto ({new_x}, {new_y}):** `{prediction}`")
    
    st.write("**Vizinhos Mais Próximos:**")
    neighbors_df = pd.DataFrame(neighbors)
    neighbors_df['point'] = neighbors_df['point'].apply(lambda p: f"({p[0]}, {p[1]})")
    st.dataframe(neighbors_df[['point', 'label', 'distance']].rename(columns={
        'point': 'Ponto', 'label': 'Classe', 'distance': 'Distância'
    }))

with col_knn2:
    st.subheader("Visualização dos Dados")
    
    # Criar DataFrame para o novo ponto e vizinhos
    df_new = pd.DataFrame([new_point], columns=['Renda (k$)', 'Idade'])
    df_new['Classe'] = 'Novo'
    
    neighbor_points = np.array([n['point'] for n in neighbors])
    df_neighbors = pd.DataFrame(neighbor_points, columns=['Renda (k$)', 'Idade'])
    df_neighbors['Classe'] = 'Vizinho'

    # Gráfico base
    base_chart = alt.Chart(df_knn).mark_circle(size=100).encode(
        x=alt.X('Renda (k$)', scale=alt.Scale(domain=[20, 100])),
        y=alt.Y('Idade', scale=alt.Scale(domain=[20, 50])),
        color=alt.Color('Classe', scale=alt.Scale(domain=['A', 'B', 'Novo', 'Vizinho'], range=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'])),
        tooltip=['Renda (k$)', 'Idade', 'Classe']
    ).interactive()

    # Camada para o novo ponto
    new_point_chart = alt.Chart(df_new).mark_circle(size=200, opacity=0.8).encode(
        x='Renda (k$)', y='Idade', color='Classe'
    )
    
    # Camada para os vizinhos
    neighbors_chart = alt.Chart(df_neighbors).mark_square(size=150, opacity=0.8, filled=False, strokeWidth=2).encode(
        x='Renda (k$)', y='Idade', color='Classe'
    )
    
    # Combinar camadas
    final_chart = base_chart + new_point_chart + neighbors_chart
    st.altair_chart(final_chart, use_container_width=True)