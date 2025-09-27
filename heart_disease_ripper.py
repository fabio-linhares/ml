import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import wittgenstein as lw

def analyze_heart_disease(data_path='heart.csv'):
    """
    Carrega o dataset Heart Disease, treina uma Árvore de Decisão e um
    modelo RIPPER, e retorna os resultados.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        return None, "Erro: Arquivo 'heart.csv' não encontrado. Por favor, baixe o dataset e coloque-o no diretório do projeto."

    # Pré-processamento simples
    # Label Encoding para todas as colunas, pois RIPPER e Árvores lidam bem com isso
    df_encoded = df.apply(LabelEncoder().fit_transform)
    
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 1. Árvore de Decisão (Scikit-learn)
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, y_pred_tree)
    tree_report = classification_report(y_test, y_pred_tree, output_dict=True)

    # 2. RIPPER (Wittgenstein)
    ripper_model = lw.RIPPER(random_state=42)
    ripper_model.fit(X_train, y_train, feature_names=X.columns.tolist())
    y_pred_ripper = ripper_model.predict(X_test)
    ripper_accuracy = accuracy_score(y_test, y_pred_ripper)
    ripper_report = classification_report(y_test, y_pred_ripper, output_dict=True)
    
    results = {
        "decision_tree": {
            "accuracy": tree_accuracy,
            "report": tree_report,
            "model": tree_model
        },
        "ripper": {
            "accuracy": ripper_accuracy,
            "report": ripper_report,
            "ruleset": str(ripper_model.ruleset_)
        }
    }
    
    return results, "Análise concluída com sucesso."

if __name__ == '__main__':
    # Exemplo de como usar a função
    analysis_results, message = analyze_heart_disease()
    if analysis_results:
        print("--- Resultados da Árvore de Decisão ---")
        print(f"Acurácia: {analysis_results['decision_tree']['accuracy']:.2%}")
        print(classification_report(
            pd.Series(analysis_results['decision_tree']['report']['y_true']),
            pd.Series(analysis_results['decision_tree']['report']['y_pred'])
        ))
        
        print("\n--- Resultados do RIPPER ---")
        print(f"Acurácia: {analysis_results['ripper']['accuracy']:.2%}")
        print(f"Regras Geradas:\n{analysis_results['ripper']['ruleset']}")
    else:
        print(message)
