import pandas as pd
from optimized_tree import OptimizedDecisionTree

class ID3DecisionTree:
    """
    Classe adaptadora para o algoritmo ID3.
    
    Esta classe usa o motor OptimizedDecisionTree configurado especificamente
    para o comportamento do ID3 (usando Information Gain e lidando apenas 
    com atributos categóricos).
    """
    
    def __init__(self, max_depth=10):
        """
        Inicializa o modelo ID3 encapsulando o motor otimizado.
        """
        # Instancia o motor principal, travando o algoritmo em 'id3'
        self.model = OptimizedDecisionTree(algorithm='id3', max_depth=max_depth)
        self.feature_names_in_ = None
        self.target_name_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina o modelo ID3. Requer que os dados em X sejam categóricos.
        """
        print("INFO: Iniciando treinamento do modelo ID3...")
        
        # O motor OptimizedDecisionTree já sabe como lidar com o ID3
        self.model.fit(X, y)
        
        self.feature_names_in_ = list(X.columns)
        self.target_name_ = y.name
        
        print("INFO: Treinamento ID3 concluído.")
        return self

    def predict(self, X: pd.DataFrame) -> list:
        """
        Faz predições para novos dados.
        """
        if self.model.tree is None:
            raise RuntimeError("O modelo deve ser treinado com .fit() antes de prever.")
            
        # O motor já tem a lógica de predição
        return self.model.predict(X)
        
    def get_tree_structure(self) -> dict:
        """
        Retorna a árvore gerada como um dicionário para visualização.
        """
        return self.model.tree
        
    def get_performance_metrics(self) -> dict:
        """
        Retorna as métricas de performance do cache do motor.
        """
        metrics = self.model.memo_table.metrics
        total_hits = sum(metrics['hits'].values())
        total_misses = sum(metrics['misses'].values())
        total_calls = total_hits + total_misses
        hit_rate = total_hits / total_calls if total_calls > 0 else 0
        
        return {
            'cache_hit_rate': hit_rate,
            'cache_hits': total_hits,
            'cache_misses': total_misses
        }