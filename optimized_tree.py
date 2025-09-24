import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd


# =====================================================================
# 1. ESTRUTURAS DE DADOS E CONFIGURAÇÃO
# =====================================================================

class AlgorithmType(Enum):
    """Enumeração para identificar tipos de algoritmo."""
    ID3 = "id3"
    C45 = "c45"
    CART = "cart"

@dataclass(frozen=True, eq=True)
class ComputationKey:
    """Chave imutável e hashable para memoização."""
    data_hash: int
    feature: Optional[str] = None
    computation_type: str = "default"
    algorithm: Optional[AlgorithmType] = None


# =====================================================================
# 2. SISTEMA DE MEMOIZAÇÃO
# =====================================================================

class AdvancedMemoizationTable:
    """Tabela de memoização otimizada com múltiplas estratégias de cache."""

    def __init__(self, max_cache_size=20000):
        self.caches = {alg: {} for alg in AlgorithmType}
        self.caches['shared'] = {}
        self.metrics = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
        }

    def _hash_data(self, data: Union[pd.Series, np.ndarray]) -> int:
        """Cria hash eficiente para dados pandas/numpy."""
        if isinstance(data, pd.Series):
            return int(pd.util.hash_pandas_object(data, index=True).sum())
        elif isinstance(data, np.ndarray):
            return hash(data.tobytes())
        else:
            return hash(tuple(data))

    def get_cached_result(self, key: ComputationKey) -> Optional[Any]:
        """Recupera resultado do cache."""
        cache = self.caches.get(key.algorithm, self.caches['shared'])
        result = cache.get(key)
        if result is not None:
            self.metrics['hits'][key.algorithm or 'shared'] += 1
            return result
        
        self.metrics['misses'][key.algorithm or 'shared'] += 1
        return None

    def cache_result(self, key: ComputationKey, result: Any):
        """Armazena resultado no cache."""
        cache = self.caches.get(key.algorithm, self.caches['shared'])
        cache[key] = result

    def _calculate_metric(self, y: pd.Series, metric_func: Callable, computation_type: str, algorithm: AlgorithmType) -> float:
        """Função genérica para calcular e cachear uma métrica (Entropia, Gini)."""
        if y.empty:
            return 0.0

        key = ComputationKey(
            data_hash=self._hash_data(y),
            computation_type=computation_type,
            algorithm=algorithm
        )
        
        cached = self.get_cached_result(key)
        if cached is not None:
            return cached
        
        result = metric_func(y)
        self.cache_result(key, result)
        return result

    def entropy(self, y: pd.Series, algorithm: AlgorithmType) -> float:
        """Calcula a Entropia de Shannon com memoização."""
        def _entropy_calc(data: pd.Series) -> float:
            counts = data.value_counts()
            probs = counts / len(data)
            probs = probs[probs > 0] # Evita log(0)
            return -np.sum(probs * np.log2(probs))
        
        return self._calculate_metric(y, _entropy_calc, "entropy", algorithm)

    def gini_impurity(self, y: pd.Series, algorithm: AlgorithmType) -> float:
        """Calcula a Impureza de Gini com memoização."""
        def _gini_calc(data: pd.Series) -> float:
            counts = data.value_counts()
            probs = counts / len(data)
            return 1 - np.sum(probs**2)

        return self._calculate_metric(y, _gini_calc, "gini", algorithm)


# =====================================================================
# 3. CONSTRUTOR DA ÁRVORE OTIMIZADO
# =====================================================================

class OptimizedTreeBuilder:
    """Constrói árvores com memoização e suporte a dados contínuos."""
    
    def __init__(self, algorithm_type: AlgorithmType, memo_table: AdvancedMemoizationTable):
        self.algorithm_type = algorithm_type
        self.memo_table = memo_table
        
        if self.algorithm_type == AlgorithmType.ID3:
            self.metric_func = self.memo_table.entropy
            self.split_eval_func = self._evaluate_split_gain
        elif self.algorithm_type == AlgorithmType.C45:
            self.metric_func = self.memo_table.entropy
            self.split_eval_func = self._evaluate_split_gain_ratio
        elif self.algorithm_type == AlgorithmType.CART:
            self.metric_func = self.memo_table.gini_impurity
            self.split_eval_func = self._evaluate_split_gini_reduction
        else:
            raise ValueError("Algoritmo desconhecido")

    def _find_best_continuous_split(self, X: pd.DataFrame, y: pd.Series, feature: str, parent_metric: float) -> tuple[float, Optional[float]]:
        best_metric_value = -1.0
        best_threshold = None
        df = pd.DataFrame({'feature': X[feature], 'target': y}).sort_values(by='feature')
        unique_values = df['feature'].unique()
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i+1]) / 2.0
            left_mask = df['feature'] <= threshold
            y_left, y_right = df['target'][left_mask], df['target'][~left_mask]
            
            if y_left.empty or y_right.empty:
                continue
            metric_value = self.split_eval_func(parent_metric, y, [y_left, y_right])
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
        return best_metric_value, best_threshold

    def _evaluate_split_gain(self, parent_metric: float, y: pd.Series, subsets_y: list) -> float:
        n = len(y)
        if n == 0: return 0.0
        weighted_child_metric = sum((len(subset) / n) * self.metric_func(subset, self.algorithm_type) for subset in subsets_y)
        return parent_metric - weighted_child_metric
    
    def _evaluate_split_gain_ratio(self, parent_metric: float, y: pd.Series, subsets_y: list) -> float:
        gain = self._evaluate_split_gain(parent_metric, y, subsets_y)
        if gain == 0: return 0.0
        n = len(y)
        if n == 0: return 0.0
        props = [len(s) / n for s in subsets_y if len(s) > 0]
        split_info = -sum(p * math.log2(p) for p in props)
        return gain / split_info if split_info > 0 else 0.0

    def _evaluate_split_gini_reduction(self, parent_metric: float, y: pd.Series, subsets_y: list) -> float:
        return self._evaluate_split_gain(parent_metric, y, subsets_y)

    def find_best_split(self, X: pd.DataFrame, y: pd.Series, features: list) -> Optional[Dict[str, Any]]:
        best_split = {'metric': -1.0, 'feature': None, 'threshold': None}
        parent_metric = self.metric_func(y, self.algorithm_type)
        for feature in features:
            current_metric, current_threshold = -1.0, None
            if pd.api.types.is_numeric_dtype(X[feature].dtype) and self.algorithm_type in [AlgorithmType.C45, AlgorithmType.CART]:
                current_metric, current_threshold = self._find_best_continuous_split(X, y, feature, parent_metric)
            else:
                subsets_y = [y[X[feature] == val] for val in X[feature].unique()]
                current_metric = self.split_eval_func(parent_metric, y, subsets_y)
            if current_metric > best_split['metric']:
                best_split = {'metric': current_metric, 'feature': feature, 'threshold': current_threshold}
        return best_split if best_split['metric'] > 0 else None

    def build_tree(self, X: pd.DataFrame, y: pd.Series, features: list, depth: int = 0, max_depth: int = 10) -> Optional[dict]:
        if y.empty:
            return None
        class_distribution = y.value_counts().to_dict()
        parent_majority_class = y.mode()[0]
        if len(y.unique()) == 1 or not features or depth >= max_depth:
            return {'leaf_value': parent_majority_class, 'samples': len(y), 'value': class_distribution}
        best_split = self.find_best_split(X, y, features)
        if not best_split:
            return {'leaf_value': parent_majority_class, 'samples': len(y), 'value': class_distribution}
        feature = best_split['feature']
        threshold = best_split.get('threshold')
        remaining_features = [f for f in features if f != feature]
        tree = {
            'feature': feature, 'threshold': threshold if threshold is not None else 'Categórico',
            'samples': len(y), 'value': class_distribution, 'children': {}
        }
        if threshold is not None:
            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold
            left_subtree = self.build_tree(X.loc[left_mask], y.loc[left_mask], remaining_features, depth + 1, max_depth)
            tree['children']['<='] = left_subtree or {'leaf_value': parent_majority_class, 'samples': 0, 'value': {}}
            right_subtree = self.build_tree(X.loc[right_mask], y.loc[right_mask], remaining_features, depth + 1, max_depth)
            tree['children']['>'] = right_subtree or {'leaf_value': parent_majority_class, 'samples': 0, 'value': {}}
        else:
            for value in X[feature].unique():
                mask = X[feature] == value
                subtree = self.build_tree(X.loc[mask], y.loc[mask], remaining_features, depth + 1, max_depth)
                tree['children'][value] = subtree or {'leaf_value': parent_majority_class, 'samples': 0, 'value': {}}
        return tree


# =====================================================================
# 4. API PRINCIPAL UNIFICADA
# =====================================================================

class OptimizedDecisionTree:
    """Classe unificada para treinar e usar árvores de decisão otimizadas."""

    def __init__(self, algorithm: str = 'cart', max_depth: int = 10):
        try:
            self.algorithm_type = AlgorithmType(algorithm.lower())
        except ValueError:
            raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use 'id3', 'c45', ou 'cart'.")
        self.max_depth = max_depth
        self.memo_table = AdvancedMemoizationTable()
        self.tree_builder = OptimizedTreeBuilder(self.algorithm_type, self.memo_table)
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Treina o modelo de árvore de decisão."""
        print(f"🚀 INICIANDO TREINAMENTO ({self.algorithm_type.name})...")
        start_time = time.perf_counter()
        self.tree = self.tree_builder.build_tree(X, y, list(X.columns), 0, self.max_depth)
        end_time = time.perf_counter()
        print(f"   ⚡ Treinamento concluído em {end_time - start_time:.3f}s")
        metrics = self.memo_table.metrics
        total_hits = sum(metrics['hits'].values())
        total_misses = sum(metrics['misses'].values())
        total_calls = total_hits + total_misses
        hit_rate = total_hits / total_calls if total_calls > 0 else 0
        print(f"   📈 Cache Hit Rate: {hit_rate:.1%}")
        print(f"   💾 Cache Hits: {total_hits}, Misses: {total_misses}")
        return self

    def _predict_single(self, x: pd.Series, node: Union[str, dict]) -> str:
        """Navega na árvore para prever um único exemplo."""
        if not isinstance(node, dict) or 'leaf_value' in node:
            return node['leaf_value'] if isinstance(node, dict) else node
        
        feature = node['feature']
        threshold = node['threshold']
        
        if threshold != 'Categórico':
            key = '<=' if x[feature] <= threshold else '>'
            child_node = node['children'].get(key)
        else:
            value = x[feature]
            child_node = node['children'].get(value)

        if child_node is None:
            # Fallback: retorna a classe majoritária do nó atual se o caminho não for encontrado
            return max(node['value'], key=node['value'].get)

        return self._predict_single(x, child_node)

    def predict(self, X: pd.DataFrame) -> list:
        """Faz predições para um conjunto de dados X."""
        if self.tree is None:
            raise RuntimeError("O modelo deve ser treinado com .fit() antes de prever.")
        return [self._predict_single(row, self.tree) for _, row in X.iterrows()]

    def _get_tree_graph(self, dot, node, class_names, parent_name=None, edge_label=""):
        """Função recursiva para construir o gráfico da árvore com Graphviz (versão melhorada)."""
        if not isinstance(node, dict) or 'leaf_value' in node:
            leaf_value = node['leaf_value'] if isinstance(node, dict) else node
            samples = node.get('samples', '?')
            value_dict = node.get('value', {})
            value_str = '\\n'.join([f"{c}: {v}" for c, v in value_dict.items()])
            label = f"Classe: {leaf_value}\\nsamples = {samples}\\nvalue =\\n{value_str}"
            node_name = f"leaf_{id(node)}"
            color = "#a0d1e6"
            try:
                if leaf_value == class_names[0]: color = "#f2c2a0"
                if len(class_names) > 1 and leaf_value == class_names[1]: color = "#a0e6b8"
                if len(class_names) > 2 and leaf_value == class_names[2]: color = "#d3a0e6"
            except IndexError:
                pass # Mantém a cor padrão se houver mais classes que cores
            dot.node(node_name, label, shape='ellipse', style='filled', fillcolor=color)
            if parent_name:
                dot.edge(parent_name, node_name, label=str(edge_label))
            return

        feature, threshold = node['feature'], node['threshold']
        samples, value_dict = node.get('samples', '?'), node.get('value', {})
        value_str = '\\n'.join([f"{c}: {v}" for c, v in value_dict.items()])
        node_name = f"node_{id(node)}"
        label = f"{feature}"
        if threshold != 'Categórico':
            label += f" <= {threshold:.2f}"
        label += f"\\nsamples = {samples}\\nvalue =\\n{value_str}"
        dot.node(node_name, label, shape='box', style='filled', fillcolor='lightgrey')
        if parent_name:
            dot.edge(parent_name, node_name, label=str(edge_label))
        for edge, child_node in node['children'].items():
            self._get_tree_graph(dot, child_node, class_names, parent_name=node_name, edge_label=str(edge))