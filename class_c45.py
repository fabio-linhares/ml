class C45DecisionTree:
    """
    Implementação do algoritmo C4.5 para árvores de decisão
    
    MELHORIAS DO C4.5 EM RELAÇÃO AO ID3:
    1. Usa Gain Ratio em vez de Information Gain (evita bias para atributos com muitos valores)
    2. Trata atributos contínuos (discretização automática)
    3. Trata valores ausentes (missing values)
    4. Implementa poda pós-processamento (pruning)
    """
    
    def __init__(self, min_samples_split=5, confidence_threshold=0.25, memo_table=None):
        from memoization_table import MemoizationTable
        self.tree = None
        self.feature_names = None
        self.class_name = None
        self.min_samples_split = min_samples_split  # Mínimo para dividir
        self.confidence_threshold = confidence_threshold  # Para poda
        self.memo = memo_table if memo_table is not None else MemoizationTable()
    
    def entropia(self, y):
        """Calcula entropia usando memoização compartilhada"""
        return self.memo.entropia(y)
    
    def split_information(self, X, feature):
        """
        Calcula Split Information usando memoização compartilhada
        """
        return self.memo.split_information(X, feature, algorithm=self.algorithm_name)
    
    def gain_ratio(self, X, y, feature):
        """
        Calcula Gain Ratio usando memoização compartilhada
        """
        entropy_before = self.entropia(y)
        gain_ratio = self.memo.gain_ratio(X, y, feature, entropy_before=entropy_before, algorithm=self.algorithm_name)
        # For compatibility, also return information gain and split info
        information_gain = self.memo.information_gain(X, y, feature, entropy_before=entropy_before, algorithm=self.algorithm_name)
        split_info = self.memo.split_information(X, feature, algorithm=self.algorithm_name)
        return gain_ratio, information_gain, split_info
    
    def discretize_continuous(self, X, y, feature):
        """
        Discretiza atributo contínuo encontrando melhor ponto de corte
        
        JUSTIFICATIVA: C4.5 pode tratar atributos contínuos automaticamente.
        Testa pontos de corte entre valores ordenados e escolhe o melhor.
        """
        print(f"    Discretizando atributo contínuo '{feature}':")
        
        # Combinar X e y para ordenação
        combined = list(zip(X[feature], y))
        combined.sort(key=lambda x: x[0])  # Ordenar por valor do atributo
        
        best_gain = -1
        best_threshold = None
        best_left_entropy = None
        best_right_entropy = None
        
        # Testar pontos de corte entre valores diferentes
        for i in range(len(combined) - 1):
            current_val = combined[i][0]
            next_val = combined[i + 1][0]
            
            if current_val != next_val:  # Só testar quando valores são diferentes
                threshold = (current_val + next_val) / 2
                
                # Dividir dados pelo threshold
                left_y = [item[1] for item in combined[:i+1]]
                right_y = [item[1] for item in combined[i+1:]]
                
                # Calcular entropia ponderada
                n = len(combined)
                left_weight = len(left_y) / n
                right_weight = len(right_y) / n
                
                left_entropy = self.entropia(left_y) if left_y else 0
                right_entropy = self.entropia(right_y) if right_y else 0
                
                weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
                original_entropy = self.entropia([item[1] for item in combined])
                
                gain = original_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_left_entropy = left_entropy
                    best_right_entropy = right_entropy
        
        print(f"    Melhor threshold: {best_threshold:.2f} (gain: {best_gain:.3f})")
        return best_threshold, best_gain
    
    def find_best_split_c45(self, X, y, available_features):
        """
        Encontra melhor divisão usando Gain Ratio (característica do C4.5)
        
        JUSTIFICATIVA: C4.5 usa Gain Ratio em vez de Information Gain puro
        para evitar bias towards atributos com muitos valores únicos.
        """
        print(f"\nEncontrando melhor divisão C4.5 entre: {available_features}")
        
        best_feature = None
        best_gain_ratio = -1
        best_threshold = None
        
        for feature in available_features:
            # Verificar se atributo é contínuo (numérico)
            if X[feature].dtype in ['int64', 'float64']:
                print(f"  Tratando '{feature}' como contínuo:")
                threshold, gain = self.discretize_continuous(X, y, feature)
                
                # Para contínuos, usar Information Gain diretamente 
                # (Split Information seria sempre log2(2)=1 para divisão binária)
                gain_ratio = gain
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold
            else:
                # Atributo categórico - usar Gain Ratio normal
                gain_ratio, info_gain, split_info = self.gain_ratio(X, y, feature)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = None
        
        print(f"Melhor atributo C4.5: '{best_feature}' com gain ratio = {best_gain_ratio:.3f}")
        if best_threshold:
            print(f"Threshold para '{best_feature}': {best_threshold:.2f}")
        
        return best_feature, best_gain_ratio, best_threshold
    
    def build_tree_c45(self, X, y, available_features, depth=0, max_depth=15):
        """
        Constrói árvore C4.5 com melhorias sobre ID3
        
        MELHORIAS IMPLEMENTADAS:
        1. Gain Ratio em vez de Information Gain
        2. Tratamento de atributos contínuos
        3. Critério de parada baseado em tamanho mínimo
        4. Estrutura preparada para poda
        """
        indent = "  " * depth
        print(f"{indent}Construindo nó C4.5 (profundidade {depth}, {len(y)} exemplos)")
        
        # CRITÉRIO DE PARADA 1: Conjunto puro
        if len(set(y)) == 1:
            class_label = y.iloc[0] if hasattr(y, 'iloc') else y[0]
            print(f"{indent}→ Folha: {class_label} (conjunto puro)")
            return class_label
        
        # CRITÉRIO DE PARADA 2: Muito poucos exemplos (prevenção overfitting)
        if len(y) < self.min_samples_split:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (poucos exemplos: {len(y)})")
            return majority_class
        
        # CRITÉRIO DE PARADA 3: Sem atributos disponíveis
        if not available_features:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (sem mais atributos)")
            return majority_class
        
        # CRITÉRIO DE PARADA 4: Profundidade máxima
        if depth >= max_depth:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (profundidade máxima)")
            return majority_class
        
        # Encontrar melhor divisão usando C4.5
        best_feature, best_gain_ratio, threshold = self.find_best_split_c45(X, y, available_features)
        
        # Se não há ganho suficiente, criar folha
        if best_gain_ratio <= 0:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (sem ganho suficiente)")
            return majority_class
        
        # Criar nó interno
        print(f"{indent}→ Nó interno: {best_feature}")
        tree = {
            'feature': best_feature, 
            'children': {}, 
            'threshold': threshold,
            'samples': len(y),
            'entropy': self.entropia(y)
        }
        
        # Dividir dados
        if threshold is not None:
            # Divisão contínua (binária)
            print(f"{indent}Divisão contínua: {best_feature} <= {threshold:.2f}")
            
            mask_left = X[best_feature] <= threshold
            mask_right = X[best_feature] > threshold
            
            # Ramo esquerdo (<= threshold)
            X_left, y_left = X[mask_left], y[mask_left]
            if len(y_left) > 0:
                subtree_left = self.build_tree_c45(X_left, y_left, available_features, depth+1, max_depth)
                tree['children'][f"<= {threshold:.2f}"] = subtree_left
            
            # Ramo direito (> threshold)  
            X_right, y_right = X[mask_right], y[mask_right]
            if len(y_right) > 0:
                subtree_right = self.build_tree_c45(X_right, y_right, available_features, depth+1, max_depth)
                tree['children'][f"> {threshold:.2f}"] = subtree_right
        else:
            # Divisão categórica (múltipla)
            unique_values = X[best_feature].unique()
            remaining_features = [f for f in available_features if f != best_feature]
            
            for value in unique_values:
                print(f"{indent}Processando ramo '{best_feature}' = '{value}':")
                
                mask = X[best_feature] == value
                X_subset = X[mask]
                y_subset = y[mask]
                
                if len(y_subset) > 0:
                    subtree = self.build_tree_c45(X_subset, y_subset, remaining_features, depth+1, max_depth)
                    tree['children'][value] = subtree
        
        return tree
    
    def fit(self, X, y):
        """Treina o modelo C4.5"""
        print("INICIANDO TREINAMENTO DO C4.5:")
        print(f"Dataset: {len(X)} exemplos, {len(X.columns)} atributos")
        print(f"Configurações: min_samples_split={self.min_samples_split}")
        
        self.feature_names = list(X.columns)
        self.class_name = y.name if hasattr(y, 'name') else 'target'
        
        # C4.5 pode tratar tanto categóricos quanto contínuos
        continuous_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        print(f"Atributos contínuos: {list(continuous_features)}")
        print(f"Atributos categóricos: {list(categorical_features)}")
        
        self.tree = self.build_tree_c45(X, y, self.feature_names)
        print("ÁRVORE C4.5 CONSTRUÍDA COM SUCESSO!")
        
        return self
    
    def predict_single_c45(self, x, tree):
        """Prediz classe para um exemplo usando árvore C4.5"""
        if isinstance(tree, str):  # É uma folha
            return tree
        
        feature = tree['feature']
        value = x[feature]
        threshold = tree.get('threshold')
        
        if threshold is not None:
            # Divisão contínua
            if value <= threshold:
                key = f"<= {threshold:.2f}"
            else:
                key = f"> {threshold:.2f}"
        else:
            # Divisão categórica
            key = value
        
        if key in tree['children']:
            return self.predict_single_c45(x, tree['children'][key])
        else:
            # Valor não visto - retornar classe conservadora
            return "Baixo"
    
    def predict(self, X):
        """Prediz classes para múltiplos exemplos"""
        predictions = []
        for _, row in X.iterrows():
            pred = self.predict_single_c45(row, self.tree)
            predictions.append(pred)
        return predictions
    
    def print_tree_c45(self, tree=None, depth=0, branch=""):
        """Imprime árvore C4.5 de forma legível"""
        if tree is None:
            tree = self.tree
        
        indent = "│   " * depth
        
        if isinstance(tree, str):  # É uma folha
            print(f"{indent}└── {branch} → {tree}")
        else:
            feature = tree['feature']
            samples = tree.get('samples', '?')
            entropy = tree.get('entropy', 0)
            
            if depth == 0:
                print(f"C4.5 Decision Tree:")
                print(f"└── {feature} (samples: {samples}, entropy: {entropy:.3f})")
            
            for i, (value, subtree) in enumerate(tree['children'].items()):
                is_last = i == len(tree['children']) - 1
                branch_symbol = "└──" if is_last else "├──"
                next_indent = "    " if is_last else "│   "
                
                if isinstance(subtree, str):
                    print(f"{indent}{next_indent}{branch_symbol} {value} → {subtree}")
                else:
                    print(f"{indent}{next_indent}{branch_symbol} {value}")
                    self.print_tree_c45(subtree, depth+1)
