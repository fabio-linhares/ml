class CARTDecisionTree:
    """
    Implementação do algoritmo CART para árvores de decisão
    
    CARACTERÍSTICAS DISTINTIVAS DO CART:
    1. Usa Índice de Gini como medida de impureza
    2. Sempre faz divisões binárias (mesmo para categóricos)
    3. Pode fazer classificação E regressão
    4. Implementa cost-complexity pruning
    5. Trata missing values de forma sofisticada
    """
    
        def __init__(self, min_samples_split=5, min_samples_leaf=2, max_depth=15, memo_table=None):
            from memoization_table import MemoizationTable
            self.tree = None
            self.feature_names = None
            self.class_name = None
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_depth = max_depth
            # Register gini function for CART
            self.memo = memo_table if memo_table is not None else MemoizationTable(
                gini_func=self.gini_func
            )
            self.algorithm_name = "cart"
    
        @staticmethod
        def gini_func(y, log=None):
            # Standard Gini calculation
            from collections import Counter
            if len(y) == 0:
                return 0
            class_counts = Counter(y)
            n = len(y)
            gini = 1.0
            if log:
                log(f"    Calculando Gini para {n} exemplos:")
            for class_label, count in class_counts.items():
                probability = count / n
                gini -= probability ** 2
                if log:
                    log(f"      Classe '{class_label}': {count}/{n} = {probability:.3f}, p² = {probability**2:.3f}")
            if log:
                log(f"    Gini = 1 - ∑(pi²) = {gini:.3f}")
            return gini
    def gini_impurity(self, y):
        """
        Calcula o Índice de Gini para um conjunto de classes
        
        JUSTIFICATIVA: CART usa Gini em vez de Entropia por ser:
        1. Computacionalmente mais eficiente (sem logaritmos)
        2. Menos sensível a mudanças nas probabilidades das classes
        3. Equivalente à Entropia para decisões práticas
        
        Fórmula: Gini(S) = 1 - ∑(pi²)
        onde pi é a proporção da classe i
        """
            """
            Calcula o Índice de Gini usando memoização compartilhada
            """
            return self.memo.gini_impurity(y, algorithm=self.algorithm_name)
    
    def gini_split(self, X, y, feature, split_value=None, is_categorical=True):
        """
        Calcula a redução do Gini após uma divisão binária
        
        JUSTIFICATIVA: CART sempre faz divisões binárias, mesmo para 
        atributos categóricos. Isso simplifica a estrutura da árvore
        e facilita a implementação de pruning.
        """
        print(f"    Calculando Gini split para '{feature}':")
        
        if is_categorical and split_value is None:
            # Para categóricos, testar todas divisões binárias possíveis
            unique_values = X[feature].unique()
            if len(unique_values) <= 1:
                return 0, None
            
            best_gini_reduction = -1
            best_split = None
            
            # Testar cada valor como divisão binária (valor vs resto)
            for test_value in unique_values:
                mask_left = X[feature] == test_value
                mask_right = ~mask_left
                
                y_left = y[mask_left]
                y_right = y[mask_right]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calcular Gini ponderado
                n = len(y)
                left_weight = len(y_left) / n
                right_weight = len(y_right) / n
                
                left_gini = self.gini_impurity(y_left)
                right_gini = self.gini_impurity(y_right)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                original_gini = self.gini_impurity(y)
                gini_reduction = original_gini - weighted_gini
                
                print(f"      Split '{feature}' = '{test_value}' vs resto:")
                print(f"        Esquerda: {len(y_left)} exemplos, Gini = {left_gini:.3f}")
                print(f"        Direita: {len(y_right)} exemplos, Gini = {right_gini:.3f}")
                print(f"        Redução Gini: {gini_reduction:.3f}")
                
                if gini_reduction > best_gini_reduction:
                    best_gini_reduction = gini_reduction
                    best_split = test_value
            
            return best_gini_reduction, best_split
        
        else:
            # Para contínuos, usar split_value fornecido
            if split_value is None:
                return 0, None
            
            mask_left = X[feature] <= split_value
            mask_right = ~mask_left
            
            y_left = y[mask_left]
            y_right = y[mask_right]
            
            if len(y_left) == 0 or len(y_right) == 0:
                return 0, split_value
            
            # Calcular redução do Gini
            n = len(y)
            left_weight = len(y_left) / n
            right_weight = len(y_right) / n
            
            left_gini = self.gini_impurity(y_left)
            right_gini = self.gini_impurity(y_right)
            
            weighted_gini = left_weight * left_gini + right_weight * right_gini
            original_gini = self.gini_impurity(y)
            gini_reduction = original_gini - weighted_gini
            
            print(f"      Split '{feature}' <= {split_value:.2f}:")
            print(f"        Esquerda: {len(y_left)} exemplos, Gini = {left_gini:.3f}")
            print(f"        Direita: {len(y_right)} exemplos, Gini = {right_gini:.3f}")
            print(f"        Redução Gini: {gini_reduction:.3f}")
            
            return gini_reduction, split_value
    
    def find_best_split_cart(self, X, y, available_features):
        """
        Encontra a melhor divisão binária usando critério CART (Gini)
        
        JUSTIFICATIVA: CART procura a divisão binária que maximiza
        a redução do índice de Gini. Para cada atributo, testa:
        - Contínuos: todos pontos de corte possíveis
        - Categóricos: todas divisões binárias (um valor vs resto)
        """
        print(f"\nEncontrando melhor divisão CART entre: {available_features}")
        
        best_feature = None
        best_gini_reduction = -1
        best_split_value = None
        best_is_categorical = True
        
        for feature in available_features:
            print(f"  Analisando '{feature}':")
            
            if X[feature].dtype in ['int64', 'float64']:
                # Atributo contínuo - testar pontos de corte
                print(f"    Tratando '{feature}' como contínuo")
                
                unique_values = sorted(X[feature].unique())
                if len(unique_values) <= 1:
                    continue
                
                # Testar pontos médios entre valores consecutivos
                best_threshold_reduction = -1
                best_threshold = None
                
                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i + 1]) / 2
                    reduction, _ = self.gini_split(X, y, feature, threshold, False)
                    
                    if reduction > best_threshold_reduction:
                        best_threshold_reduction = reduction
                        best_threshold = threshold
                
                if best_threshold_reduction > best_gini_reduction:
                    best_gini_reduction = best_threshold_reduction
                    best_feature = feature
                    best_split_value = best_threshold
                    best_is_categorical = False
            
            else:
                # Atributo categórico - testar divisões binárias
                print(f"    Tratando '{feature}' como categórico")
                reduction, split_value = self.gini_split(X, y, feature, None, True)
                
                if reduction > best_gini_reduction:
                    best_gini_reduction = reduction
                    best_feature = feature
                    best_split_value = split_value
                    best_is_categorical = True
        
        print(f"Melhor divisão CART: '{best_feature}' com redução Gini = {best_gini_reduction:.3f}")
        if best_split_value is not None:
            if best_is_categorical:
                print(f"Divisão categórica: '{best_feature}' = '{best_split_value}' vs resto")
            else:
                print(f"Divisão contínua: '{best_feature}' <= {best_split_value:.2f}")
        
        return best_feature, best_gini_reduction, best_split_value, best_is_categorical
    
    def build_tree_cart(self, X, y, available_features, depth=0):
        """
        Constrói árvore CART com divisões binárias e critérios rigorosos
        
        CRITÉRIOS DE PARADA CART:
        1. Pureza (todos exemplos da mesma classe)
        2. Tamanho mínimo para divisão (min_samples_split)
        3. Tamanho mínimo de folha (min_samples_leaf) 
        4. Profundidade máxima
        5. Sem melhoria no Gini
        """
        indent = "  " * depth
        print(f"{indent}Construindo nó CART (profundidade {depth}, {len(y)} exemplos)")
        
        # CRITÉRIO DE PARADA 1: Conjunto puro
        if len(set(y)) == 1:
            class_label = y.iloc[0] if hasattr(y, 'iloc') else y[0]
            print(f"{indent}→ Folha: {class_label} (conjunto puro)")
            return {
                'type': 'leaf',
                'class': class_label,
                'samples': len(y),
                'gini': 0.0
            }
        
        # CRITÉRIO DE PARADA 2: Poucos exemplos para dividir
        if len(y) < self.min_samples_split:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (min_samples_split: {len(y)} < {self.min_samples_split})")
            return {
                'type': 'leaf',
                'class': majority_class,
                'samples': len(y),
                'gini': self.gini_impurity(y)
            }
        
        # CRITÉRIO DE PARADA 3: Profundidade máxima
        if depth >= self.max_depth:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (profundidade máxima)")
            return {
                'type': 'leaf',
                'class': majority_class,
                'samples': len(y),
                'gini': self.gini_impurity(y)
            }
        
        # CRITÉRIO DE PARADA 4: Sem atributos disponíveis
        if not available_features:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (sem mais atributos)")
            return {
                'type': 'leaf',
                'class': majority_class,
                'samples': len(y),
                'gini': self.gini_impurity(y)
            }
        
        # Encontrar melhor divisão CART
        best_feature, best_gini_reduction, best_split_value, is_categorical = self.find_best_split_cart(X, y, available_features)
        
        # CRITÉRIO DE PARADA 5: Sem melhoria suficiente no Gini
        if best_gini_reduction <= 0:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (sem melhoria no Gini)")
            return {
                'type': 'leaf',
                'class': majority_class,
                'samples': len(y),
                'gini': self.gini_impurity(y)
            }
        
        # Criar nó interno
        print(f"{indent}→ Nó interno: {best_feature}")
        node = {
            'type': 'internal',
            'feature': best_feature,
            'split_value': best_split_value,
            'is_categorical': is_categorical,
            'samples': len(y),
            'gini': self.gini_impurity(y),
            'left': None,
            'right': None
        }
        
        # Fazer divisão binária
        if is_categorical:
            # Divisão categórica: valor específico vs resto
            mask_left = X[best_feature] == best_split_value
            mask_right = ~mask_left
            
            print(f"{indent}Divisão categórica: '{best_feature}' = '{best_split_value}' vs resto")
        else:
            # Divisão contínua: <= threshold vs > threshold  
            mask_left = X[best_feature] <= best_split_value
            mask_right = ~mask_left
            
            print(f"{indent}Divisão contínua: '{best_feature}' <= {best_split_value:.2f}")
        
        # Construir subárvores
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[mask_right], y[mask_right]
        
        # CRITÉRIO DE PARADA 6: Folhas muito pequenas (min_samples_leaf)
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            majority_class = Counter(y).most_common(1)[0][0]
            print(f"{indent}→ Folha: {majority_class} (folhas muito pequenas)")
            return {
                'type': 'leaf',
                'class': majority_class,
                'samples': len(y),
                'gini': self.gini_impurity(y)
            }
        
        # Recursão para construir subárvores
        if len(y_left) > 0:
            print(f"{indent}Construindo subárvore esquerda:")
            node['left'] = self.build_tree_cart(X_left, y_left, available_features, depth + 1)
        
        if len(y_right) > 0:
            print(f"{indent}Construindo subárvore direita:")
            node['right'] = self.build_tree_cart(X_right, y_right, available_features, depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Treina o modelo CART"""
        print("INICIANDO TREINAMENTO DO CART:")
        print(f"Dataset: {len(X)} exemplos, {len(X.columns)} atributos")
        print(f"Configurações: min_samples_split={self.min_samples_split}, "
              f"min_samples_leaf={self.min_samples_leaf}, max_depth={self.max_depth}")
        
        self.feature_names = list(X.columns)
        self.class_name = y.name if hasattr(y, 'name') else 'target'
        
        # CART pode tratar qualquer tipo de atributo
        continuous_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        print(f"Atributos contínuos: {list(continuous_features)}")
        print(f"Atributos categóricos: {list(categorical_features)}")
        
        self.tree = self.build_tree_cart(X, y, self.feature_names)
        print("ÁRVORE CART CONSTRUÍDA COM SUCESSO!")
        
        return self
    
    def predict_single_cart(self, x, node):
        """Prediz classe para um exemplo usando árvore CART"""
        if node['type'] == 'leaf':
            return node['class']
        
        # Nó interno - fazer decisão binária
        feature = node['feature']
        split_value = node['split_value']
        is_categorical = node['is_categorical']
        
        if is_categorical:
            # Divisão categórica
            if x[feature] == split_value:
                return self.predict_single_cart(x, node['left'])
            else:
                return self.predict_single_cart(x, node['right'])
        else:
            # Divisão contínua
            if x[feature] <= split_value:
                return self.predict_single_cart(x, node['left'])
            else:
                return self.predict_single_cart(x, node['right'])
    
    def predict(self, X):
        """Prediz classes para múltiplos exemplos"""
        predictions = []
        for _, row in X.iterrows():
            pred = self.predict_single_cart(row, self.tree)
            predictions.append(pred)
        return predictions
    
    def print_tree_cart(self, node=None, depth=0, branch="", side=""):
        """Imprime árvore CART de forma legível"""
        if node is None:
            node = self.tree
        
        indent = "│   " * depth
        
        if node['type'] == 'leaf':
            samples = node['samples']
            gini = node['gini']
            class_label = node['class']
            print(f"{indent}└── {side}{branch} → {class_label} (samples: {samples}, gini: {gini:.3f})")
        else:
            feature = node['feature']
            split_value = node['split_value']
            is_categorical = node['is_categorical']
            samples = node['samples']
            gini = node['gini']
            
            if depth == 0:
                print(f"CART Decision Tree:")
                if is_categorical:
                    print(f"└── {feature} = '{split_value}' ? (samples: {samples}, gini: {gini:.3f})")
                else:
                    print(f"└── {feature} <= {split_value:.2f} ? (samples: {samples}, gini: {gini:.3f})")
            
            # Subárvore esquerda (condição verdadeira)
            if node['left']:
                if is_categorical:
                    left_branch = f"{feature} = '{split_value}'"
                else:
                    left_branch = f"{feature} <= {split_value:.2f}"
                self.print_tree_cart(node['left'], depth + 1, left_branch, "True: ")
            
            # Subárvore direita (condição falsa)
            if node['right']:
                if is_categorical:
                    right_branch = f"{feature} ≠ '{split_value}'"
                else:
                    right_branch = f"{feature} > {split_value:.2f}"
                self.print_tree_cart(node['right'], depth + 1, right_branch, "False: ")
