import numpy as np
from collections import Counter

class KNNClassifier:
    """
    Implementação didática do algoritmo k-Nearest Neighbors (kNN).
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Armazena os dados de treinamento."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        """Calcula a distância euclidiana entre dois pontos."""
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        """
        Prediz as classes para um conjunto de dados de teste.
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Prediz a classe para um único ponto de dados.
        """
        # 1. Calcular as distâncias para todos os pontos de treino
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Obter os k vizinhos mais próximos (índices e distâncias)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Obter os rótulos dos k vizinhos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Votação: retornar a classe mais comum entre os vizinhos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def get_neighbors(self, x):
        """
        Retorna os k vizinhos mais próximos para um ponto de dado,
        junto com suas distâncias e rótulos.
        """
        x = np.array(x)
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        
        neighbors = []
        for i in k_indices:
            neighbors.append({
                'point': self.X_train[i],
                'label': self.y_train[i],
                'distance': distances[i]
            })
        return neighbors

if __name__ == '__main__':
    # Exemplo de uso
    # Dados de exemplo (Renda em k$, Idade) -> Classe (A=0, B=1)
    X_train_data = np.array([[50, 30], [80, 40], [90, 35], [40, 25], [85, 45], [60, 35]])
    y_train_data = np.array([0, 1, 1, 0, 1, 0]) # A, B, B, A, B, A
    
    # Ponto a ser classificado
    x_new = np.array([70, 30])
    
    print(f"Ponto a ser classificado: {x_new}")
    
    for k_val in [1, 3, 5]:
        knn = KNNClassifier(k=k_val)
        knn.fit(X_train_data, y_train_data)
        prediction = knn.predict([x_new])
        neighbors_info = knn.get_neighbors(x_new)
        
        print(f"\n--- Para k = {k_val} ---")
        print(f"Predição: Classe {'A' if prediction[0] == 0 else 'B'}")
        print("Vizinhos mais próximos:")
        for neighbor in neighbors_info:
            print(f"  Ponto: {neighbor['point']}, Classe: {'A' if neighbor['label'] == 0 else 'B'}, Distância: {neighbor['distance']:.2f}")
