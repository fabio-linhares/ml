#!/usr/bin/env python3
"""
Script para baixar e preparar o dataset Heart Disease UCI
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings('ignore')

def download_heart_disease_dataset():
    """
    Baixa o dataset Heart Disease UCI e salva como heart.csv
    """
    print("Baixando dataset Heart Disease UCI...")
    
    try:
        # Tentar baixar do OpenML
        heart = fetch_openml(name='heart', version=1, as_frame=True, parser='auto')
        df = heart.frame
        
        # Renomear a coluna target se necessário
        if 'class' in df.columns:
            df = df.rename(columns={'class': 'target'})
        
        # Converter target para binário (0: sem doença, 1: com doença)
        if df['target'].dtype == 'object':
            df['target'] = df['target'].map({'1': 0, '2': 1, '3': 1, '4': 1, '5': 1})
        else:
            df['target'] = (df['target'] > 0).astype(int)
            
    except Exception as e:
        print(f"Erro ao baixar do OpenML: {e}")
        print("Criando dataset sintético baseado no Heart Disease UCI...")
        
        # Criar dataset sintético similar ao Heart Disease UCI
        np.random.seed(42)
        n_samples = 303
        
        df = pd.DataFrame({
            'age': np.random.randint(29, 78, n_samples),
            'sex': np.random.choice([0, 1], n_samples, p=[0.32, 0.68]),
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08]),
            'trestbps': np.random.normal(131, 17, n_samples).astype(int),
            'chol': np.random.normal(246, 52, n_samples).astype(int),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.48, 0.03]),
            'thalach': np.random.normal(150, 23, n_samples).astype(int),
            'exang': np.random.choice([0, 1], n_samples, p=[0.67, 0.33]),
            'oldpeak': np.random.uniform(0, 6.2, n_samples).round(1),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.46, 0.33]),
            'ca': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.58, 0.22, 0.12, 0.06, 0.02]),
            'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.04, 0.38, 0.56])
        })
        
        # Criar target com alguma correlação com as features
        risk_score = (
            (df['age'] > 55).astype(int) * 0.3 +
            (df['sex'] == 1).astype(int) * 0.2 +
            (df['cp'] > 0).astype(int) * 0.3 +
            (df['trestbps'] > 140).astype(int) * 0.1 +
            (df['chol'] > 240).astype(int) * 0.1 +
            (df['exang'] == 1).astype(int) * 0.2 +
            (df['oldpeak'] > 1).astype(int) * 0.2 +
            (df['ca'] > 0).astype(int) * 0.3 +
            np.random.uniform(-0.2, 0.2, n_samples)
        )
        
        df['target'] = (risk_score > 0.5).astype(int)
    
    # Garantir que os valores estejam nos ranges corretos
    df['age'] = df['age'].clip(29, 77)
    df['trestbps'] = df['trestbps'].clip(94, 200)
    df['chol'] = df['chol'].clip(126, 564)
    df['thalach'] = df['thalach'].clip(71, 202)
    df['oldpeak'] = df['oldpeak'].clip(0, 6.2)
    
    # Salvar o dataset
    df.to_csv('heart.csv', index=False)
    print(f"Dataset salvo como 'heart.csv' com {len(df)} amostras")
    print(f"Distribuição do target: {df['target'].value_counts().to_dict()}")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    download_heart_disease_dataset()