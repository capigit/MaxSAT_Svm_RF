import numpy as np
from sklearn.datasets import load_breast_cancer

def get_figure1_toy_data():
    X = np.array([
        [0, 0, 0, 1, 1],  
        [0, 0, 0, 0, 1],  
        [1, 1, 1, 1, 0],
        [1, 0, 1, 0, 0], 
        [1, 0, 1, 1, 1] 
    ])
    y = np.array([1, 1, 1, 0, 0])
    return X, y

def get_synthetic_data(n_samples=150, n_features=15, random_state=42):
    np.random.seed(random_state)
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    # La classe dépend uniquement des 3 premières caractéristiques
    y = (np.sum(X[:, 0:3], axis=1) >= 2).astype(int)
    return X, y

def get_sklearn_breast_cancer_binarized():
    X, y = load_breast_cancer(return_X_y=True)
    # Binarisation simple : 1 si > moyenne, 0 sinon
    X_bin = (X > X.mean(axis=0)).astype(int)
    return X_bin, y