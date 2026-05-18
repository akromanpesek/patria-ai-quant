import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class TradingModel:
    def __init__(self, name, model_type, **kwargs):
        self.name = name
        self.model_type = model_type
        
        # Inicializace 5 různých architektur
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(**kwargs)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(**kwargs)
        elif model_type == "SVM":
            kwargs.pop('probability', None)
            self.model = SVC(probability=True, **kwargs)
        elif model_type == "KNN":
            self.model = KNeighborsClassifier(**kwargs)
        else:
            raise ValueError("Neznámý typ modelu")
            
        self.is_trained = False
        
    def train(self, X, y):
        """Trénink modelu na historických datech"""
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, X):
        """Vrací 1 (Koupit/Držet) nebo 0 (Prodat/Nekoupit)"""
        if not self.is_trained:
            raise Exception("Model ještě nebyl natrénován!")
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

def create_initial_population():
    """Vytvoří výchozích 5 modelů do turnaje."""
    models = [
        TradingModel("Alpha_LogReg", "LogisticRegression", C=1.0, max_iter=1000),
        TradingModel("Beta_RF", "RandomForest", n_estimators=50, max_depth=5),
        TradingModel("Gamma_GB", "GradientBoosting", n_estimators=50, learning_rate=0.1),
        TradingModel("Delta_SVM", "SVM", C=1.0, kernel='rbf'),
        TradingModel("Epsilon_KNN", "KNN", n_neighbors=5)
    ]
    return models
