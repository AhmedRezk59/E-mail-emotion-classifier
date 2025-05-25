import os
import pandas as pd
import joblib

class PersistenceManager:
    
    def __init__(self):
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'data.csv'))
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl'))
        if not os.path.exists(model_path):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')), exist_ok=True)
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl'))

    def load(self):
        df = pd.read_csv(self.path)
        return df

    def save_model(self, model):
        joblib.dump(model, self.model_path)