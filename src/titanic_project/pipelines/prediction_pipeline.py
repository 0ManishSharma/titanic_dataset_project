import pickle
import pandas as pd
import numpy as np
import os

class TitanicPredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load with explicit binary mode"""
        artifacts_path = 'artifacts'
        
        # Verify files exist
        if not os.path.exists(os.path.join(artifacts_path, 'preprocessor.pkl')):
            raise FileNotFoundError("preprocessor.pkl not found in artifacts/")
        if not os.path.exists(os.path.join(artifacts_path, 'model.pkl')):
            raise FileNotFoundError("model.pkl not found in artifacts/")
        
        # Load with binary mode (fixes Unicode error)
        try:
            with open(os.path.join(artifacts_path, 'preprocessor.pkl'), 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            with open(os.path.join(artifacts_path, 'model.pkl'), 'rb') as f:
                model_bundle = pickle.load(f)
                self.model = model_bundle['model']
                
        except UnicodeDecodeError:
            # Windows UTF-8 fix
            import codecs
            with codecs.open(os.path.join(artifacts_path, 'preprocessor.pkl'), 'rb', 
                           encoding='utf-8', errors='replace') as f:
                self.preprocessor = pickle.load(f)
            with codecs.open(os.path.join(artifacts_path, 'model.pkl'), 'rb', 
                           encoding='utf-8', errors='replace') as f:
                model_bundle = pickle.load(f)
                self.model = model_bundle['model']
    
    def preprocess_features(self, features):
        data = pd.DataFrame([features])
        
        # Feature engineering
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        data['Fare'] = np.log1p(data['Fare'])
        
        data['AgeGroup'] = pd.cut(
            data['Age'], bins=[0,12,20,40,60,80],
            labels=['Child','Teen','Adult','MiddleAge','Senior']
        )
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        
        feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 
                       'Embarked', 'AgeGroup']
        return data[feature_cols]
    
    def predict(self, features):
        X = self.preprocess_features(features)
        X_transformed = self.preprocessor.transform(X)
        prediction = self.model.predict(X_transformed)[0]
        probabilities = self.model.predict_proba(X_transformed)[0]
        return {
            'Survived': int(prediction),
            'Survival_Probability': float(probabilities[1]),
            'Death_Probability': float(probabilities[0])
        }
