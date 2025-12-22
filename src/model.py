# src/model.py
"""Model training module using XGBoost."""
from features import Features
from labeling import Labeler
from config import TEST_SIZE, XGBOOST_PARAMS
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


class Model:
    """XGBoost classifier for market direction prediction."""
    
    def __init__(self, df):
        self.df = df
        self.model = None

    def get_data(self):
        """Apply feature engineering and labeling."""
        gen = Features(self.df)
        self.df = gen.apply_all()

        lab = Labeler()
        self.df = lab.create_target(self.df)
        return self.df

    def train(self):
        """Train the XGBoost model."""
        cols_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close', 'TNX_Close']
        X = self.df.drop(columns=[c for c in cols_drop if c in self.df.columns])
        y = self.df['Target']

        # Temporal split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        # Class weights for balancing
        weights = compute_sample_weight(class_weight='balanced', y=y_train)

        print(f"Training XGBoost with {len(X_train)} samples...")

        self.model = xgb.XGBClassifier(
            n_estimators=XGBOOST_PARAMS.get('n_estimators', 1000),
            learning_rate=XGBOOST_PARAMS.get('learning_rate', 0.01),
            max_depth=XGBOOST_PARAMS.get('max_depth', 4),
            subsample=XGBOOST_PARAMS.get('subsample', 0.7),
            colsample_bytree=XGBOOST_PARAMS.get('colsample_bytree', 0.7),
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=XGBOOST_PARAMS.get('random_state', 42),
            n_jobs=-1,
            early_stopping_rounds=XGBOOST_PARAMS.get('early_stopping_rounds', 50)
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print(f"Model trained. Test Score: {self.model.score(X_test, y_test):.4f}")

    def save(self, path='models/sniper_v6.pkl'):
        """Save trained model to disk."""
        joblib.dump(self.model, path)