import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score

# Importing local modules for feature engineering, labeling, and configuration
from src.features import Features
from src.labeling import Labeler
from src.config import (
    TEST_SIZE, VALIDATION_SPLITS,
    DEFAULT_P_ENTRY, DEFAULT_P_EXIT,
    DEFAULT_RSI_BUY_MAX, DEFAULT_RSI_BUY_MIN,
    DEFAULT_RSI_SELL_MIN, DEFAULT_RSI_EXIT_BULL, DEFAULT_RSI_EXIT_BEAR
)


class TradingStrategy:
    """
    Main class that holds the XGBoost model and the specific rules for the strategy.
    We save this whole object using joblib to make sure the strategy works the same way
    when we use it later as it did during training.
    """

    # Columns we want to remove before training.
    # We remove raw prices (Open, Close) so the model learns patterns, not specific prices.
    # We remove 'Target' because that is the answer we are trying to guess.
    COLS_TO_DROP = [
        'Target', 'Open', 'High', 'Low', 'Close', 'Volume',
        'VIX_Close', 'VIX3M_Close', 'VIX_Term_Structure'
    ]

    def __init__(self):
        # The machine learning model (starts empty)
        self.model = None
        self.feature_names = None

        # Strategy settings (loaded from config)
        # These control when we actually buy or sell based on the model's signal
        self.p_entry = DEFAULT_P_ENTRY
        self.p_exit = DEFAULT_P_EXIT
        self.rsi_buy_max = DEFAULT_RSI_BUY_MAX
        self.rsi_buy_min = DEFAULT_RSI_BUY_MIN
        self.rsi_sell_min = DEFAULT_RSI_SELL_MIN
        self.rsi_exit_bull = DEFAULT_RSI_EXIT_BULL
        self.rsi_exit_bear = DEFAULT_RSI_EXIT_BEAR

        # A dictionary to store how well the model performed during training
        self.train_metrics = {}

    def _prepare_features(self, df):
        """
        Cleans the data before feeding it to the model.
        It removes the 'target' (answer) and raw price columns.
        """
        cols_to_drop = [c for c in self.COLS_TO_DROP if c in df.columns]
        # Drop columns, ignore error if a column is missing
        X = df.drop(columns=cols_to_drop, errors='ignore')
        return X

    def fit(self, df):
        """
        The main function to start training the model.

        Args:
            df: The data containing market features and the 'Target'.
            use_walk_forward: If True, use a rolling window training (more realistic).
                              If False, use a simple split (faster).
        """
        # Prepare the input features (X) and the target/answer (y)
        X = self._prepare_features(df)
        y = df['Target']

        # Save the names of the features for future reference
        self.feature_names = list(X.columns)

        # Choose the training method

        self._train_walk_forward(X, y)


        return self

    def _train_walk_forward(self, X, y):
        """
        Walk-Forward Validation: This simulates real life.
        We train on a chunk of past data, test on the immediate future,
        then slide the window forward and repeat.
        """
        # Split time-series data into chunks
        tscv = TimeSeriesSplit(n_splits=VALIDATION_SPLITS)

        scores = []
        f1_scores = []

        print(f"\nWalk-Forward Validation ({VALIDATION_SPLITS} splits)...")

        # Loop through each time chunk
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Balance weights for this specific chunk
            weights = compute_sample_weight(class_weight='balanced', y=y_train)

            # Create a temporary model for this chunk
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=30
            )

            # Train on the current chunk
            model.fit(
                X_train, y_train,
                sample_weight=weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Score this chunk
            score = model.score(X_val, y_val)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')

            scores.append(score)
            f1_scores.append(f1)
            print(f"   Fold {fold + 1}: Accuracy={score:.4f}, F1={f1:.4f}")

        # IMPORTANT: After validating that the strategy works on sliding windows,
        # we train the FINAL model using ALL available data.
        weights = compute_sample_weight(class_weight='balanced', y=y)

        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

        # No 'eval_set' here because we are using all data to train for production
        self.model.fit(X, y, sample_weight=weights, verbose=False)

        # Save average performance metrics
        self.train_metrics['cv_scores'] = scores
        self.train_metrics['cv_f1_scores'] = f1_scores
        self.train_metrics['mean_cv_score'] = np.mean(scores)
        self.train_metrics['mean_cv_f1'] = np.mean(f1_scores)

        print(f"\nResults Walk-Forward:")
        print(f"Mean Accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        print(f"Mean F1 Macro: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")

    def predict_proba(self, df):
        """Returns the probability (confidence) for each class (Buy, Sell, Hold)."""
        X = self._prepare_features(df)
        return self.model.predict_proba(X)

    def predict(self, df):
        """Returns the single most likely class."""
        X = self._prepare_features(df)
        return self.model.predict(X)

    def get_feature_importance(self, top_n=20):
        """
        Shows which data points (indicators) were most useful for the model.
        Useful for debugging or understanding the strategy.
        """
        if self.model is None:
            return None

        importance = self.model.feature_importances_
        # Sort indices from highest importance to lowest
        indices = np.argsort(importance)[::-1][:top_n]

        print(f"\nTop {top_n} Features:")
        for i, idx in enumerate(indices):
            print(f"   {i + 1}. {self.feature_names[idx]}: {importance[idx]:.4f}")

        return [(self.feature_names[i], importance[i]) for i in indices]

    def set_strategy_params(self, **params):
        """Allows updating strategy settings manually."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, path):
        """Saves the entire object (model + parameters) to a file."""
        joblib.dump(self, path)
        print(f"Model saved at: {path}")

    @classmethod
    def load(cls, path):
        """Loads a saved model from a file."""
        return joblib.load(path)


# Compatibility Wrapper
class Model:
    """
    It basically wraps the new 'TradingStrategy' logic.
    """

    def __init__(self, df):
        self.df = df
        self.strategy = TradingStrategy()

    def get_data(self):
        """Generates features and labels the data."""
        gen = Features(self.df)
        self.df = gen.apply_all()

        lab = Labeler()
        self.df = lab.create_target(self.df)
        return self.df

    def train(self, use_walk_forward=True):
        """Trains the internal strategy."""
        self.strategy.fit(self.df)
        self.strategy.get_feature_importance(15)

    def save(self, path='models/sniper_v7.pkl'):
        """Saves using the strategy's save method."""
        self.strategy.save(path)