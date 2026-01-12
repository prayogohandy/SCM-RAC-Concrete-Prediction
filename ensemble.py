from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold
from functions import process_array
from collections import OrderedDict
from config import input_cols

# Regression
class L0BaggingModelRegressor:
    def __init__(self, base_model, score_func, n_splits=5, random_state=42,
                 scaler=None, feature_info=None, feature_mask=None):
        self.base_model = base_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.score_func = score_func
        self.models = []
        self.oof_preds = None
        self.oof_score = None
        self.test_preds = None
        self.test_score = None
        self.scaler = scaler
        self.feature_info = feature_info  # Feature information for scaling
        self.feature_mask = feature_mask  # Mask to skip features if needed
        self.scaler_info = []  # Store scaler information if needed

    def _clone_model(self):
        try:
            return clone(self.base_model)
        except Exception:
            return deepcopy(self.base_model)
    
    def fit(self, X_train, y_train, X_test, y_test):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_preds = np.zeros(len(X_train))
        self.test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val = X_train[val_idx]
            X_te = X_test

            # Apply scaler if specified
            if self.scaler is not None and self.feature_info is not None:
                X_tr, infos = process_array(X_tr, scaler=self.scaler, feature_info=self.feature_info, existing_info=None)
                X_val, _ = process_array(X_val, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                X_te, _ = process_array(X_test, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                self.scaler_info.append(infos)
            else:
                self.scaler_info.append(None)

            # Apply feature mask if provided
            if self.feature_mask is not None:
                X_tr = X_tr[:, self.feature_mask]
                X_val = X_val[:, self.feature_mask]
                X_te = X_te[:, self.feature_mask]

            model = self._clone_model()
            model.fit(X_tr, y_tr)
            self.models.append(model)

            self.oof_preds[val_idx] = model.predict(X_val)
            self.test_preds += model.predict(X_te) / self.n_splits
            
        if self.score_func:
            self.oof_score = self.score_func(y_train, self.oof_preds)
            self.test_score = self.score_func(y_test, self.test_preds)
        return self

    def predict(self, X_original):
        if not self.models:
            raise ValueError("No models trained. Call `.fit()` first.")
        preds = []
        for i, model in enumerate(self.models):
            if self.scaler is not None and self.feature_info is not None and self.scaler_info[i] is not None:
                X, _ = process_array(X_original, scaler=self.scaler, feature_info=self.feature_info, existing_info=self.scaler_info[i])
            else:
                X = X_original
            if self.feature_mask is not None:
                X = X[:, self.feature_mask]
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

    def strip_model(self):
        self.models = []  # remove trained models
        return self

def get_name(model):
    name = model.__class__.__name__
    model_abbreviations = {
        "RandomForestRegressor": "RF",
        "ExtraTreesRegressor": "ET",
        "GradientBoostingRegressor": "GB",
        "DecisionTreeRegressor": "DT",
        "KNeighborsRegressor": "KNN",
        "SVR": "SVR",
        "XGBRegressor": "XGB",
        "LGBMRegressor": "LGB",
        "CatBoostRegressor": "CB",
        "MLPRegressor": "MLP",
        "AdaBoostRegressor": "ADA",
        "HistGradientBoostingRegressor": "HGB",
        "GaussianProcessRegressor": "GPR",
        "Ridge": "Ridge",
    }
    return model_abbreviations.get(name, name)

class HybridChainPredictor:
    """
    Hybrid predictor with:
    - A chained subset of outputs (information transfer)
    - Independent standalone outputs
    """

    def __init__(self, chain_models, standalone_models=None, no_fe_outputs=[]):
        if not isinstance(chain_models, OrderedDict):
            raise TypeError("chain_models must be an OrderedDict in chaining order.")
        if len(chain_models) == 0:
            raise ValueError("chain_models cannot be empty.")

        self.chain_models = chain_models
        self.standalone_models = standalone_models or {}
        self.no_fe_outputs = no_fe_outputs

    def predict(self, X):
        # Preserve index if DataFrame
        if isinstance(X, pd.DataFrame):
            index = X.index
            X_aug = X.values
        else:
            index = None
            X_aug = np.asarray(X)

        preds = {}

        # ---- Chained predictions ----
        for out, model in self.chain_models.items():
            y_pred = np.asarray(model.predict(X_aug)).reshape(-1)
            preds[out] = y_pred
            X_aug = np.hstack([X_aug, y_pred[:, None]])

        # ---- Standalone predictions ----
        for out, model in self.standalone_models.items():
            if out in self.no_fe_outputs:
                # Predict without augmented features
                num_cols = len(input_cols)
                preds[out] = np.asarray(model.predict(X[:,:num_cols])).reshape(-1) # only take first num_cols columns
            else:
                # Predict with augmented features
                preds[out] = np.asarray(model.predict(X)).reshape(-1)

        return preds