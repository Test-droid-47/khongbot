import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class ExpertModel:
    
    def __init__(
        self,
        task_name: str,
        task_type: str,
        cfg: Dict = None,
        n_classes: int = 3
    ):
        self.task_name = task_name
        self.task_type = task_type
        self.n_classes = n_classes if task_type == 'classification' else 1
        self.cfg = cfg or {}
        self.scaler = RobustScaler()
        self.xgb_model = None
        self.lgb_model = None
        self._feat_cols: List[str] = []
        self._weights: Dict[str, float] = {'xgb': 0.5, 'lgb': 0.5}
        self._trained = False
        
        self.n_estimators = self.cfg.get('ensemble_n_estimators', 300)
        self.max_depth = self.cfg.get('ensemble_max_depth', 6)
        self.learning_rate = self.cfg.get('ensemble_learning_rate', 0.05)
        self.subsample = self.cfg.get('ensemble_subsample', 0.8)
        self.colsample = self.cfg.get('ensemble_colsample', 0.8)
        self.early_stop_rounds = self.cfg.get('ensemble_early_stop', 20)

    def _fit_scaler(self, X: np.ndarray) -> None:
        X_finite = np.where(np.isfinite(X), X, 0.0)
        self.scaler.fit(X_finite)

    def _fill_and_scale(self, X_raw: np.ndarray) -> Optional[np.ndarray]:
        nan_mask = ~np.isfinite(X_raw)
        X_finite = np.where(nan_mask, 0.0, X_raw)
        try:
            X_scaled = self.scaler.transform(X_finite)
            X_scaled[nan_mask] = np.nan
            return X_scaled
        except Exception as exc:
            print(f"Scaler transform failed for {self.task_name}: {exc}")
            return None

    def train_ensemble(self, X_tr_raw: np.ndarray, y_tr: np.ndarray, X_val_raw: np.ndarray, y_val: np.ndarray) -> None:
        if len(X_tr_raw) < 100:
            print(f"[{self.task_name}] Insufficient training samples ({len(X_tr_raw)}). Skipping.")
            return
            
        if X_val_raw is None or len(X_val_raw) == 0:
            print(f"[{self.task_name}] Empty validation set provided. Skipping.")
            return

        X_tr_raw = np.nan_to_num(X_tr_raw, nan=np.nan, posinf=np.nan, neginf=np.nan)
        X_val_raw = np.nan_to_num(X_val_raw, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        if X_tr_raw.ndim == 1:
            X_tr_raw = X_tr_raw.reshape(-1, 1)
        if X_val_raw.ndim == 1:
            X_val_raw = X_val_raw.reshape(-1, 1)

        X_tr_df = pd.DataFrame(X_tr_raw).ffill().values
        X_val_df = pd.DataFrame(X_val_raw).ffill().values

        self._fit_scaler(X_tr_df)
        X_tr = self._fill_and_scale(X_tr_df)
        X_val = self._fill_and_scale(X_val_df)

        if X_tr is None or X_val is None:
            print(f"[{self.task_name}] Scaling failed. Aborting.")
            return

        self._feat_cols = [f"feat_{i}" for i in range(X_tr.shape[1])]
        is_classification = self.task_type == 'classification'

        if XGB_AVAILABLE:
            try:
                if is_classification:
                    if self.n_classes == 2:
                        objective = 'binary:logistic'
                        eval_metric = 'logloss'
                    else:
                        objective = 'multi:softmax'
                        eval_metric = 'mlogloss'
                    model_class = xgb.XGBClassifier
                else:
                    objective = 'reg:squarederror'
                    eval_metric = 'mae'
                    model_class = xgb.XGBRegressor

                params = {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0,
                    'early_stopping_rounds': self.early_stop_rounds,
                }
                if is_classification:
                    params['eval_metric'] = eval_metric
                    if self.n_classes == 2:
                        neg = np.sum(y_tr == 0)
                        pos = np.sum(y_tr == 1)
                        params['scale_pos_weight'] = neg / max(pos, 1)
                else:
                    params['eval_metric'] = eval_metric

                self.xgb_model = model_class(**params)
                self.xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                if is_classification:
                    acc = (self.xgb_model.predict(X_val) == y_val).mean()
                    print(f"[{self.task_name}] XGB val accuracy: {acc:.4f}")
                else:
                    preds = self.xgb_model.predict(X_val)
                    mae = np.mean(np.abs(preds - y_val))
                    print(f"[{self.task_name}] XGB val MAE: {mae:.4f}")
            except Exception as exc:
                print(f"[{self.task_name}] XGB failed: {exc}")
                self.xgb_model = None

        if LGB_AVAILABLE:
            try:
                if is_classification:
                    if self.n_classes == 2:
                        objective = 'binary'
                        metric = 'binary_logloss'
                    else:
                        objective = 'multiclass'
                        metric = 'multi_logloss'
                    model_class = lgb.LGBMClassifier
                else:
                    objective = 'regression'
                    metric = 'mae'
                    model_class = lgb.LGBMRegressor

                params = {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': -1,
                }
                if is_classification:
                    params['class_weight'] = 'balanced'

                self.lgb_model = model_class(**params)
                callbacks = [lgb.early_stopping(self.early_stop_rounds, verbose=False)]
                self.lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
                
                if is_classification:
                    acc = (self.lgb_model.predict(X_val) == y_val).mean()
                    print(f"[{self.task_name}] LGB val accuracy: {acc:.4f}")
                else:
                    preds = self.lgb_model.predict(X_val)
                    mae = np.mean(np.abs(preds - y_val))
                    print(f"[{self.task_name}] LGB val MAE: {mae:.4f}")
            except Exception as exc:
                print(f"[{self.task_name}] LGB failed: {exc}")
                self.lgb_model = None

        self._trained = self.xgb_model is not None or self.lgb_model is not None
        if self._trained:
            self._update_weights(X_val, y_val)

    def _update_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        model_accs = {}

        if self.xgb_model is not None:
            if self.task_type == 'classification':
                acc = (self.xgb_model.predict(X_val) == y_val).mean()
            else:
                preds = self.xgb_model.predict(X_val)
                acc = 1.0 / (1.0 + np.mean(np.abs(preds - y_val)))
            model_accs['xgb'] = max(0.1, float(acc))

        if self.lgb_model is not None:
            if self.task_type == 'classification':
                acc = (self.lgb_model.predict(X_val) == y_val).mean()
            else:
                preds = self.lgb_model.predict(X_val)
                acc = 1.0 / (1.0 + np.mean(np.abs(preds - y_val)))
            model_accs['lgb'] = max(0.1, float(acc))

        if model_accs:
            total = sum(model_accs.values())
            self._weights = {k: v / total for k, v in model_accs.items()}
            print(f"[{self.task_name}] Ensemble weights: {self._weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            if self.task_type == 'classification':
                return np.full((len(X), self.n_classes), 1.0 / self.n_classes)
            else:
                return np.zeros(len(X))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_clean = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
        X_df = pd.DataFrame(X_clean).ffill().values
        X_scaled = self._fill_and_scale(X_df)
        if X_scaled is None:
            if self.task_type == 'classification':
                return np.full((len(X), self.n_classes), 1.0 / self.n_classes)
            else:
                return np.zeros(len(X))

        predictions = []
        weights = []

        if self.xgb_model is not None:
            try:
                if self.task_type == 'classification':
                    pred = self.xgb_model.predict_proba(X_scaled)
                    if self.n_classes == 2 and pred.shape[1] == 2:
                        pass
                    elif self.n_classes == 2:
                        pred = np.column_stack([1 - pred[:, 1], pred[:, 1]])
                else:
                    pred = self.xgb_model.predict(X_scaled).reshape(-1, 1)
                predictions.append(pred)
                weights.append(self._weights.get('xgb', 0.5))
            except Exception:
                pass

        if self.lgb_model is not None:
            try:
                if self.task_type == 'classification':
                    pred = self.lgb_model.predict_proba(X_scaled)
                    if self.n_classes == 2 and pred.shape[1] == 2:
                        pass
                    elif self.n_classes == 2:
                        pred = np.column_stack([1 - pred[:, 1], pred[:, 1]])
                else:
                    pred = self.lgb_model.predict(X_scaled).reshape(-1, 1)
                predictions.append(pred)
                weights.append(self._weights.get('lgb', 0.5))
            except Exception:
                pass

        if not predictions:
            if self.task_type == 'classification':
                return np.full((len(X), self.n_classes), 1.0 / self.n_classes)
            else:
                return np.zeros(len(X))

        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]

        max_cols = max(p.shape[1] for p in predictions)
        final_pred = np.zeros((len(X), max_cols))
        for pred, w in zip(predictions, weights):
            if pred.shape[1] < max_cols:
                pad = np.zeros((len(X), max_cols - pred.shape[1]))
                pred = np.hstack([pred, pad])
            final_pred += pred * w

        if self.task_type == 'regression':
            return final_pred.squeeze(-1)
        return final_pred

    def save(self, path: str) -> None:
        data = {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'n_classes': self.n_classes,
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'scaler': self.scaler,
            'feat_cols': self._feat_cols,
            'weights': self._weights,
            'trained': self._trained,
            'cfg': self.cfg
        }
        joblib.dump(data, path)
        print(f"[{self.task_name}] Saved to {path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"[{self.task_name}] File not found: {path}")
            return False
        try:
            data = joblib.load(path)
            self.task_name = data.get('task_name', self.task_name)
            self.task_type = data.get('task_type', self.task_type)
            self.n_classes = data.get('n_classes', self.n_classes)
            self.xgb_model = data.get('xgb_model')
            self.lgb_model = data.get('lgb_model')
            self.scaler = data.get('scaler', RobustScaler())
            self._feat_cols = data.get('feat_cols', [])
            self._weights = data.get('weights', {'xgb': 0.5, 'lgb': 0.5})
            self._trained = data.get('trained', True)
            self.cfg = data.get('cfg', self.cfg)
            print(f"[{self.task_name}] Loaded from {path}")
            return True
        except Exception as exc:
            print(f"[{self.task_name}] Load failed: {exc}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained and (self.xgb_model is not None or self.lgb_model is not None)


class ExpertEnsemble:
    
    TASK_CONFIGS = {
        'direction': {'type': 'classification', 'n_classes': 3},
        'price_pred': {'type': 'regression', 'n_classes': 1},
        'entry_quality': {'type': 'regression', 'n_classes': 1},
        'exit_bar': {'type': 'regression', 'n_classes': 1},
    }

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.models: Dict[str, ExpertModel] = {}
        self._feature_names: List[str] = []
        self._trained = False
        
        for task_name, task_cfg in self.TASK_CONFIGS.items():
            self.models[task_name] = ExpertModel(
                task_name=task_name,
                task_type=task_cfg['type'],
                cfg=self.cfg,
                n_classes=task_cfg.get('n_classes', 1)
            )

    def train(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[str, np.ndarray]
    ) -> None:
        print("=" * 60)
        print("Training 4 Expert Ensembles (XGBoost + LightGBM)")
        print("=" * 60)
        
        for task_name, model in self.models.items():
            print(f"\n>>> Training expert: {task_name}")
            if task_name not in y_train_dict or task_name not in y_val_dict:
                print(f"  Target '{task_name}' missing. Skipping.")
                continue
            
            y_tr = y_train_dict[task_name]
            y_val = y_val_dict[task_name]
            
            valid_tr = ~np.isnan(y_tr)
            valid_val = ~np.isnan(y_val)
            
            X_tr_clean = X_train[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_val_clean = X_val[valid_val]
            y_val_clean = y_val[valid_val]
            
            if len(X_tr_clean) < 100 or len(X_val_clean) == 0:
                print(f"  Insufficient clean samples for {task_name}. Skipping.")
                continue
            
            model.train_ensemble(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean)
            print(f"  {task_name} training complete.")
        
        self._trained = any(m.is_trained for m in self.models.values())
        print("\n" + "=" * 60)
        print("All expert models training complete.")
        print("=" * 60)

    def predict_expert_signals(self, X: np.ndarray) -> Dict[str, Any]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        results = {}
        
        for task_name, model in self.models.items():
            if not model.is_trained:
                if task_name == 'direction':
                    results['direction_prob'] = np.full(len(X), 0.5) if len(X) > 1 else 0.5
                elif task_name == 'price_pred':
                    results['expected_move'] = np.full(len(X), 0.003) if len(X) > 1 else 0.003
                elif task_name == 'entry_quality':
                    results['entry_quality'] = np.full(len(X), 0.5) if len(X) > 1 else 0.5
                elif task_name == 'exit_bar':
                    results['exit_distance'] = np.full(len(X), 0.02) if len(X) > 1 else 0.02
                continue
            
            pred = model.predict(X)
            
            if task_name == 'direction':
                if pred.ndim == 1:
                    val = pred
                else:
                    if pred.shape[1] >= 3:
                        val = pred[:, 2]
                    else:
                        val = pred[:, 0]
                results['direction_prob'] = val if len(X) > 1 else float(val[0])
            
            elif task_name == 'price_pred':
                val = pred
                results['expected_move'] = np.maximum(0.0001, np.abs(val)) if len(X) > 1 else float(max(0.0001, abs(val[0])))
            
            elif task_name == 'entry_quality':
                val = pred
                results['entry_quality'] = np.clip(val, 0, 1) if len(X) > 1 else float(np.clip(val[0], 0, 1))
            
            elif task_name == 'exit_bar':
                val = pred
                res = np.clip(val * 0.05 + 0.01, 0.01, 0.06)
                results['exit_distance'] = res if len(X) > 1 else float(res[0])
        
        return results

    def predict_batch(self, X: np.ndarray) -> pd.DataFrame:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch_signals = self.predict_expert_signals(X)
        return pd.DataFrame(batch_signals)

    def save(self, base_path: str = "models/expert_ensemble") -> None:
        dirname = os.path.dirname(base_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        for task_name, model in self.models.items():
            path = f"{base_path}_{task_name}.pkl"
            model.save(path)
        print(f"Expert ensemble saved to {base_path}_*.pkl")

    def load(self, base_path: str = "models/expert_ensemble") -> bool:
        success = True
        for task_name, model in self.models.items():
            path = f"{base_path}_{task_name}.pkl"
            if not model.load(path):
                success = False
        self._trained = success
        return success

    @property
    def is_trained(self) -> bool:
        return self._trained and all(m.is_trained for m in self.models.values())

    def get_feature_importances(self) -> Dict[str, pd.DataFrame]:
        importances = {}
        for task_name, model in self.models.items():
            if model.is_trained:
                imp = None
                if model.xgb_model is not None:
                    try:
                        imp = model.xgb_model.feature_importances_
                    except Exception:
                        pass
                elif model.lgb_model is not None:
                    try:
                        imp = model.lgb_model.feature_importances_
                    except Exception:
                        pass
                        
                if imp is not None:
                    cols = model._feat_cols if model._feat_cols else [f"feat_{i}" for i in range(len(imp))]
                    df = pd.DataFrame({'feature': cols, 'importance': imp})
                    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
                    importances[task_name] = df
        return importances
