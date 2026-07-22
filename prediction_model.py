import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D,
    Conv1D, Concatenate
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import RobustScaler
from numpy.lib.stride_tricks import sliding_window_view

class PredictionModel:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.model: Optional[Model] = None
        self.scaler = None
        self._feature_cols: List[str] = []
        self._close_idx: int = 0
        self._cont_indices: List[int] = []
        self._cat_indices: List[int] = []
        self._num_cont_features: int = 0
        self._num_cat_features: int = 0
        self.embedding_dim = self.cfg.get('embedding_dim', 16)
        self.horizons = self.cfg.get('aux_horizons', [1, 4, 12, 24])
        self.envelope_horizon = self.cfg.get('envelope_horizon', 12)

    def _transformer_block(self, x, num_heads, key_dim, ff_dim, dropout, name=''):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'mha_{name}')(query=x, value=x, key=x)
        attn = Dropout(dropout)(attn)
        x1 = LayerNormalization(name=f'ln1_{name}')(x + attn)
        ff = Dense(ff_dim, activation='gelu', name=f'ff1_{name}')(x1)
        ff = Dense(x1.shape[-1], name=f'ff2_{name}')(ff)
        ff = Dropout(dropout)(ff)
        return LayerNormalization(name=f'ln2_{name}')(x1 + ff)

    def build(self, input_shape: Tuple[int, int]) -> Model:
        window = input_shape[0]
        dr = self.cfg.get('dropout_rate', 0.2)
        
        if self._num_cont_features == 0 and self._num_cat_features == 0:
            if self._feature_cols:
                self._num_cat_features = len([c for c in self._feature_cols if (c.startswith('regime_') and not c.startswith('regime_p_')) or 'trigger' in c.lower()])
                self._num_cont_features = len(self._feature_cols) - self._num_cat_features
            else:
                self._num_cat_features = 0
                self._num_cont_features = input_shape[1]

        in_cont = Input(shape=(window, self._num_cont_features), name='cont_input')

        x_cont = Conv1D(128, kernel_size=3, padding='causal', activation='gelu', name='conv_local')(in_cont)
        x_cont = Conv1D(64, kernel_size=5, padding='causal', activation='gelu', name='conv_med')(x_cont)
        x_cont = LayerNormalization(name='ln_conv')(x_cont)
        
        x_cont = LSTM(self.cfg.get('lstm_units_1', 128), return_sequences=True, name='lstm_1')(x_cont)
        x_cont = Dropout(dr, name='drop_lstm1')(x_cont)
        x_cont = LSTM(self.cfg.get('lstm_units_2', 64), return_sequences=True, name='lstm_2')(x_cont)
        x_cont = Dropout(dr, name='drop_lstm2')(x_cont)

        if self._num_cat_features > 0:
            in_cat = Input(shape=(window, self._num_cat_features), name='cat_input')
            x_cat = Dense(32, activation='gelu', name='cat_latent_projection')(in_cat)
            fused = Concatenate(axis=-1, name='quant_feature_fusion')([x_cont, x_cat])
        else:
            fused = x_cont

        x = self._transformer_block(fused, self.cfg.get('attention_heads', 8), self.cfg.get('attention_key_dim', 64), 256, dr, 't1')
        x = self._transformer_block(x, self.cfg.get('attention_heads', 8)//2, self.cfg.get('attention_key_dim', 64), 128, dr, 't2')

        trunk = GlobalAveragePooling1D(name='gap')(x)
        trunk = Dense(256, activation='gelu', name='trunk_1')(trunk)
        trunk = Dropout(dr * 0.5, name='drop_trunk')(trunk)
        trunk = Dense(128, activation='gelu', name='trunk_2')(trunk)

        embedding = Dense(self.embedding_dim, activation='linear', name='embedding')(trunk)

        aux_returns = Dense(len(self.horizons), activation='linear', name='aux_returns')(trunk)
        aux_envelope = Dense(2, activation='linear', name='aux_envelope')(trunk)

        model = Model(
            inputs=[in_cont, in_cat] if self._num_cat_features > 0 else in_cont,
            outputs=[aux_returns, aux_envelope, embedding],
            name='FeatureExtractor'
        )

        initial_lr = self.cfg.get('learning_rate', 0.001)
        model.compile(
            optimizer=AdamW(learning_rate=initial_lr, weight_decay=1e-4),
            loss={
                'aux_returns': 'huber',
                'aux_envelope': 'huber',
                'embedding': None
            },
            loss_weights={
                'aux_returns': 1.0,
                'aux_envelope': 1.0,
                'embedding': 0.0
            },
            metrics={
                'aux_returns': ['mae'],
                'aux_envelope': ['mae']
            }
        )
        self.model = model
        return model

    @staticmethod
    def _engineer_auxiliary_targets(df: pd.DataFrame, horizons: List[int], envelope_horizon: int) -> Dict[str, np.ndarray]:
        n = len(df)
        closes = df['close'].values.astype(np.float64)
        
        rets = {}
        for h in horizons:
            ret_full = np.full(n, 0.0, dtype=np.float32)
            if h < n:
                ret_full[:-h] = np.log(closes[h:] / (closes[:-h] + 1e-10))
            rets[f'ret_{h}h'] = ret_full
        
        max_up = np.full(n, 0.0, dtype=np.float32)
        max_down = np.full(n, 0.0, dtype=np.float32)
        
        if n > envelope_horizon:
            future_view = sliding_window_view(closes[1:], window_shape=envelope_horizon)
            max_future = np.max(future_view, axis=1)
            min_future = np.min(future_view, axis=1)
            valid_len = min(n - envelope_horizon, len(max_future))
            max_up[:valid_len] = (max_future[:valid_len] - closes[:valid_len]) / (closes[:valid_len] + 1e-10)
            max_down[:valid_len] = (closes[:valid_len] - min_future[:valid_len]) / (closes[:valid_len] + 1e-10)
        
        return {
            'returns': np.column_stack([rets[f'ret_{h}h'] for h in horizons]),
            'envelope': np.column_stack([max_up, max_down])
        }

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str] = None, is_training: bool = True):
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        closes = numeric_df['close'].values
        atrs = numeric_df['atr'].values if 'atr' in numeric_df.columns else closes * 0.01
        smc_price_cols = [col for col in numeric_df.columns if any(x in col.lower() for x in ['ob_', 'fvg_', 'liquidity_'])]
        for col in smc_price_cols:
            numeric_df[col] = (numeric_df[col] - closes) / (atrs + 1e-10)

        if 'hurst_exp' in numeric_df.columns:
            numeric_df['hurst_exp'] = numeric_df['hurst_exp'].ewm(span=8, adjust=False).mean()
        if 'efficiency_ratio_20' in numeric_df.columns:
            numeric_df['efficiency_ratio_20'] = numeric_df['efficiency_ratio_20'].ewm(span=10, adjust=False).mean()

        skew_cols = [col for col in numeric_df.columns if 'skew' in col.lower()]
        kurt_cols = [col for col in numeric_df.columns if 'kurt' in col.lower()]
        if skew_cols or kurt_cols:
            if self.cfg.get('drop_skew_kurt', True):
                numeric_df.drop(columns=skew_cols + kurt_cols, inplace=True, errors='ignore')

        if 'regime' in numeric_df.columns:
            regime_dummies = pd.get_dummies(numeric_df['regime'], prefix='regime').astype(np.float32)
            numeric_df = pd.concat([numeric_df.drop(columns=['regime']), regime_dummies], axis=1)

        if feature_cols is None:
            feature_cols = [c for c in numeric_df.columns if c not in ['timestamp']]
        else:
            missing = [c for c in feature_cols if c not in numeric_df.columns]
            if missing:
                raise KeyError(f"features not in DataFrame: {missing}")
            feature_cols = [c for c in feature_cols if c in numeric_df.columns]

        if 'close' not in numeric_df.columns:
            raise ValueError("DataFrame must contain 'close' column for target engineering.")
        
        if feature_cols is not None and 'close' in feature_cols:
            close_idx = feature_cols.index('close')
            feature_cols = [f for f in feature_cols if f != 'close']
        else:
            close_idx = numeric_df.columns.get_loc('close')
        
        if feature_cols is not None and 'close' in feature_cols:
            feature_cols.remove('close')

        data = numeric_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        self._feature_cols = [c for c in feature_cols if c != 'close']
        all_cat_cols = [c for c in self._feature_cols if (c.startswith('regime_') and not c.startswith('regime_p_')) or 'trigger' in c.lower()]
        all_cont_cols = [c for c in self._feature_cols if c not in all_cat_cols]
        
        self._cont_indices = [self._feature_cols.index(c) for c in all_cont_cols]
        self._cat_indices = [self._feature_cols.index(c) for c in all_cat_cols]
        self._num_cont_features = len(all_cont_cols)
        self._num_cat_features = len(all_cat_cols)

        if is_training:
            if self.scaler is None:
                self.scaler = RobustScaler()
                self.scaler.fit(data[all_cont_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler is not fitted yet. Ensure you process training data with is_training=True first.")

        scaled = data[self._feature_cols].values.astype(np.float32)
        cont_indices_in_features = [self._feature_cols.index(c) for c in all_cont_cols]
        scaled[:, [self._feature_cols.index(c) for c in all_cont_cols]] = self.scaler.transform(data[all_cont_cols])

        aux_targets = self._engineer_auxiliary_targets(df, self.horizons, self.envelope_horizon)
        y_returns = aux_targets['returns'].astype(np.float32)
        y_envelope = aux_targets['envelope'].astype(np.float32)

        window = self.cfg.get('window', 120)
        n = len(scaled)
        if n <= window:
            raise ValueError(f"Data length {n} <= window {window}")

        try:
            X = sliding_window_view(scaled, window_shape=(window, scaled.shape[1])).squeeze(1).astype(np.float32)
            X = X[:-1]
        except:
            X = np.array([scaled[i-window:i] for i in range(window, n)], dtype=np.float32)

        y_returns = y_returns[window-1:n-1]
        y_envelope = y_envelope[window-1:n-1]

        min_len = min(len(X), len(y_returns), len(y_envelope))
        X, y_returns, y_envelope = X[:min_len], y_returns[:min_len], y_envelope[:min_len]

        self._close_idx = close_idx

        y_dict = {
            'aux_returns': y_returns,
            'aux_envelope': y_envelope
        }

        return X, y_dict, self._feature_cols, close_idx

    def _split_to_multi_input(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        max_idx = X.shape[2] - 1
        for i in self._cont_indices:
            if i > max_idx:
                raise ValueError(f"Continuous feature index {i} out of bounds for input shape {X.shape}")
        for i in self._cat_indices:
            if i > max_idx:
                raise ValueError(f"Categorical feature index {i} out of bounds for input shape {X.shape}")
                
        if self._num_cat_features > 0:
            return {
                'cont_input': X[:, :, self._cont_indices],
                'cat_input': X[:, :, self._cat_indices]
            }
        else:
            return {
                'cont_input': X[:, :, self._cont_indices]
            }

    def train(self, X_train, X_val, y_train, y_val):
        if self.model is None:
            self.build((X_train.shape[1], X_train.shape[2]))

        steps_per_epoch = len(X_train) // self.cfg.get('batch_size', 32)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        total_steps = self.cfg.get('epochs', 50) * steps_per_epoch
        
        lr_schedule = CosineDecay(
            initial_learning_rate=self.cfg.get('learning_rate', 0.001),
            decay_steps=total_steps,
            alpha=1e-4
        )

        self.model.compile(
            optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss={
                'aux_returns': 'huber',
                'aux_envelope': 'huber',
                'embedding': None
            },
            loss_weights={
                'aux_returns': 1.0,
                'aux_envelope': 1.0,
                'embedding': 0.0
            },
            metrics={
                'aux_returns': ['mae'],
                'aux_envelope': ['mae']
            }
        )

        X_train_multi = self._split_to_multi_input(X_train)
        X_val_multi = self._split_to_multi_input(X_val)

        print("[PredictionModel] Training LSTM feature extractor...")
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=self.cfg.get('early_stop_patience', 15),
                restore_best_weights=True,
                verbose=1
            ),
        ]

        history = self.model.fit(
            X_train_multi, y_train,
            validation_data=(X_val_multi, y_val),
            epochs=self.cfg.get('epochs', 50),
            batch_size=self.cfg.get('batch_size', 32),
            callbacks=callbacks,
            shuffle=False,
            verbose=1
        )
        print("[PredictionModel] LSTM training complete.")

    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded/trained. Call load() or train() first.")
        
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        
        X_multi = self._split_to_multi_input(X)
        out = self.model.predict(X_multi, verbose=0)
        
        if isinstance(out, dict):
            return out['embedding']
        elif isinstance(out, (list, tuple)):
            if hasattr(self.model, 'output_names') and 'embedding' in self.model.output_names:
                return out[self.model.output_names.index('embedding')]
            return out[2]
        return out

    def save(self, path=None):
        if self.model is None:
            raise ValueError("No model instance found to save.")
            
        save_path = path or self.cfg.get('model_save_path', 'models/feature_extractor.keras')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        if self.scaler:
            joblib.dump(self.scaler, self.cfg.get('scaler_save_path', 'models/scaler.pkl'))
        
        config = {
            'feature_cols': self._feature_cols,
            'close_idx': self._close_idx,
            'cont_indices': self._cont_indices,
            'cat_indices': self._cat_indices,
            'num_cont': self._num_cont_features,
            'num_cat': self._num_cat_features,
            'embedding_dim': self.embedding_dim,
            'horizons': self.horizons,
            'envelope_horizon': self.envelope_horizon
        }
        with open(save_path.replace('.keras', '_config.json'), 'w') as f:
            json.dump(config, f)
        print(f"[PredictionModel] Saved to {save_path}")

    def load(self, path=None):
        load_path = path or self.cfg.get('model_save_path', 'models/feature_extractor.keras')
        config_path = load_path.replace('.keras', '_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file missing at {config_path}")
            
        self.model = tf.keras.models.load_model(load_path)
        scaler_path = self.cfg.get('scaler_save_path', 'models/scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        with open(config_path, 'r') as f:
            c = json.load(f)
        self._feature_cols = c.get('feature_cols', [])
        self._close_idx = c.get('close_idx', 0)
        self._cont_indices = c.get('cont_indices', [])
        self._cat_indices = c.get('cat_indices', [])
        self._num_cont_features = c.get('num_cont', 0)
        self._num_cat_features = c.get('num_cat', 0)
        self.embedding_dim = c.get('embedding_dim', 16)
        self.horizons = c.get('horizons', [1, 4, 12, 24])
        self.envelope_horizon = c.get('envelope_horizon', 12)
        print(f"[PredictionModel] Loaded from {load_path}")
