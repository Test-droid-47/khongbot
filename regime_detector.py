import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from sklearn.mixture import GaussianMixture

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class MarketRegimeDetector:
    REGIME_NAMES = {0: 'Ranging', 1: 'StrongBull', 2: 'Bull', 3: 'Bear'}

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        gold_features = [
            'vol_regime_score',
            'efficiency_ratio_20',
            'natr',
            'perm_entropy',
            'vol_of_vol'
        ]
        self.feature_list = self.cfg.get('regime_features', gold_features)
        self.n_components = self.cfg.get('gmm_components', 4)
        self.map_path = self.cfg.get('regime_map_path', 'regime_label_map.json')
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42,
            max_iter=500,
            n_init=5,
            tol=1e-4
        )
        self._fitted = False
        self._remap: Dict[int, int] = {}

    def _build_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_arrays = []
        for feat in self.feature_list:
            if feat not in df.columns:
                raise KeyError(f"Regime feature '{feat}' not found in DataFrame. Ensure feature_engine.py generates it.")
            col = df[feat].values.astype(np.float64)
            feature_arrays.append(col)
        X = np.column_stack(feature_arrays).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, df: pd.DataFrame) -> None:
        required_cols = ['close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing")
        X = self._build_regime_features(df)
        n_samples = len(X)
        min_samples = self.n_components * 10
        if n_samples < min_samples:
            print(f"⚠️ Only {n_samples} samples, need at least {min_samples}. Regime detection may be unreliable.")
        self.gmm.fit(X)
        self._fitted = True

        sort_feat = self.feature_list[0]
        sort_idx = self.feature_list.index(sort_feat)
        means = self.gmm.means_[:, sort_idx]
        rank = np.argsort(means)

        if self.n_components == 4:
            self._remap = {int(rank[0]): 3, int(rank[1]): 0, int(rank[2]): 2, int(rank[3]): 1}
        elif self.n_components == 3:
            self._remap = {int(rank[0]): 3, int(rank[1]): 0, int(rank[2]): 1}
        else:
            for i, idx in enumerate(rank):
                self._remap[int(idx)] = i

        self.save_map()
        print(f"GMM fitted. Regime mapping: {self._remap}")
        for regime_id in range(self.n_components):
            print(f"  Component {regime_id}: mean_ret={means[regime_id]:.6f}")

    def save_map(self) -> None:
        meta_data = {
            'remap': self._remap,
            'n_components': self.n_components,
            'fitted': self._fitted,
            'feature_list': self.feature_list
        }
        try:
            dir_name = os.path.dirname(self.map_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(self.map_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            gmm_model_path = self.map_path.replace('.json', '.pkl')
            joblib.dump(self.gmm, gmm_model_path)
            print(f"Metadata saved to {self.map_path} and model state to {gmm_model_path}")
        except Exception as e:
            print(f"Failed to save map: {e}")

    def load_map(self) -> bool:
        if not os.path.exists(self.map_path):
            print(f"Map file not found: {self.map_path}")
            return False
        try:
            with open(self.map_path, 'r') as f:
                data = json.load(f)
            self._remap = {int(k): int(v) for k, v in data.get('remap', {}).items()}
            self.n_components = data.get('n_components', 4)
            self._fitted = data.get('fitted', False)
            self.feature_list = data.get('feature_list', self.feature_list)
            gmm_model_path = self.map_path.replace('.json', '.pkl')
            if os.path.exists(gmm_model_path):
                self.gmm = joblib.load(gmm_model_path)
            else:
                if self._fitted:
                    print(f"Binary model state missing at {gmm_model_path} but metadata claims fitted=True")
                    self._fitted = False
                    return False
            print(f"Full runtime state loaded from {self.map_path}")
            return True
        except Exception as e:
            print(f"Failed to load map: {e}")
            return False

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            if self.load_map():
                if not self._fitted:
                    raise RuntimeError("GMM not fitted and could not load map.")
            else:
                raise RuntimeError("GMM not fitted. Call fit() or load_map() first.")
        X = self._build_regime_features(df)
        labels = self.gmm.predict(X)
        probs = self.gmm.predict_proba(X)
        mapped_labels = np.array([self._remap.get(int(l), 0) for l in labels], dtype=np.int8)
        df = df.copy()
        df['regime'] = mapped_labels
        df['regime_name'] = df['regime'].map(self.REGIME_NAMES).fillna('Unknown')
        for i in range(self.n_components):
            df[f'regime_p_{i}'] = probs[:, i].astype(np.float32)
        df['regime_confidence'] = np.max(probs, axis=1).astype(np.float32)
        df['regime_entropy'] = -np.sum(probs * np.log(probs + 1e-10), axis=1).astype(np.float32)
        regime_counts = df['regime'].value_counts().sort_index()
        for regime, count in regime_counts.items():
            print(f"  {self.REGIME_NAMES.get(regime, 'Unknown')}: {count} bars ({count/len(df)*100:.1f}%)")
        return df

    def predict_live(self, feature_row: np.ndarray) -> Dict[str, Any]:
        if not self._fitted:
            if self.load_map():
                if not self._fitted:
                    return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}
            else:
                return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}
        if feature_row.ndim == 1:
            feature_row = feature_row.reshape(1, -1)
        expected_dim = self.gmm.means_.shape[1]
        if feature_row.shape[1] != expected_dim:
            if feature_row.shape[1] < expected_dim:
                pad = np.zeros((1, expected_dim - feature_row.shape[1]), dtype=np.float32)
                feature_row = np.hstack([feature_row, pad])
            else:
                feature_row = feature_row[:, :expected_dim]
        try:
            probs = self.gmm.predict_proba(feature_row)[0]
            raw = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = self._remap.get(raw, 0)
            return {
                'regime': label,
                'regime_name': self.REGIME_NAMES.get(label, 'Unknown'),
                'probs': probs.tolist(),
                'confidence': confidence,
                'raw_component': raw
            }
        except Exception as e:
            print(f"Live prediction failed: {e}")
            return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}
