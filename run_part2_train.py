import os
import sys
import time
import json
import argparse
import logging
import warnings
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
warnings.filterwarnings('ignore', message='Gradients do not exist')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engine import FeatureEngine
from regime_detector import MarketRegimeDetector
from prediction_model import PredictionModel
from ensemble_model import ExpertEnsemble
from ppo_agent import PPOAgent
from trading_env import TradingEnvironment

logger = logging.getLogger('TrainingPipeline')
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class TrainingPipeline:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.stats = {
            'success': False,
            'duration_seconds': 0,
            'ohlcv_bars': 0,
            'feature_count': 0,
            'embedding_dim': self.config.get('embedding_dim', 16),
            'lstm_trained': False,
            'ensemble_trained': False,
            'ppo_trained': False,
            'regime_fitted': False
        }

    def _load_config(self, config_path: str = None) -> dict:
        defaults = {
            'symbol': 'BTC/USDT', 'timeframe': '1h', 'window': 120,
            'train_split': 0.8,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.0003,
            'lstm_units_1': 128,
            'lstm_units_2': 64,
            'attention_heads': 8,
            'attention_key_dim': 64,
            'dropout_rate': 0.2,
            'early_stop_patience': 20,
            'embedding_dim': 16,
            'aux_horizons': [1, 6, 12, 24],
            'envelope_horizon': 12,
            'ensemble_n_estimators': 300,
            'ensemble_max_depth': 6,
            'ensemble_learning_rate': 0.05,
            'ensemble_subsample': 0.8,
            'ensemble_colsample': 0.8,
            'ensemble_early_stop': 20,
            'enable_ppo': True,
            'rl_n_episodes': 200,
            'rl_ppo_epochs': 10,
            'rl_gamma': 0.99,
            'rl_clip_epsilon': 0.2,
            'rl_entropy_coeff': 0.01,
            'ppo_actions': {
                'buy_levels': [0.10, 0.30, 0.50],
                'sell_levels': [0.10, 0.30, 0.50]
            },
            'initial_capital': 10000,
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'max_risk_per_trade': 0.02,
            'max_position_pct': 0.5,
            'drawdown_penalty': 2.0,
            'trading_mode': 'spot',
            'leverage': 10,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.03,
            'target_col': 'target'
        }
        paths = [config_path, 'config.json', os.path.join(os.path.dirname(__file__), 'config.json')]
        cfg = defaults.copy()
        for p in paths:
            if p and os.path.exists(p):
                with open(p, 'r') as f:
                    cfg.update(json.load(f))
                logger.info(f"Config loaded from {p}")
                break
        return cfg

    def _find_data_csv(self) -> str:
        for path in ['.', './data', '../data']:
            for pat in ['ohlcv_data.csv', '*ohlcv*.csv']:
                matches = glob.glob(os.path.join(path, pat))
                if matches:
                    return matches[0]
        raise FileNotFoundError("No OHLCV CSV found. Run Part 1 first.")

    def load_data(self) -> pd.DataFrame:
        path = self._find_data_csv()
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        logger.info(f"Loaded {len(df)} bars")
        self.stats['ohlcv_bars'] = len(df)
        return df

    def run(self) -> dict:
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print("PART 2 – 3-STAGE HIERARCHICAL TRAINING PIPELINE (DUAL FEATURE SET)")
        print("=" * 70 + "\n")

        try:
            df_raw = self.load_data()
            split_ratio = self.config.get('train_split', 0.8)
            split_idx = int(len(df_raw) * split_ratio)
            win = self.config.get('window', 120)

            df_train_raw = df_raw.iloc[:split_idx].copy().reset_index(drop=True)
            df_val_raw = df_raw.iloc[split_idx - win:].copy().reset_index(drop=True)

            if 'timestamp' in df_train_raw.columns:
                df_train_raw = df_train_raw.set_index(pd.to_datetime(df_train_raw['timestamp'], utc=True))
            if 'timestamp' in df_val_raw.columns:
                df_val_raw = df_val_raw.set_index(pd.to_datetime(df_val_raw['timestamp'], utc=True))

            logger.info(f"Train: {len(df_train_raw)} bars | Val: {len(df_val_raw) - win} bars")

            print("[FeatureEngine] Building 17 dual-set features...")
            fe = FeatureEngine(cfg=self.config)

            with tqdm(total=1, desc="Train Features") as pbar:
                df_train_feats = fe.build_all(df_train_raw.copy())
                pbar.update(1)
            with tqdm(total=1, desc="Val Features") as pbar:
                df_val_feats = fe.build_all(df_val_raw.copy())
                pbar.update(1)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_train_raw.columns:
                    df_train_feats[col] = df_train_raw[col]
                    df_val_feats[col] = df_val_raw[col]

            df_train_feats = df_train_feats.reset_index(drop=True).ffill().bfill()
            df_val_feats = df_val_feats.reset_index(drop=True).ffill().bfill()

            print("[RegimeDetector] Fitting GMM and annotating regimes...")
            regime_detector = MarketRegimeDetector(self.config)
            regime_detector.fit(df_train_feats)
            df_train_feats = regime_detector.annotate(df_train_feats)
            df_val_feats = regime_detector.annotate(df_val_feats)
            regime_detector.save_map()
            self.stats['regime_fitted'] = True

            regime_dummy_cols = [c for c in df_train_feats.columns if c.startswith('regime_') and not c.startswith('regime_p_')]
            df_train_feats.drop(columns=regime_dummy_cols, errors='ignore', inplace=True)
            df_val_feats.drop(columns=regime_dummy_cols, errors='ignore', inplace=True)

            DEEP_FEATURES = [
                'amihud_illiq', 'vpin', 'roll_spread', 'ou_half_life',
                'wavelet_hf_ratio', 'higuchi_fd',
                'regime_p_0', 'regime_p_1', 'regime_p_2'
            ]

            ENSEMBLE_FEATURES = [
                'hurst_exp', 'efficiency_ratio_20', 'natr',
                'rejection_high', 'rejection_low', 'range_percentile',
                'price_accel', 'vol_aggression', 'stop_buy_dist',
                'stop_sell_dist', 'vwap_ema_spread'
            ]

            all_cols = df_train_feats.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in all_cols:
                all_cols.remove('timestamp')
            self.stats['feature_count'] = len(ENSEMBLE_FEATURES)
            print(f"[Pipeline] LSTM Input: {len(DEEP_FEATURES)} features (6 PhD + 3 Regime)")
            print(f"[Pipeline] Ensemble Input: {len(ENSEMBLE_FEATURES)} features (11 Micro-Scalping)")

            print("[LSTM] Preparing sequences and training feature extractor...")
            lstm_model = PredictionModel(self.config)

            X_train, y_train_dict, feat_cols, close_idx = lstm_model.prepare_data(
                df_train_feats, feature_cols=DEEP_FEATURES, is_training=True
            )
            X_val, y_val_dict, _, _ = lstm_model.prepare_data(
                df_val_feats, feature_cols=DEEP_FEATURES, is_training=False
            )

            embedding_dim = self.config.get('embedding_dim', 16)

            for df_feats, y_dict, X_mat in [(df_train_feats, y_train_dict, X_train), (df_val_feats, y_val_dict, X_val)]:
                n = len(X_mat)
                df_slice = df_feats.iloc[-n:].copy()

                future_ret = (df_slice['close'].shift(-1) / df_slice['close']) - 1
                direction = np.zeros(n, dtype=np.float32)
                direction[future_ret > 0.005] = 2.0
                direction[future_ret < -0.005] = 0.0
                direction[(future_ret >= -0.005) & (future_ret <= 0.005)] = 1.0
                y_dict['direction'] = np.nan_to_num(direction, nan=1.0)

                y_dict['price_pred'] = future_ret.fillna(0).values.astype(np.float32)

                atr_pct = df_slice['natr'].values
                y_dict['entry_quality'] = (1 / (1 + atr_pct * 100)).astype(np.float32)
                y_dict['exit_bar'] = np.clip(atr_pct * 3.0, 0.002, 0.03).astype(np.float32)

            for df_feats, y_dict, X_mat in [(df_train_feats, y_train_dict, X_train), (df_val_feats, y_val_dict, X_val)]:
                df_feats['tg_1'] = np.log(df_feats['close'].shift(-1) / df_feats['close'])
                df_feats['tg_6'] = np.log(df_feats['close'].shift(-6) / df_feats['close'])
                df_feats['tg_12'] = np.log(df_feats['close'].shift(-12) / df_feats['close'])
                df_feats['tg_24'] = np.log(df_feats['close'].shift(-24) / df_feats['close'])

                aux_matrix = df_feats[['tg_1', 'tg_6', 'tg_12', 'tg_24']].bfill().ffill().fillna(0.0).values.astype(np.float32)
                aligned_returns = aux_matrix[-len(X_mat):]

                orig_envelope = y_dict.get('aux_envelope', None)
                if orig_envelope is None:
                    orig_envelope = np.zeros((len(X_mat), 2), dtype=np.float32)

                dummy_embedding = np.zeros((len(X_mat), embedding_dim), dtype=np.float32)

                y_dict['aux_returns'] = aligned_returns
                y_dict['aux_envelope'] = orig_envelope
                y_dict['embedding'] = dummy_embedding

            print("[LSTM] Training feature extractor...")
            lstm_train_keys = ['aux_returns', 'aux_envelope', 'embedding']
            lstm_model.train(
                X_train, X_val,
                {k: y_train_dict[k] for k in lstm_train_keys},
                {k: y_val_dict[k] for k in lstm_train_keys}
            )
            lstm_model.save('models/feature_extractor.keras')
            self.stats['lstm_trained'] = True
            print("[LSTM] Feature extractor trained and saved.")

            print("[LSTM] Extracting embeddings...")
            lstm_model.load('models/feature_extractor.keras')

            X_train_multi = lstm_model._split_to_multi_input(X_train)
            X_val_multi = lstm_model._split_to_multi_input(X_val)

            train_embeds = lstm_model.model.predict(X_train_multi, verbose=0)
            val_embeds = lstm_model.model.predict(X_val_multi, verbose=0)

            if isinstance(train_embeds, dict):
                train_embeds = train_embeds['embedding']
                val_embeds = val_embeds['embedding']
            elif isinstance(train_embeds, list):
                train_embeds = train_embeds[2]
                val_embeds = val_embeds[2]

            print(f"[LSTM] Embeddings shape: Train {train_embeds.shape}, Val {val_embeds.shape}")

            print("[Ensemble] Training 4 expert models (XGB+LGB)...")
            train_raw_aligned = df_train_feats[ENSEMBLE_FEATURES].iloc[-len(X_train):].reset_index(drop=True)
            val_raw_aligned = df_val_feats[ENSEMBLE_FEATURES].iloc[-len(X_val):].reset_index(drop=True)

            X_train_combined = np.concatenate([
                train_raw_aligned.values.astype(np.float32),
                train_embeds.astype(np.float32)
            ], axis=1)

            X_val_combined = np.concatenate([
                val_raw_aligned.values.astype(np.float32),
                val_embeds.astype(np.float32)
            ], axis=1)

            ensemble_targets = ['direction', 'price_pred', 'entry_quality', 'exit_bar']
            y_train_ensemble = {k: y_train_dict[k] for k in ensemble_targets if k in y_train_dict}
            y_val_ensemble = {k: y_val_dict[k] for k in ensemble_targets if k in y_val_dict}

            expert_ensemble = ExpertEnsemble(self.config)
            expert_ensemble.train(
                X_train_combined, y_train_ensemble,
                X_val_combined, y_val_ensemble
            )
            expert_ensemble.save('models/expert_ensemble')
            self.stats['ensemble_trained'] = True
            print("[Ensemble] 4 expert models trained and saved.")

            if self.config.get('enable_ppo', True):
                print("[PPO] Training meta-learner (12-D state)...")
                env = TradingEnvironment(self.config, expert_ensemble=expert_ensemble)

                all_cat_cols = [c for c in DEEP_FEATURES if c.startswith('regime_p_')]
                all_cont_cols = [c for c in DEEP_FEATURES if c not in all_cat_cols]

                df_train_scaled_ctx = df_train_feats.copy()
                if len(all_cont_cols) > 0:
                    df_train_scaled_ctx[all_cont_cols] = lstm_model.scaler.transform(
                        df_train_feats[all_cont_cols].values
                    )

                state_features = DEEP_FEATURES
                scaled_train = df_train_scaled_ctx[state_features].values.astype(np.float32)

                env.reset(df_train_feats, scaled_train, close_idx=-1)

                ppo = PPOAgent(self.config, state_dim=12)
                ppo.train(env, n_episodes=self.config.get('rl_n_episodes', 200))
                ppo.save('models/ppo_agent')
                self.stats['ppo_trained'] = True
                print("[PPO] Meta-learner trained and saved.")

            os.makedirs('models', exist_ok=True)

            with open('models/final_features.json', 'w') as f:
                json.dump({
                    'deep_features': DEEP_FEATURES,
                    'ensemble_features': ENSEMBLE_FEATURES,
                    'embedding_dim': self.config.get('embedding_dim', 16),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)

            with open('models/training_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)

            with open('models/training_config.json', 'w') as f:
                json.dump(self.config, f, indent=2, default=str)

            self.stats['duration_seconds'] = round(time.time() - self.start_time, 2)
            self.stats['success'] = True

            print("\n" + "=" * 70)
            print("PART 2 – TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Duration:        {self.stats['duration_seconds']} sec")
            print(f"OHLCV bars:      {self.stats['ohlcv_bars']}")
            print(f"LSTM Features:   {len(DEEP_FEATURES)} (6 PhD + 3 Regime)")
            print(f"Ensemble Features: {len(ENSEMBLE_FEATURES)} (11 Micro-Scalping)")
            print(f"Embedding dim:   {self.stats['embedding_dim']}")
            print(f"LSTM trained:    {self.stats['lstm_trained']}")
            print(f"Ensembles:       {self.stats['ensemble_trained']} (4 experts)")
            print(f"PPO trained:     {self.stats['ppo_trained']} (State 12-D)")
            print(f"Regime fitted:   {self.stats['regime_fitted']} (EWM smoothed)")
            print("=" * 70 + "\n")

            return self.stats

        except Exception as e:
            self.stats['success'] = False
            self.stats['error'] = str(e)
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return self.stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    result = pipeline.run()
    return 0 if result['success'] else 1

if __name__ == '__main__':
    exit(main())
