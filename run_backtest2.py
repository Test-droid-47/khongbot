import os
import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore')

SIGNAL_BUY = 1
SIGNAL_SELL = 0
SIGNAL_HOLD = 2

class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    cyan = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    FORMATS = {
        logging.DEBUG: grey + "[%(asctime)s] [DEBUG] %(message)s" + reset,
        logging.INFO: green + "[%(asctime)s] [INFO] %(message)s" + reset,
        logging.WARNING: yellow + "[%(asctime)s] [WARNING] %(message)s" + reset,
        logging.ERROR: red + "[%(asctime)s] [ERROR] %(message)s" + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] [CRITICAL] %(message)s" + reset,
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        fmt = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return fmt.format(record)

logger = logging.getLogger('Backtest')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    fh = logging.FileHandler('backtest.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from feature_engine import FeatureEngine
    from prediction_model import PredictionModel
    from risk_manager import RiskManager
    from ppo_agent import PPOAgent
    from ensemble_model import ExpertEnsemble
except ImportError as e:
    logger.error(f"Critical import failed: {e}")
    sys.exit(1)

class FileDiscovery:
    @staticmethod
    def find_models(base_path: str = '.') -> Dict[str, str]:
        found = {}
        if not base_path:
            base_path = '.'
        search_paths = [base_path, os.path.join(base_path, 'models')]
        logger.info("Searching for model files...")
        for sp in search_paths:
            if not os.path.exists(sp) or not os.path.isdir(sp):
                continue
            for f in os.listdir(sp):
                fp = os.path.join(sp, f)
                fl = f.lower()
                if fl == 'feature_extractor.keras':
                    found['lstm'] = fp
                elif fl == 'scaler.pkl':
                    found['scaler'] = fp
                elif f.startswith('expert_ensemble_') and f.endswith('.pkl'):
                    name = f.replace('expert_ensemble_', '').replace('.pkl', '')
                    found[f'expert_{name}'] = fp
                elif fl == 'ppo_agent.index':
                    found['ppo_index'] = fp
                elif fl == 'ppo_agent.data-00000-of-00001':
                    found['ppo_data'] = fp
                elif f.endswith('.json') and 'final_features' in fl:
                    found['features'] = fp
        if 'ppo_index' in found and 'ppo_data' in found:
            found['ppo_checkpoint'] = os.path.splitext(found['ppo_index'])[0]
        return found

    @staticmethod
    def find_data(data_path: str = None) -> Dict[str, str]:
        found = {}
        search = []
        if data_path:
            if os.path.isfile(data_path) and data_path.endswith('.csv'):
                found['ohlcv'] = data_path
                search.append(os.path.dirname(data_path))
            else:
                search.append(data_path)
        search.extend(['.', './data'])
        for sp in search:
            if not sp or not os.path.exists(sp):
                continue
            if os.path.isfile(sp) and sp.endswith('.csv'):
                if 'ohlcv' in sp.lower() or 'price' in sp.lower():
                    found['ohlcv'] = sp
                elif 'fear' in sp.lower() or 'greed' in sp.lower():
                    found['fear_greed'] = sp
            elif os.path.isdir(sp):
                for f in os.listdir(sp):
                    fp = os.path.join(sp, f)
                    if f.endswith('.csv'):
                        if 'ohlcv' in f.lower() or 'price' in f.lower():
                            found['ohlcv'] = fp
                        elif 'fear' in f.lower() or 'greed' in f.lower():
                            found['fear_greed'] = fp
        return found

class BacktestRunner:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.models = {}
        self.data = {}
        self.warmup_bars = self.config.get('warmup_bars', 200)
        self.risk_manager = RiskManager(self.config)
        self.feature_lists = {'lstm': [], 'ensemble': [], 'regime': []}
        self.embedding_dim = 16

    def _load_config(self, path: str = None) -> dict:
        defaults = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'initial_capital': 10000.0,
            'window': 120,
            'max_position_pct': 0.5,
            'stop_loss_pct': 0.0015,
            'take_profit_pct': 0.003,
            'enable_ppo': True,
            'warmup_bars': 200,
            'state_dim': 11,
            'min_trade_amount_usdt': 5.0,
            'trailing_activation_pct': 0.002,
            'trailing_callback_pct': 0.001,
            'tp_multiplier': 1.1,
            'tp_cap_pct': 0.005,
            'entry_quality_threshold_long': 0.3,
            'entry_quality_threshold_short': 0.3,
            'trading_mode': 'spot',
            'leverage': 10,
            'ppo_actions': {
                'buy_levels': [0.10, 0.30, 0.50],
                'sell_levels': [0.10, 0.30, 0.50]
            }
        }
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    user_cfg = json.load(f)
                defaults.update(user_cfg)
                logger.info(f"Config loaded from {path}")
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        return defaults

    def _parse_timestamp(self, series: pd.Series) -> pd.Series:
        for kwargs in [{'utc': True, 'infer_datetime_format': True}, {'unit': 'ms', 'utc': True}, {'unit': 's', 'utc': True}]:
            try:
                return pd.to_datetime(series, **kwargs)
            except Exception:
                continue
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize('UTC')
        else:
            parsed = parsed.dt.tz_convert('UTC')
        return parsed

    def _load_feature_lists(self, features_path: str) -> bool:
        try:
            with open(features_path, 'r') as f:
                data = json.load(f)
            self.feature_lists['all'] = data.get('all_features', [])
            self.feature_lists['lstm'] = data.get('lstm_features', [])
            self.feature_lists['ensemble'] = data.get('ensemble_features', [])
            self.feature_lists['regime'] = data.get('regime_features', [])
            self.embedding_dim = data.get('embedding_dim', 16)
            logger.info(f"Loaded feature lists: LSTM({len(self.feature_lists['lstm'])}) "
                        f"Ensemble({len(self.feature_lists['ensemble'])}) "
                        f"Regime({len(self.feature_lists['regime'])}) "
                        f"Embedding dim: {self.embedding_dim}")
            return True
        except Exception as e:
            logger.error(f"Feature list load failed: {e}")
            return False

    def _discover_and_load_models(self, models_dir: str = None) -> bool:
        found = FileDiscovery.find_models(models_dir or '.')
        required = ['lstm', 'scaler']
        if not all(k in found for k in required):
            logger.error(f"Missing required models. Found: {found.keys()}")
            return False

        try:
            from tensorflow.keras.models import load_model
            import joblib

            # Critical 1: Load LSTM as PredictionModel wrapper
            lstm_wrapper = PredictionModel(self.config)
            lstm_wrapper.load(found['lstm'])
            self.models['lstm_wrapper'] = lstm_wrapper
            self.models['lstm'] = lstm_wrapper.model  # raw Keras model for predict

            self.models['scaler'] = joblib.load(found['scaler'])
            if hasattr(self.models['scaler'], 'feature_names_in_'):
                self.models['scaler_features'] = list(self.models['scaler'].feature_names_in_)
            else:
                self.models['scaler_features'] = []

            # Load feature lists from final_features.json
            if 'features' in found:
                if not self._load_feature_lists(found['features']):
                    logger.warning("Feature list load failed, using default lists.")
            else:
                logger.warning("final_features.json not found. Using empty feature lists.")
                self.feature_lists = {'lstm': [], 'ensemble': [], 'regime': [], 'all': []}

            # Load Expert Ensemble
            expert_ensemble = ExpertEnsemble(self.config)
            base_path = os.path.dirname(found.get('expert_direction', 'models/expert_ensemble_direction.pkl'))
            if not os.path.exists(os.path.join(base_path, 'expert_ensemble_direction.pkl')):
                if 'expert_direction' in found:
                    base_path = os.path.dirname(found['expert_direction'])
                else:
                    expert_files = [v for k, v in found.items() if k.startswith('expert_')]
                    if expert_files:
                        base_path = os.path.dirname(expert_files[0])
                    else:
                        raise FileNotFoundError("No expert ensemble files found.")
            
            if expert_ensemble.load(base_path):
                self.models['expert_ensemble'] = expert_ensemble
                logger.info("Loaded ExpertEnsemble")
            else:
                raise RuntimeError(f"ExpertEnsemble load failed from {base_path}")

            if 'ppo_checkpoint' in found:
                try:
                    ppo = PPOAgent(cfg=self.config, state_dim=self.config.get('state_dim', 11))
                    if ppo.load(found['ppo_checkpoint']):
                        self.models['ppo_agent'] = ppo
                        logger.info("Loaded PPO from checkpoint.")
                    else:
                        logger.warning("PPO load failed.")
                except Exception as e:
                    logger.warning(f"PPO load exception: {e}")

            return True
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False

    def _discover_and_load_data(self, data_path: str = None) -> bool:
        found = FileDiscovery.find_data(data_path)
        if 'ohlcv' not in found:
            logger.error("No OHLCV data found")
            return False
        try:
            df = pd.read_csv(found['ohlcv'])
            df.columns = [c.strip().lower() for c in df.columns]
            if 'timestamp' in df.columns:
                df['timestamp'] = self._parse_timestamp(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return False
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.data['raw_df'] = df
            return True
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return False

    def _prepare_features_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            fe = FeatureEngine(cfg=self.config)
            df_copy = df.copy()
            if 'timestamp' in df_copy.columns:
                df_copy = df_copy.set_index(pd.to_datetime(df_copy['timestamp'], utc=True))
            features = fe.build_all(df_copy)
            features = features.reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    features[col] = df[col].values
            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            return features.ffill().fillna(0.0)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df

    def _reorder_df(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        available = [c for c in cols if c in df.columns]
        if len(available) < len(cols):
            missing = [c for c in cols if c not in df.columns]
            logger.warning(f"Missing columns for ordering: {missing}")
        return df[available] if available else df

    def _get_embedding(self, feat_slice: pd.DataFrame, window: int) -> np.ndarray:
        embedding = np.zeros(self.embedding_dim)
        if 'lstm_wrapper' not in self.models or 'scaler' not in self.models:
            return embedding
        try:
            lstm_wrapper = self.models['lstm_wrapper']
            scaler = self.models['scaler']
            feat_cols = self.feature_lists.get('lstm', [])
            if not feat_cols or not all(c in feat_slice.columns for c in feat_cols):
                return embedding

            # Critical 3: Enforce column order for scaler
            if hasattr(scaler, 'feature_names_in_'):
                scaler_cols = list(scaler.feature_names_in_)
                # Use only columns that exist in both scaler and feat_slice
                common = [c for c in scaler_cols if c in feat_slice.columns]
                if not common:
                    return embedding
                num_df = feat_slice[common].ffill().fillna(0.0)
                # Ensure order matches scaler's training order
                num_df = num_df[scaler_cols]
            else:
                num_df = feat_slice[feat_cols].ffill().fillna(0.0)

            scaled = scaler.transform(num_df)
            if len(scaled) >= window:
                X = scaled[-window:].reshape(1, window, -1)
                X_multi = lstm_wrapper._split_to_multi_input(X)
                emb_out = lstm_wrapper.model(X_multi, training=False)
                if isinstance(emb_out, list):
                    emb_out = emb_out[2]
                elif hasattr(emb_out, 'numpy'):
                    emb_out = emb_out.numpy()
                if len(emb_out) > 0:
                    embedding = emb_out[-1].astype(np.float32)
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
        return embedding

    def _get_regime_probs(self, feat_slice: pd.DataFrame, reg_cols: List[str]) -> np.ndarray:
        default = np.array([0.33, 0.33, 0.34])
        if 'regime_detector' not in self.models or not reg_cols:
            return default
        try:
            # Critical 4: Enforce column order for regime detector
            detector = self.models['regime_detector']
            # Use detector's feature_list for order
            det_cols = detector.feature_list
            available = [c for c in det_cols if c in feat_slice.columns]
            if not available:
                return default
            reg_df = feat_slice[available].ffill().fillna(0.0)
            # Reorder to match detector's expected order
            reg_df = reg_df[det_cols]
            if len(reg_df) > 0:
                row = reg_df.iloc[-1].values.astype(np.float32)
                res = detector.predict_live(row)
                probs = res.get('probs', [0.33, 0.33, 0.34])
                return np.array(probs[:3], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Regime probs failed: {e}")
        return default

    def _get_expert_signals(self, feat_slice: pd.DataFrame, embedding: np.ndarray,
                             ens_cols: List[str]) -> Dict[str, float]:
        default = {'direction_prob': 0.5, 'expected_move': 0.003,
                   'entry_quality': 0.5, 'exit_distance': 0.02}
        if 'expert_ensemble' not in self.models:
            return default
        try:
            # Critical 5: Enforce column order for ensemble
            # Use the exact order from stored feature list
            available = [c for c in ens_cols if c in feat_slice.columns]
            if not available:
                return default
            raw_vals = []
            for col in ens_cols:
                val = feat_slice.iloc[-1].get(col, 0.0)
                raw_vals.append(float(val) if not pd.isna(val) else 0.0)
            combined = np.concatenate([np.array(raw_vals), embedding]).reshape(1, -1)
            sigs = self.models['expert_ensemble'].predict_expert_signals(combined)
            return {
                'direction_prob': float(sigs.get('direction_prob', 0.5)),
                'expected_move': float(sigs.get('expected_move', 0.003)),
                'entry_quality': float(sigs.get('entry_quality', 0.5)),
                'exit_distance': float(sigs.get('exit_distance', 0.02))
            }
        except Exception as e:
            logger.debug(f"Ensemble signals failed: {e}")
        return default

    def _get_ppo_action(self, expert: Dict, regime_probs: np.ndarray,
                        pos_info: Dict, l2_val: float) -> Tuple[int, str, float]:
        default = (0, 'HOLD', 0.0)
        if 'ppo_agent' not in self.models:
            return default
        try:
            state = np.array([
                expert['direction_prob'],
                expert['entry_quality'],
                expert['expected_move'],
                expert['exit_distance'],
                l2_val,
                regime_probs[0], regime_probs[1], regime_probs[2],
                pos_info['position_status'],
                pos_info['pnl_pct'],
                pos_info['available_margin']
            ], dtype=np.float32)
            action_idx, _ = self.models['ppo_agent'].act(state, greedy=True)
            buy_levels = self.config.get('ppo_actions', {}).get('buy_levels', [0.10, 0.30, 0.50])
            sell_levels = self.config.get('ppo_actions', {}).get('sell_levels', [0.10, 0.30, 0.50])
            n_buy = len(buy_levels)
            if action_idx == 0:
                return action_idx, 'HOLD', 0.0
            elif action_idx <= n_buy:
                return action_idx, 'BUY', buy_levels[action_idx - 1]
            elif action_idx <= n_buy + len(sell_levels):
                return action_idx, 'SELL', sell_levels[action_idx - n_buy - 1]
            else:
                return action_idx, 'CLOSE_ALL', 1.0
        except Exception as e:
            logger.debug(f"PPO action failed: {e}")
        return default

    def _run_backtest_loop(self, reset_risk: bool = True) -> Dict:
        df_raw = self.data['df']
        df_feats = self.data['full_feats']
        
        assert len(df_raw) == len(df_feats), "Raw and Feature DataFrames length mismatch!"

        initial_capital = float(self.config.get('initial_capital', 10000.0))
        if initial_capital <= 0:
            initial_capital = 10000.0

        fee_rate = self.config.get('fee_rate', 0.001)
        slippage = self.config.get('slippage', 0.0005)
        window = self.config.get('window', 120)
        tp_multiplier = self.config.get('tp_multiplier', 1.1)
        tp_cap = self.config.get('tp_cap_pct', 0.005)
        entry_threshold_long = self.config.get('entry_quality_threshold_long', 0.3)
        entry_threshold_short = self.config.get('entry_quality_threshold_short', 0.3)
        min_trade_usdt = self.config.get('min_trade_amount_usdt', 5.0)
        trading_mode = self.config.get('trading_mode', 'spot')
        leverage = self.config.get('leverage', 10)

        if reset_risk:
            self.risk_manager = RiskManager(self.config)
            self.risk_manager.consecutive_losses = 0

        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        side = 'long'
        dynamic_sl = 0.0
        dynamic_tp = 0.0
        trailing_peak = 0.0
        trailing_activated = False
        trades = []
        portfolio = [capital]
        total_bars = len(df_raw)

        regime_cols = self.feature_lists.get('regime', [])
        ens_cols = self.feature_lists.get('ensemble', [])

        if 'regime_detector' not in self.models:
            try:
                from regime_detector import MarketRegimeDetector
                rd = MarketRegimeDetector(self.config)
                if rd.load_map():
                    self.models['regime_detector'] = rd
                    logger.info("Regime detector loaded dynamically.")
            except Exception as e:
                logger.warning(f"Regime detector load failed: {e}")

        for i in tqdm(range(window, total_bars), desc="Running Backtest", unit="bar"):
            try:
                raw_row = df_raw.iloc[i]
                feat_row = df_feats.iloc[i]

                close_p = float(raw_row['close'])
                high_p = float(raw_row['high'])
                low_p = float(raw_row['low'])

                if any(p <= 0 or np.isnan(p) for p in (close_p, high_p, low_p)):
                    portfolio.append(portfolio[-1])
                    continue

                # Update unrealized PnL and margin (with leverage for futures)
                if position > 0:
                    if side == 'long':
                        pnl_pct = (close_p - entry_price) / entry_price
                        total_equity = capital + position * close_p
                    else:
                        pnl_pct = (entry_price - close_p) / entry_price
                        total_equity = capital + position * (entry_price - close_p)
                    # Margin calculation: if futures, required margin = notional / leverage
                    if trading_mode == 'future':
                        notional = position * close_p
                        required_margin = notional / leverage
                        available_margin = capital / (required_margin + 1e-10)
                    else:
                        available_margin = capital / (total_equity + 1e-10)
                    position_status = 1 if side == 'long' else -1
                else:
                    pnl_pct = 0.0
                    available_margin = 1.0
                    position_status = 0
                    total_equity = capital

                # Check Stop Loss / Take Profit
                if position > 0:
                    # Trailing stop logic
                    if side == 'long':
                        if high_p > trailing_peak:
                            trailing_peak = high_p
                        profit_pct = (close_p - entry_price) / entry_price
                        if profit_pct >= self.config.get('trailing_activation_pct', 0.002):
                            trailing_activated = True
                        if trailing_activated:
                            new_sl = trailing_peak * (1 - self.config.get('trailing_callback_pct', 0.001))
                            if new_sl > dynamic_sl:
                                dynamic_sl = new_sl
                    else:
                        if low_p < trailing_peak or trailing_peak == 0.0:
                            trailing_peak = low_p
                        profit_pct = (entry_price - close_p) / entry_price
                        if profit_pct >= self.config.get('trailing_activation_pct', 0.002):
                            trailing_activated = True
                        if trailing_activated:
                            new_sl = trailing_peak * (1 + self.config.get('trailing_callback_pct', 0.001))
                            if new_sl < dynamic_sl or dynamic_sl == 0.0:
                                dynamic_sl = new_sl

                    if (side == 'long' and low_p <= dynamic_sl) or (side == 'short' and high_p >= dynamic_sl):
                        exit_price = dynamic_sl * (1 - slippage) if side == 'long' else dynamic_sl * (1 + slippage)
                        realized_pnl = (exit_price - entry_price) / entry_price if side == 'long' else (entry_price - exit_price) / entry_price
                        gross_revenue = position * exit_price
                        fee = gross_revenue * fee_rate
                        net_revenue = gross_revenue - fee
                        capital += net_revenue
                        trades.append({'type':'sell','price':exit_price,'pnl':realized_pnl,'bar':i,'reason':'stop_loss'})
                        self.risk_manager.register_trade(realized_pnl)
                        position = 0.0
                        portfolio.append(capital)
                        continue

                    if (side == 'long' and high_p >= dynamic_tp) or (side == 'short' and low_p <= dynamic_tp):
                        exit_price = dynamic_tp * (1 + slippage) if side == 'long' else dynamic_tp * (1 - slippage)
                        realized_pnl = (exit_price - entry_price) / entry_price if side == 'long' else (entry_price - exit_price) / entry_price
                        gross_revenue = position * exit_price
                        fee = gross_revenue * fee_rate
                        net_revenue = gross_revenue - fee
                        capital += net_revenue
                        trades.append({'type':'sell','price':exit_price,'pnl':realized_pnl,'bar':i,'reason':'take_profit'})
                        self.risk_manager.register_trade(realized_pnl)
                        position = 0.0
                        portfolio.append(capital)
                        continue

                # Get Signal from cached features
                pos_info = {
                    'position_status': position_status,
                    'pnl_pct': pnl_pct,
                    'available_margin': available_margin
                }

                feat_slice = df_feats.iloc[max(0, i-window):i+1]
                raw_slice = df_raw.iloc[max(0, i-window):i+1]
                if len(raw_slice) < window:
                    portfolio.append(portfolio[-1])
                    continue

                # Extract features for signal generation
                l2_val = float(feat_slice.iloc[-1].get('l2_proxy', 0.0))
                embedding = self._get_embedding(feat_slice, window)
                regime_probs = self._get_regime_probs(feat_slice, regime_cols)
                expert = self._get_expert_signals(feat_slice, embedding, ens_cols)
                action_idx, action_str, size = self._get_ppo_action(expert, regime_probs, pos_info, l2_val)

                # Adaptive SL/TP using feature row
                natr = float(feat_row.get('natr', 1.0))
                if natr <= 0:
                    natr = 1.0
                # Critical 6: natr is in %, so divide by 100 to get ratio
                volatility_ratio = natr / 100.0
                regime = 0
                if 'regime_p_0' in feat_row.index:
                    probs = [feat_row.get('regime_p_0', 0.33), feat_row.get('regime_p_1', 0.33), feat_row.get('regime_p_2', 0.34)]
                    regime = int(np.argmax(probs))
                adx = float(feat_row.get('adx', 25.0))
                if adx <= 0:
                    adx = 25.0
                hurst = float(feat_row.get('hurst_exp', 0.5))
                if hurst <= 0 or hurst > 1:
                    hurst = 0.5

                sl_pct = self.risk_manager.get_stop_loss_pct(volatility_ratio, regime, adx)
                tp_pct = self.risk_manager.get_take_profit_pct(volatility_ratio, regime, adx, hurst)
                expected_move = expert.get('expected_move', 0.003)
                exit_distance = expert.get('exit_distance', 0.02)
                # Use exit_distance if available and reasonable
                if exit_distance > 0:
                    tp_pct = max(tp_pct, exit_distance * 0.8)  # use 80% of exit distance as TP
                if expected_move > 0:
                    tp_pct = max(tp_pct, expected_move * tp_multiplier)
                tp_pct = min(tp_pct, tp_cap)

                # Execute Entry
                entry_quality = expert.get('entry_quality', 0.0)
                if position == 0:
                    if action_str in ['BUY', 'LONG'] and entry_quality > entry_threshold_long:
                        trade_size = self.risk_manager.get_position_size(capital, confidence=entry_quality, volatility=volatility_ratio)
                        trade_size = min(trade_size, capital * self.config.get('max_position_pct', 0.5))
                        if trade_size > min_trade_usdt:
                            fee = trade_size * fee_rate
                            total_cost = trade_size + fee
                            if total_cost <= capital:
                                buy_price = close_p * (1 + slippage)
                                position = trade_size / buy_price
                                capital -= total_cost
                                entry_price = buy_price
                                side = 'long'
                                dynamic_sl = entry_price * (1 - sl_pct)
                                dynamic_tp = entry_price * (1 + tp_pct)
                                trailing_peak = entry_price
                                trailing_activated = False
                                trades.append({'type':'buy','price':buy_price,'bar':i,'quality':entry_quality,'cash_spent':total_cost})
                                portfolio.append(capital + position * close_p)
                                continue

                    elif trading_mode == 'future' and action_str in ['SHORT', 'SELL'] and entry_quality > entry_threshold_short:
                        trade_size = self.risk_manager.get_position_size(capital, confidence=entry_quality, volatility=volatility_ratio)
                        trade_size = min(trade_size, capital * self.config.get('max_position_pct', 0.5))
                        if trade_size > min_trade_usdt:
                            fee = trade_size * fee_rate
                            total_cost = trade_size + fee
                            if total_cost <= capital:
                                sell_price = close_p * (1 - slippage)
                                position = trade_size / sell_price
                                capital -= total_cost
                                entry_price = sell_price
                                side = 'short'
                                dynamic_sl = entry_price * (1 + sl_pct)
                                dynamic_tp = entry_price * (1 - tp_pct)
                                trailing_peak = entry_price
                                trailing_activated = False
                                trades.append({'type':'sell','price':sell_price,'bar':i,'quality':entry_quality,'cash_spent':total_cost})
                                portfolio.append(capital + position * (entry_price - close_p))
                                continue

                elif position > 0:
                    if action_str in ['SELL', 'COVER', 'CLOSE_ALL']:
                        exit_price = close_p * (1 - slippage) if side == 'long' else close_p * (1 + slippage)
                        realized_pnl = (exit_price - entry_price) / entry_price if side == 'long' else (entry_price - exit_price) / entry_price
                        gross_revenue = position * exit_price
                        fee = gross_revenue * fee_rate
                        net_revenue = gross_revenue - fee
                        capital += net_revenue
                        trades.append({'type':'sell','price':exit_price,'pnl':realized_pnl,'bar':i,'reason':'signal'})
                        self.risk_manager.register_trade(realized_pnl)
                        position = 0.0

                if position > 0:
                    if side == 'long':
                        port_val = capital + position * close_p
                    else:
                        port_val = capital + position * (entry_price - close_p)
                else:
                    port_val = capital
                portfolio.append(port_val)

            except Exception as e:
                logger.error(f"Error at bar {i}: {e}")
                portfolio.append(portfolio[-1])

        # Forced close at end
        if position > 0:
            exit_price = df_raw['close'].iloc[-1] * (1 - slippage)
            realized_pnl = (exit_price - entry_price) / entry_price if side == 'long' else (entry_price - exit_price) / entry_price
            gross_revenue = position * exit_price
            fee = gross_revenue * fee_rate
            net_revenue = gross_revenue - fee
            capital += net_revenue
            trades.append({'type':'sell','price':exit_price,'pnl':realized_pnl,'bar':total_bars-1,'reason':'forced_close'})
            self.risk_manager.register_trade(realized_pnl)
            portfolio[-1] = capital

        # Metrics
        tf = self.config.get('timeframe', '1h').lower()
        tf_map = {'1m':365*24*60,'5m':365*24*12,'15m':365*24*4,'30m':365*24*2,'1h':365*24,'4h':365*6,'1d':365}
        ann_factor = tf_map.get(tf, 365*24)
        rets = np.diff(portfolio) / (np.array(portfolio[:-1]) + 1e-10)
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(ann_factor)) if len(rets) else 0.0
        neg = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg) + 1e-10) * np.sqrt(ann_factor)) if len(neg) else sharpe
        max_dd = float(((np.array(portfolio) - np.maximum.accumulate(portfolio)) / (np.maximum.accumulate(portfolio) + 1e-10)).min())
        sell_trades = [t for t in trades if t['type'] == 'sell']
        wins = [t['pnl'] for t in sell_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in sell_trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(sell_trades) if sell_trades else 0.0
        profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else (float('inf') if sum(wins) > 0 else 0.0)
        avg_pnl = float(np.mean([t['pnl'] for t in sell_trades])) if sell_trades else 0.0

        return {
            'total_return': (portfolio[-1] - initial_capital) / initial_capital * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': avg_pnl,
            'total_trades': len(sell_trades),
            'final_capital': portfolio[-1],
            'portfolio': portfolio,
            'trades': trades
        }

    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        start_time = time.time()
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " BACKTEST ENGINE v10.0 – FINAL PRODUCTION ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")

        if not self._discover_and_load_models(models_dir):
            return {'error': 'Model loading failed'}
        if not self._discover_and_load_data(data_path):
            return {'error': 'Data loading failed'}

        warmup = self.warmup_bars
        raw_df = self.data['raw_df']
        if len(raw_df) <= warmup:
            return {'error': f'Data too short: {len(raw_df)} <= {warmup}'}

        self.data['df'] = raw_df.iloc[warmup:].reset_index(drop=True)
        
        logger.info("Pre-computing features for entire dataset...")
        self.data['full_feats'] = self._prepare_features_signal(self.data['df'])
        logger.info(f"Features computed. Shape: {self.data['full_feats'].shape}")

        results = self._run_backtest_loop(reset_risk=True)
        elapsed = time.time() - start_time

        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " BACKTEST RESULTS ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        for k, v, unit in [
            ("Total Return", results.get('total_return', 0), "%"),
            ("Sharpe Ratio", results.get('sharpe', 0), ""),
            ("Sortino Ratio", results.get('sortino', 0), ""),
            ("Max Drawdown", results.get('max_drawdown', 0)*100, "%"),
            ("Win Rate", results.get('win_rate', 0)*100, "%"),
            ("Profit Factor", results.get('profit_factor', 0), ""),
            ("Total Trades", results.get('total_trades', 0), ""),
            ("Avg PnL", results.get('avg_pnl', 0)*100, "%"),
            ("Final Capital", results.get('final_capital', 0), " $"),
        ]:
            print(f"║  {k:<25}: {v:>18.4f}{unit:<4} ║")
        print("╠" + "═" * 78 + "╣")
        print(f"║  Elapsed Time: {elapsed:.2f} seconds".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        return results

class WalkForwardValidator:
    def __init__(self, config_path: str = None):
        self.backtest = BacktestRunner(config_path)

    def run_validation(self, models_dir: str = None, data_path: str = None) -> Tuple[bool, Dict]:
        logger.info("Starting Walk-Forward Validation (4 folds)...")
        if not self.backtest._discover_and_load_models(models_dir):
            return False, {'error': 'Model loading failed'}
        found = FileDiscovery.find_data(data_path)
        if 'ohlcv' not in found:
            return False, {'error': 'No OHLCV data found'}
        try:
            df_raw = pd.read_csv(found['ohlcv'])
            df_raw.columns = [c.strip().lower() for c in df_raw.columns]
            if 'timestamp' in df_raw.columns:
                df_raw['timestamp'] = self.backtest._parse_timestamp(df_raw['timestamp'])
            for col in ['open','high','low','close','volume']:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            df_raw.dropna(subset=['close'], inplace=True)
            df_raw.reset_index(drop=True, inplace=True)
        except Exception as e:
            return False, {'error': f'Data load failed: {e}'}

        warmup = self.backtest.warmup_bars
        window = self.backtest.config.get('window', 120)
        
        if len(df_raw) <= warmup + window:
            return False, {'error': 'Data too short'}

        logger.info("Pre-computing features for walk-forward validation...")
        full_feats = self.backtest._prepare_features_signal(df_raw)
        
        test_start = warmup
        total_bars = len(df_raw) - test_start
        num_folds = 4
        fold_size = total_bars // num_folds
        folds_metrics = []
        fold_results = []

        for i in range(num_folds):
            start = test_start + (i * fold_size)
            fold_end = test_start + ((i+1) * fold_size) if i < num_folds-1 else len(df_raw)
            slice_start = max(0, start - window - 50)
            
            raw_slice = df_raw.iloc[slice_start:fold_end].copy().reset_index(drop=True)
            feat_slice = full_feats.iloc[slice_start:fold_end].copy().reset_index(drop=True)
            
            trim_start = start - slice_start
            raw_test = raw_slice.iloc[trim_start:].reset_index(drop=True)
            feat_test = feat_slice.iloc[trim_start:].reset_index(drop=True)
            
            if len(raw_test) < window:
                logger.warning(f"Fold {i+1} too short ({len(raw_test)} < {window}). Skipping.")
                continue

            logger.info(f"Running Fold {i+1}/{num_folds} ({len(raw_test)} test bars)...")
            self.backtest.data['df'] = raw_test
            self.backtest.data['full_feats'] = feat_test
            res = self.backtest._run_backtest_loop(reset_risk=True)
            folds_metrics.append(res.get('sharpe', 0))
            fold_results.append(res)
            logger.info(f"  Fold {i+1} Sharpe: {res['sharpe']:.4f}")

        if not folds_metrics:
            logger.error("All folds failed")
            return False, {'mean_sharpe': 0.0, 'folds': []}

        mean_sharpe = np.mean(folds_metrics)
        passed = mean_sharpe > 0.5 and min(folds_metrics) > 0
        logger.info(f"Walk-Forward Verdict: {'PASSED' if passed else 'FAILED'} (Mean Sharpe: {mean_sharpe:.4f})")
        return passed, {'mean_sharpe': mean_sharpe, 'folds': folds_metrics, 'details': fold_results}

def main():
    parser = argparse.ArgumentParser(description='Backtest Engine v10.0 – FINAL PRODUCTION')
    parser.add_argument('--mode', required=True, choices=['backtest', 'validate'])
    parser.add_argument('--models', default=None, help='Models directory')
    parser.add_argument('--data', default=None, help='Data directory or file')
    parser.add_argument('--config', default=None, help='Config file path')
    args = parser.parse_args()

    if args.mode == 'backtest':
        runner = BacktestRunner(config_path=args.config)
        runner.run(models_dir=args.models, data_path=args.data)
    else:
        validator = WalkForwardValidator(config_path=args.config)
        validator.run_validation(models_dir=args.models, data_path=args.data)

if __name__ == '__main__':
    main()