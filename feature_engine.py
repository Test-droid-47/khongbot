import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings
from typing import Dict, Optional, List

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Robust dummy decorator preserving signatures when Numba is missing
    def jit(*args, **kwargs):
        return lambda func: func

class FeatureEngine:

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.window = self.cfg.get('hurst_window', 100)
        self.ob_lookback = self.cfg.get('ob_lookback', 20)
        self.atr_period = self.cfg.get('atr_period', 14)
        self.fvg_atr_mult = self.cfg.get('fvg_atr_mult', 0.2)
        self.max_active_fvgs = self.cfg.get('max_active_fvgs', 500)
        self.weight_bos = self.cfg.get('smc_weight_bos', 10.0)
        self.weight_choch = self.cfg.get('smc_weight_choch', 15.0)
        self.weight_fvg = self.cfg.get('smc_weight_fvg', 8.0)
        self.weight_premium = self.cfg.get('smc_weight_premium', 5.0)
        self.weight_liq = self.cfg.get('smc_weight_liq', 12.0)
        
        # FIX 7: Alert production environment if Numba performance optimization is unavailable
        if not HAS_NUMBA:
            warnings.warn(
                "CRITICAL PERFORMANCE WARNING: Numba is not installed. "
                "SMC core loops and Hurst vectorization will run in pure Python, "
                "which will cause substantial execution lag on large datasets.",
                RuntimeWarning
            )

    @staticmethod
    @jit(nopython=True)
    def _hurst_rs_vectorized(price_series: np.ndarray, window: int) -> np.ndarray:
        n = len(price_series)
        hurst = np.full(n, 0.5, dtype=np.float64)
        if n < window + 10:
            return hurst
        log_prices = np.log(price_series)
        for i in range(window, n):
            segment = log_prices[i-window+1:i+1]
            mean_centered = segment - np.mean(segment)
            cumsum = np.cumsum(mean_centered)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.sqrt(np.sum((segment - np.mean(segment))**2) / (len(segment) - 1))
            if s > 1e-10:
                rs = r / s
                h = np.log(rs) / np.log(window)
                hurst[i] = max(0.0, min(1.0, h))
        return hurst

    @staticmethod
    @jit(nopython=True)
    def _compute_smc_core(highs, lows, closes, opens, atr, lookback, fvg_atr_mult, max_fvgs):
        n = len(highs)
        bos_bull = np.zeros(n, np.int8)
        bos_bear = np.zeros(n, np.int8)
        choch_bull = np.zeros(n, np.int8)
        choch_bear = np.zeros(n, np.int8)
        ob_bull_level = np.zeros(n, np.float32)
        ob_bear_level = np.zeros(n, np.float32)
        ob_bull_strength = np.zeros(n, np.float32)
        ob_bear_strength = np.zeros(n, np.float32)
        fvg_bull_sz = np.zeros(n, np.float32)
        fvg_bear_sz = np.zeros(n, np.float32)
        fvg_bull_fill = np.zeros(n, np.float32)
        fvg_bear_fill = np.zeros(n, np.float32)
        liq_sweep_bull = np.zeros(n, np.int8)
        liq_sweep_bear = np.zeros(n, np.int8)
        
        active_bull_l = np.zeros(max_fvgs, np.float32)
        active_bull_h = np.zeros(max_fvgs, np.float32)
        active_bull_sz = np.zeros(max_fvgs, np.float32)
        bull_count = 0
        
        active_bear_l = np.zeros(max_fvgs, np.float32)
        active_bear_h = np.zeros(max_fvgs, np.float32)
        active_bear_sz = np.zeros(max_fvgs, np.float32)
        bear_count = 0

        last_dir = 0
        last_bos_idx = -1
        active_bull_lvl = 0.0
        active_bear_lvl = 0.0
        active_bull_str = 0.0
        active_bear_str = 0.0

        for i in range(lookback, n):
            if last_bos_idx != -1 and (i - last_bos_idx) > (lookback * 2):
                last_dir = 0
            sh = np.max(highs[i-lookback:i])
            sl = np.min(lows[i-lookback:i])
            if closes[i] > sh:
                bos_bull[i] = 1
                last_bos_idx = i
                if last_dir == -1:
                    choch_bull[i] = 1
                last_dir = 1
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] < opens[j]:
                        active_bull_lvl = lows[j]
                        active_bull_str = (highs[j] - lows[j]) / (atr[j] + 1e-10)
                        break
            elif closes[i] < sl:
                bos_bear[i] = 1
                last_bos_idx = i
                if last_dir == 1:
                    choch_bear[i] = 1
                last_dir = -1
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] > opens[j]:
                        active_bear_lvl = highs[j]
                        active_bear_str = (highs[j] - lows[j]) / (atr[j] + 1e-10)
                        break
            if active_bull_lvl > 0.0 and closes[i] < active_bull_lvl:
                active_bull_lvl = 0.0
                active_bull_str = 0.0
            if active_bear_lvl > 0.0 and closes[i] > active_bear_lvl:
                active_bear_lvl = 0.0
                active_bear_str = 0.0
            ob_bull_level[i] = active_bull_lvl
            ob_bear_level[i] = active_bear_lvl
            ob_bull_strength[i] = active_bull_str
            ob_bear_strength[i] = active_bear_str

            # FIX 4: Standardize SMC rules. Invalidate/mitigate FVGs immediately when touched 
            w_bull = 0
            for k in range(bull_count):
                if lows[i] > active_bull_l[k]:  # Stays active only if current low hasn't touched/breached bottom
                    active_bull_l[w_bull] = active_bull_l[k]
                    active_bull_h[w_bull] = active_bull_h[k]
                    active_bull_sz[w_bull] = active_bull_sz[k]
                    w_bull += 1
            bull_count = w_bull

            w_bear = 0
            for k in range(bear_count):
                if highs[i] < active_bear_h[k]: # Stays active only if current high hasn't touched/breached top
                    active_bear_l[w_bear] = active_bear_l[k]
                    active_bear_h[w_bear] = active_bear_h[k]
                    active_bear_sz[w_bear] = active_bear_sz[k]
                    w_bear += 1
            bear_count = w_bear

            min_gap = atr[i] * fvg_atr_mult
            if i >= 2:
                gap_bull = lows[i] - highs[i-2]
                if gap_bull > min_gap and bull_count < max_fvgs:
                    active_bull_l[bull_count] = highs[i-2]
                    active_bull_h[bull_count] = lows[i]
                    active_bull_sz[bull_count] = gap_bull
                    bull_count += 1
                gap_bear = lows[i-2] - highs[i]
                if gap_bear > min_gap and bear_count < max_fvgs:
                    active_bear_l[bear_count] = highs[i]
                    active_bear_h[bear_count] = lows[i-2]
                    active_bear_sz[bear_count] = gap_bear
                    bear_count += 1

            if bull_count > 0:
                tot_sz = 0.0
                tot_fill = 0.0
                for k in range(bull_count):
                    tot_sz += active_bull_sz[k]
                    denom = active_bull_h[k] - active_bull_l[k] + 1e-10
                    c_fill = (active_bull_h[k] - closes[i]) / denom
                    if c_fill < 0.0: c_fill = 0.0
                    if c_fill > 1.0: c_fill = 1.0
                    tot_fill += c_fill
                fvg_bull_sz[i] = tot_sz
                fvg_bull_fill[i] = tot_fill / bull_count

            if bear_count > 0:
                tot_sz = 0.0
                tot_fill = 0.0
                for k in range(bear_count):
                    tot_sz += active_bear_sz[k]
                    denom = active_bear_h[k] - active_bear_l[k] + 1e-10
                    c_fill = (closes[i] - active_bear_l[k]) / denom
                    if c_fill < 0.0: c_fill = 0.0
                    if c_fill > 1.0: c_fill = 1.0
                    tot_fill += c_fill
                fvg_bear_sz[i] = tot_sz
                fvg_bear_fill[i] = tot_fill / bear_count

            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            if lows[i] < recent_low and closes[i] > recent_low:
                liq_sweep_bull[i] = 1
            if highs[i] > recent_high and closes[i] < recent_high:
                liq_sweep_bear[i] = 1

        return (bos_bull, bos_bear, choch_bull, choch_bear,
                ob_bull_level, ob_bear_level, ob_bull_strength, ob_bear_strength,
                fvg_bull_sz, fvg_bear_sz, fvg_bull_fill, fvg_bear_fill,
                liq_sweep_bull, liq_sweep_bear)

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        original_timestamp = df['timestamp'].values if 'timestamp' in df.columns else None
        
        highs = df['high'].values.astype(np.float32)
        lows = df['low'].values.astype(np.float32)
        closes = df['close'].values.astype(np.float32)
        opens = df['open'].values.astype(np.float32)
        volumes = df['volume'].values.astype(np.float32)
        n = len(df)

        window = min(self.window, n // 2)
        if window >= 10:
            df['hurst_exp'] = self._hurst_rs_vectorized(df['close'].values.astype(float), window).astype(np.float32)
        else:
            df['hurst_exp'] = 0.5
        df['market_memory'] = (df['hurst_exp'] - 0.5).astype(np.float32)

        direction = (df['close'] - df['close'].shift(20)).abs()
        volatility = df['close'].diff().abs().rolling(20).sum()
        df['efficiency_ratio_20'] = (direction / (volatility + 1e-10)).fillna(0).astype(np.float32)

        log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
        realized_vol = log_ret.rolling(20).std() * np.sqrt(20)
        vol_pct = realized_vol.rolling(100).rank(pct=True).fillna(0.5)
        df['vol_regime_score'] = (vol_pct * 2 - 1).astype(np.float32)

        tr = np.zeros(n, np.float32)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values.astype(np.float32)

        res = self._compute_smc_core(highs, lows, closes, opens, atr, self.ob_lookback, self.fvg_atr_mult, self.max_active_fvgs)
        
        bos_bull, bos_bear, choch_bull, choch_bear = res[0], res[1], res[2], res[3]
        ob_bull_level, ob_bear_level = res[4], res[5]
        df['ob_bull_strength'], df['ob_bear_strength'] = res[6], res[7]
        df['smc_fvg_bull_size'], df['smc_fvg_bear_size'] = res[8], res[9]
        df['fvg_bull_fill'], df['fvg_bear_fill'] = res[10], res[11]
        liq_sweep_bull, liq_sweep_bear = res[12], res[13]

        df['smc_liq_sweep_bull'] = liq_sweep_bull.astype(np.int8)
        df['smc_liq_sweep_bear'] = liq_sweep_bear.astype(np.int8)

        df['close_vs_ob_bull'] = np.where(ob_bull_level > 0.0, (df['close'] - ob_bull_level) / (df['close'] + 1e-10), 0.0).astype(np.float32)
        df['close_vs_ob_bear'] = np.where(ob_bear_level > 0.0, (df['close'] - ob_bear_level) / (df['close'] + 1e-10), 0.0).astype(np.float32)

        roll_h = df['high'].rolling(50).max()
        roll_l = df['low'].rolling(50).min()
        eq = ((roll_h + roll_l) / 2.0).astype(np.float32)
        smc_premium = (df['close'] > eq).astype(np.float32)
        smc_discount = (df['close'] < eq).astype(np.float32)

        trend_w = df['hurst_exp'].values
        range_w = 1.0 - trend_w

        score = np.zeros(n, np.float32)
        score += bos_bull * self.weight_bos * trend_w
        score -= bos_bear * self.weight_bos * trend_w
        score += choch_bull * self.weight_choch * trend_w
        score -= choch_bear * self.weight_choch * trend_w
        score += (df['smc_fvg_bull_size'].values > 0) * self.weight_fvg * range_w * (1.0 - df['fvg_bull_fill'].values)
        score -= (df['smc_fvg_bear_size'].values > 0) * self.weight_fvg * range_w * (1.0 - df['fvg_bear_fill'].values)
        score -= smc_premium * self.weight_premium
        score += smc_discount * self.weight_premium
        score += liq_sweep_bull * self.weight_liq * range_w
        score -= liq_sweep_bear * self.weight_liq * range_w
        df['smc_score'] = score.astype(np.float32)

        df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)
        df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)
        
        body = (df['close'] - df['open']).abs()
        rng = df['high'] - df['low'] + 1e-10
        df['displacement_ratio'] = (body / rng).astype(np.float32)

        body_ratio = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        delta = df['volume'] * body_ratio
        df['vw_delta'] = (delta.rolling(20).mean() / (df['volume'].rolling(20).mean() + 1e-10)).fillna(0.0).astype(np.float32)

        if 'timestamp' in df.columns:
            df_dt = df[['timestamp', 'high', 'low']].copy()
            df_dt['timestamp'] = pd.to_datetime(df_dt['timestamp'])
            df_dt = df_dt.set_index('timestamp')
            df_4h = df_dt.resample('4h').agg({'high': 'max', 'low': 'min'})
            df_4h = df_4h.reindex(df_dt.index, method='ffill')
            df['htf_high'] = df_4h['high'].values
            df['htf_low'] = df_4h['low'].values
        else:
            df['htf_high'] = df['high'].rolling(24).max()
            df['htf_low'] = df['low'].rolling(24).min()
        
        # FIX 10: Clipped denominator bounds to completely eliminate division by zero infinity errors
        htf_range = (df['htf_high'] - df['htf_low']).clip(lower=1e-8)
        df['close_vs_htf'] = ((df['close'] - df['htf_low']) / htf_range).fillna(0.5).clip(0.0, 1.0).astype(np.float32)

        z_10_m = df['close'].rolling(10).mean()
        z_10_s = df['close'].rolling(10).std()
        z_10 = (df['close'] - z_10_m) / (z_10_s + 1e-10)
        z_50_m = df['close'].rolling(50).mean()
        z_50_s = df['close'].rolling(50).std()
        z_50 = (df['close'] - z_50_m) / (z_50_s + 1e-10)
        df['zscore_divergence'] = (z_10 - z_50).fillna(0).astype(np.float32)

        # Pure Native Vectorized Math Layer (Zero dependency, safe from silent failures)
        ema_v = df['volume'].ewm(span=20, adjust=False).mean()
        df['vol_ratio'] = (df['volume'] / (ema_v + 1e-10)).fillna(1.0).astype(np.float32)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        pv = df['volume'] * typical_price
        
        # FIX 13: Transitioned to institutional true cumulative session-anchored VWAP when timeline is available
        if 'timestamp' in df.columns:
            session_dates = pd.to_datetime(df['timestamp']).dt.date
            cum_pv = pv.groupby(session_dates).cumsum()
            cum_v = df['volume'].groupby(session_dates).cumsum()
            vwap = cum_pv / (cum_v + 1e-10)
        else:
            vwap = pv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
            
        df['close_vs_vwap'] = ((df['close'] - vwap) / (vwap + 1e-10)).fillna(0.0).astype(np.float32)
        
        candle_delta = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        cvd = pd.Series(candle_delta).rolling(window=100, min_periods=20).sum()
        cvd_ema = cvd.ewm(span=20, adjust=False).mean()
        df['cvd_trend'] = (cvd > cvd_ema).astype(np.int8)

        delta_close = df['close'].diff()
        gain = np.where(delta_close > 0, delta_close, 0.0)
        loss = np.where(delta_close < 0, -delta_close, 0.0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi_14'] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).astype(np.float32)
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        df['macd_hist_norm'] = (macd_hist / (pd.Series(atr) + 1e-10)).fillna(0).astype(np.float32)
            
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_mid + (2.0 * bb_std)
        bb_lower = bb_mid - (2.0 * bb_std)
        df['bb_width'] = ((bb_upper - bb_lower) / (bb_mid + 1e-10)).fillna(0).astype(np.float32)
        
        # FIX 9: Explicitly bounded the Bollinger Band position index to preserve structural safety for ML input steps
        df['bb_pct'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)).fillna(0.5).clip(0.0, 1.0).astype(np.float32)
            
        df['natr'] = ((pd.Series(atr) / (df['close'] + 1e-10)) * 100.0).fillna(0).astype(np.float32)
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        df['close_vs_ema_20'] = ((df['close'] - ema_20) / (ema_20 + 1e-10)).fillna(0).astype(np.float32)

        # FIX 2 & 6: Optimized genuine Order-3 Combinatorial Shannon Entropy implementation
        def true_perm_entropy(series: np.ndarray, window: int = 30) -> np.ndarray:
            n_len = len(series)
            ent = np.full(n_len, 0.5, dtype=np.float32)
            if n_len < window + 3:
                return ent
            
            for i in range(window, n_len):
                seg = series[i-window:i]
                m = window - 2
                patterns = np.zeros((m, 3))
                for k in range(3):
                    patterns[:, k] = seg[k:k+m]
                
                perms = np.argsort(patterns, axis=1)
                ids = perms[:, 0] * 9 + perms[:, 1] * 3 + perms[:, 2]
                
                _, counts = np.unique(ids, return_counts=True)
                probs = counts / m
                ent[i] = -np.sum(probs * np.log2(probs + 1e-12)) / np.log2(6.0)
            return ent

        def renko_velocity(close, brick_pct=0.003):
            n_len = len(close)
            velocity = np.zeros(n_len, dtype=np.float32)
            if n_len < 2:
                return velocity
            last_price = close[0]
            bricks = 0.0
            for i in range(1, n_len):
                change = (close[i] - last_price) / (last_price + 1e-10)
                if abs(change) >= brick_pct:
                    bricks += 1.0 if change > 0 else -1.0
                    last_price = close[i]
                bricks *= 0.98
                velocity[i] = np.clip(bricks / 5.0, -1.0, 1.0)
            return velocity

        df['amihud_illiq'] = (abs(df['close'].pct_change()) / (df['volume'] * df['close'] + 1e-10)).rolling(20).mean().fillna(0).astype(np.float32)

        # FIX 8: Switched Vol-of-Vol to continuous log-differencing to prevent flat market threshold exceptions
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std()
        df['vol_of_vol'] = np.log((vol + 1e-8) / (vol.shift(1) + 1e-8)).fillna(0.0).clip(-5.0, 5.0).astype(np.float32)

        df['perm_entropy'] = true_perm_entropy(df['close'].values, window=30)
        df['renko_velocity'] = renko_velocity(df['close'].values, brick_pct=0.003)

        df['l2_proxy'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['l2_proxy'] = df['l2_proxy'].clip(-1, 1).astype(np.float32)

        # FIX 3: Fully appended 'cvd_trend' to resolve computational logic waste
        elite_features = [
            'hurst_exp', 'market_memory', 'efficiency_ratio_20', 'vol_regime_score',
            'smc_score', 'smc_fvg_bull_size', 'smc_fvg_bear_size', 'fvg_bull_fill', 'fvg_bear_fill',
            'smc_liq_sweep_bull', 'smc_liq_sweep_bear', 'close_vs_ob_bull', 'close_vs_ob_bear',
            'ob_bull_strength', 'ob_bear_strength', 'buying_pressure', 'selling_pressure',
            'displacement_ratio', 'vw_delta', 'vol_ratio', 'close_vs_vwap', 'cvd_trend', 'rsi_14',
            'macd_hist_norm', 'bb_width', 'bb_pct', 'natr', 'close_vs_ema_20',
            'zscore_divergence', 'close_vs_htf',
            'amihud_illiq', 'vol_of_vol', 'perm_entropy', 'renko_velocity', 'l2_proxy'
        ]

        output_df = df[elite_features].copy()
        
        # FIX 1: Completely removed .bfill() to ensure zero future data leakage. Clean forward fill only.
        output_df = output_df.ffill().fillna(0.0)

        if original_timestamp is not None:
            output_df.insert(0, 'timestamp', original_timestamp)

        return output_df
