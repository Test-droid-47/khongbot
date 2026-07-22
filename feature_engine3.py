import numpy as np
import pandas as pd
from typing import Dict
from scipy.stats import linregress
from scipy.signal import cwt, ricker

class FeatureEngine:

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}

    def _amihud_illiq(self, df, window=20):
        ret = df['close'].pct_change().abs()
        dollar_vol = df['close'] * df['volume']
        illiq = ret / (dollar_vol + 1e-10)
        return illiq.rolling(window, min_periods=1).mean().fillna(0).values.astype(np.float32)

    def _vpin(self, df, vol_window=50):
        price_chg = df['close'].diff().values
        volume = df['volume'].values
        buy_vol = np.where(price_chg > 0, volume, 0)
        sell_vol = np.where(price_chg < 0, volume, 0)
        b_vol_series = pd.Series(buy_vol).rolling(vol_window, min_periods=1).sum()
        s_vol_series = pd.Series(sell_vol).rolling(vol_window, min_periods=1).sum()
        total = b_vol_series + s_vol_series + 1e-10
        vpin = np.abs(b_vol_series - s_vol_series) / total
        return vpin.fillna(0).values.astype(np.float32)

    def _roll_spread(self, df, window=10):
        delta_p = df['close'].diff().values
        cov_series = pd.Series(delta_p).rolling(window, min_periods=1).cov(pd.Series(delta_p).shift(1))
        spread = np.sqrt(np.maximum(0, -cov_series.values)) * 2.0
        return np.nan_to_num(spread).astype(np.float32)

    def _ou_half_life(self, df, window=30):
        prices = df['close'].values
        n = len(prices)
        if n < window + 2:
            raise RuntimeError(f"OU Half-Life: Need at least {window+2} rows, got {n}")
        half_life = np.full(n, 10.0, dtype=np.float32)
        for i in range(window, n):
            y = prices[i-window+1:i+1]
            x = prices[i-window:i]
            if len(y) != window or len(x) != window:
                continue
            slope, intercept, r_value, p_value, stderr = linregress(x, y)
            if slope <= 0:
                continue
            lambda_ = -np.log(slope)
            if lambda_ > 1e-6:
                hl = np.log(2) / lambda_
                half_life[i] = min(hl, 100.0)
        return half_life

    def _wavelet_energy_ratio(self, df, window=60):
        prices = df['close'].values
        n = len(prices)
        if n < window:
            raise RuntimeError(f"Wavelet Energy: Need at least {window} rows, got {n}")
        ratio = np.full(n, 0.5, dtype=np.float32)
        widths = np.arange(1, 8)
        for i in range(window, n):
            seg = prices[i-window:i]
            seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)
            cwt_mat = cwt(seg_norm, ricker, widths)
            total_e = np.sum(cwt_mat**2)
            if total_e > 1e-10:
                hf_e = np.sum(cwt_mat[:3]**2)
                ratio[i] = hf_e / total_e
        return pd.Series(ratio).fillna(0.5).values.astype(np.float32)

    def _higuchi_fd(self, df, kmax=5):
        prices = df['close'].values
        n = len(prices)
        if n < 60:
            raise RuntimeError(f"Higuchi FD: Need at least 60 rows, got {n}")
        fd = np.full(n, 1.5, dtype=np.float32)
        L = 60
        for i in range(L, n):
            seg = prices[i-L:i]
            N = len(seg)
            Lk = []
            for k in range(1, min(kmax, N-1)):
                sum_abs = 0.0
                count = 0
                for m in range(k):
                    idx = np.arange(m, N, k)
                    if len(idx) < 2:
                        continue
                    sum_abs += np.sum(np.abs(seg[idx[1:]] - seg[idx[:-1]]))
                    count += len(idx) - 1
                if count > 0:
                    Lk.append(np.log(sum_abs / count))
            if len(Lk) > 2:
                x_vals = np.log(1.0 / np.arange(1, len(Lk)+1))
                slope, intercept, r_value, p_value, stderr = linregress(x_vals, np.array(Lk))
                fd[i] = max(1.0, min(2.0, slope))
        return pd.Series(fd).fillna(1.5).values.astype(np.float32)

    def _hurst_approx(self, df, window=60):
        log_prices = np.log(df['close'].values + 1e-10)
        n = len(log_prices)
        if n < window:
            raise RuntimeError(f"Hurst: Need at least {window} rows, got {n}")
        hurst = np.full(n, 0.5, dtype=np.float32)
        for i in range(window, n):
            seg = log_prices[i-window:i]
            r = np.max(seg) - np.min(seg)
            s = np.std(seg)
            if s > 1e-10:
                hurst[i] = np.log(r / s) / np.log(window)
        return np.clip(hurst, 0.0, 1.0)

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[FeatureEngine] Building 17 dual-set features...")
        df = df.copy()

        tr = np.maximum(df['high'] - df['low'],
                        np.maximum((df['high'] - df['close'].shift()).abs(),
                                   (df['low'] - df['close'].shift()).abs()))
        atr = tr.rolling(14, min_periods=1).mean().fillna(0).values

        print("  [1/6] Amihud Illiquidity...", end=" ", flush=True)
        df['amihud_illiq'] = self._amihud_illiq(df)
        print("DONE")

        print("  [2/6] VPIN (Flow Toxicity)...", end=" ", flush=True)
        df['vpin'] = self._vpin(df)
        print("DONE")

        print("  [3/6] Roll Implied Spread...", end=" ", flush=True)
        df['roll_spread'] = self._roll_spread(df)
        print("DONE")

        print("  [4/6] OU Half-Life (Mean Reversion)...", end=" ", flush=True)
        df['ou_half_life'] = self._ou_half_life(df)
        print("DONE")

        print("  [5/6] Wavelet High-Freq Energy Ratio...", end=" ", flush=True)
        df['wavelet_hf_ratio'] = self._wavelet_energy_ratio(df)
        print("DONE")

        print("  [6/6] Higuchi Fractal Dimension...", end=" ", flush=True)
        df['higuchi_fd'] = self._higuchi_fd(df)
        print("DONE")

        print("  [7/6] Hurst Exponent (Proxy)...", end=" ", flush=True)
        df['hurst_exp'] = self._hurst_approx(df)
        print("DONE")

        direction = (df['close'] - df['close'].shift(20)).abs()
        volatility = df['close'].diff().abs().rolling(20, min_periods=1).sum()
        df['efficiency_ratio_20'] = (direction / (volatility + 1e-10)).fillna(0).astype(np.float32)

        df['natr'] = (atr / (df['close'] + 1e-10)).astype(np.float32)

        high_rej = ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)).fillna(0)
        low_rej = ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)).fillna(0)
        df['rejection_high'] = high_rej.astype(np.float32)
        df['rejection_low'] = low_rej.astype(np.float32)

        roll_h = df['high'].rolling(10, min_periods=1).max()
        roll_l = df['low'].rolling(10, min_periods=1).min()
        df['range_percentile'] = ((df['close'] - roll_l) / (roll_h - roll_l + 1e-10)).fillna(0.5).astype(np.float32)

        df['price_accel'] = ((df['close'] - 2.0 * df['close'].shift(1) + df['close'].shift(2)) / (atr + 1e-10)).fillna(0).astype(np.float32)

        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        df['vol_aggression'] = (((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)) * (df['volume'] / (vol_ma + 1e-10))).fillna(0).astype(np.float32)

        buy_dist = ((df['close'] - df['low'].rolling(5, min_periods=1).min()) / (df['close'] + 1e-10)).fillna(0)
        sell_dist = ((df['high'].rolling(5, min_periods=1).max() - df['close']) / (df['close'] + 1e-10)).fillna(0)
        df['stop_buy_dist'] = buy_dist.astype(np.float32)
        df['stop_sell_dist'] = sell_dist.astype(np.float32)

        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        vwap = (df['volume'] * typical_price).rolling(20, min_periods=1).sum() / (df['volume'].rolling(20, min_periods=1).sum() + 1e-10)
        ema_5 = df['close'].ewm(span=5, adjust=False).mean()
        df['vwap_ema_spread'] = ((ema_5 - vwap) / (atr + 1e-10)).fillna(0).astype(np.float32)

        # ---------- ORIGINAL VARIABLE NAME PRESERVED ----------
        elite_features = [
            'amihud_illiq', 'vpin', 'roll_spread', 'ou_half_life', 'wavelet_hf_ratio', 'higuchi_fd',
            'hurst_exp', 'efficiency_ratio_20', 'natr', 'rejection_high', 'rejection_low',
            'range_percentile', 'price_accel', 'vol_aggression', 'stop_buy_dist', 'stop_sell_dist',
            'vwap_ema_spread'
        ]

        output_df = df[elite_features].copy()
        output_df = output_df.ffill().bfill().fillna(0.0)

        # Safety checks
        if output_df.isna().any().any():
            nan_cols = output_df.columns[output_df.isna().any()].tolist()
            raise RuntimeError(f"NaN found in columns: {nan_cols}")

        zero_cols = []
        for col in elite_features:
            if (output_df[col] == 0).all():
                zero_cols.append(col)
        if zero_cols:
            print(f"  [WARNING] All-zero columns (feature may have failed): {zero_cols}")

        if 'timestamp' in df.columns:
            output_df.insert(0, 'timestamp', df['timestamp'].values)

        print(f"[FeatureEngine] Success. Shape: {output_df.shape[0]} rows, {len(elite_features)} features.\n")
        return output_df