import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from datetime import datetime, timezone

def safe_divide(numerator: Union[np.ndarray, float, int], denominator: Union[np.ndarray, float, int], fill: float = 0.0) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-12, fill, numerator / denominator)
    return result.astype(np.float32)

def safe_log(x: Union[np.ndarray, float, int], eps: float = 1e-10) -> np.ndarray:
    return np.log(np.maximum(x, eps))

def clip_outliers(series: pd.Series, n_std: float = 5.0) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std < 1e-10:
        return series
    return series.clip(mean - n_std * std, mean + n_std * std)

def correct_sharpe(returns: np.ndarray, ann_factor: float = 8760) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return 0.0
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    if std_r < 1e-10:
        return 0.0
    return float((mean_r / std_r) * np.sqrt(ann_factor))

def sortino_ratio(returns: np.ndarray, ann_factor: float = 8760) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return 0.0
    mean_r = np.mean(r)
    downside = r[r < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    return float((mean_r / down_std) * np.sqrt(ann_factor))

def max_drawdown(port_values: np.ndarray) -> float:
    pv = np.array(port_values, dtype=np.float64)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / (peak + 1e-10)
    return float(dd.min())

def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 10:
        return -0.02
    return float(np.percentile(r, (1 - confidence) * 100))

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 10:
        return -0.03
    var = np.percentile(r, (1 - confidence) * 100)
    cvar = r[r <= var].mean()
    return float(cvar)

def calculate_calmar(annual_return: float, max_drawdown: float) -> float:
    if abs(max_drawdown) < 1e-10:
        return annual_return / 0.01
    return annual_return / abs(max_drawdown)

def rolling_sharpe(returns: pd.Series, window: int = 30, ann_factor: float = 8760) -> pd.Series:
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(ann_factor)
    return sharpe.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray, ann_factor: float = 8760) -> float:
    active_returns = returns - benchmark_returns
    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns, ddof=1)
    if std_active < 1e-10:
        return 0.0
    return float((mean_active / std_active) * np.sqrt(ann_factor))

def gain_to_pain_ratio(returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return 0.0
    total_gain = np.sum(r[r > 0])
    total_pain = np.sum(np.abs(r[r < 0]))
    if total_pain < 1e-10:
        return total_gain / 0.01
    return float(total_gain / total_pain)

def ulcer_index(port_values: np.ndarray) -> float:
    pv = np.array(port_values, dtype=np.float64)
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / (peak + 1e-10)
    squared_dd = drawdown ** 2
    ulcer = np.sqrt(np.mean(squared_dd))
    return float(ulcer)

def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    m = market_returns[~np.isnan(market_returns)]
    min_len = min(len(r), len(m))
    if min_len < 5:
        return 1.0
    r = r[:min_len]
    m = m[:min_len]
    covariance = np.cov(r, m)[0, 1]
    variance = np.var(m)
    if variance < 1e-10:
        return 1.0
    return float(covariance / variance)

def calculate_alpha(returns: np.ndarray, market_returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    beta = calculate_beta(returns, market_returns)
    mean_return = np.mean(returns)
    mean_market = np.mean(market_returns)
    alpha = mean_return - (risk_free_rate + beta * (mean_market - risk_free_rate))
    return float(alpha)

def calculate_treynor(returns: np.ndarray, market_returns: np.ndarray, ann_factor: float = 8760) -> float:
    beta = calculate_beta(returns, market_returns)
    if abs(beta) < 1e-10:
        return 0.0
    mean_return = np.mean(returns) * np.sqrt(ann_factor)
    return float(mean_return / beta)

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return 1.0
    gains = r[r > threshold]
    losses = r[r <= threshold]
    total_gain = np.sum(gains - threshold)
    total_loss = np.sum(threshold - losses)
    if total_loss < 1e-10:
        return float(total_gain / 0.01)
    return float(total_gain / total_loss)

def calculate_stability(returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 10:
        return 0.5
    cumsum = np.cumsum(r)
    x = np.arange(len(cumsum))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumsum)
    return float(r_value ** 2)

def calculate_tail_ratio(returns: np.ndarray, percentile: float = 5) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 10:
        return 1.0
    right_tail = np.percentile(r, 100 - percentile)
    left_tail = np.percentile(r, percentile)
    if abs(left_tail) < 1e-10:
        return right_tail / 0.01
    return float(right_tail / abs(left_tail))

def calculate_outlier_ratio(returns: np.ndarray, n_std: float = 3.0) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 10:
        return 0.0
    mean = np.mean(r)
    std = np.std(r)
    outliers = np.sum(np.abs(r - mean) > n_std * std)
    return float(outliers / len(r))

def calculate_historical_var(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    r = returns[~np.isnan(returns)]
    if len(r) < 50:
        return -0.02, -0.03
    var = np.percentile(r, (1 - confidence) * 100)
    cvar = r[r <= var].mean()
    return float(var), float(cvar)

def calculate_max_adverse_excursion(trades: List[Dict], lookback: int = 20) -> float:
    if not trades:
        return -0.05
    mae_values = []
    for trade in trades:
        if trade.get('type') == 'sell' and trade.get('entry_price'):
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('price', 0)
            if entry > 0:
                mae = (exit_price - entry) / entry
                mae_values.append(mae)
    if not mae_values:
        return -0.05
    return float(np.percentile(mae_values, 10))

def calculate_max_favorable_excursion(trades: List[Dict], lookback: int = 20) -> float:
    if not trades:
        return 0.08
    mfe_values = []
    for trade in trades:
        if trade.get('type') == 'sell' and trade.get('entry_price'):
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('price', 0)
            if entry > 0:
                mfe = (exit_price - entry) / entry
                if mfe > 0:
                    mfe_values.append(mfe)
    if not mfe_values:
        return 0.08
    return float(np.percentile(mfe_values, 90))

def calculate_profit_factor(trades: List[Dict]) -> float:
    gross_profit = 0.0
    gross_loss = 0.0
    for trade in trades:
        if trade.get('type') == 'sell' and trade.get('entry_price'):
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('price', 0)
            if entry > 0:
                pnl = (exit_price - entry) / entry
                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
    if gross_loss < 1e-10:
        return gross_profit / 0.01
    return float(gross_profit / gross_loss)

def calculate_expectancy(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    total_pnl = 0.0
    for trade in trades:
        if trade.get('type') == 'sell' and trade.get('entry_price'):
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('price', 0)
            if entry > 0:
                pnl = (exit_price - entry) / entry
                total_pnl += pnl
    return float(total_pnl / len(trades))

def calculate_risk_of_ruin(win_rate: float, risk_reward_ratio: float, capital_fraction: float) -> float:
    if win_rate >= 0.5:
        return 0.0
    p = win_rate
    q = 1 - p
    b = risk_reward_ratio
    if p * b <= q:
        return 1.0
    a = capital_fraction
    numerator = ((q / p) ** (1 / a)) - 1
    denominator = ((q / p) ** (1 / (a * b))) - 1
    if denominator < 1e-10:
        return 1.0
    risk = numerator / denominator
    return float(max(0.0, min(1.0, risk)))

def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss < 1e-10:
        return win_rate
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (avg_win + 1e-10)
    return float(max(0.0, min(0.25, kelly)))