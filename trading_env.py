import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class TradingEnvironment:
    def __init__(self, cfg: Dict = None, expert_ensemble=None):
        self.cfg = cfg or {}
        self.expert_ensemble = expert_ensemble  
        self.trading_mode = self.cfg.get('trading_mode', 'spot')
        self.leverage = self.cfg.get('leverage', 10) if self.trading_mode == 'future' else 1
        
        self.fee_rate = self.cfg.get('fee_rate', 0.001) if self.trading_mode == 'spot' else self.cfg.get('fee_rate', 0.0004)
        self.slippage = self.cfg.get('slippage', 0.0005)
        self.initial_capital = self.cfg.get('initial_capital', 10000.0)
        self.max_risk_per_trade = self.cfg.get('max_risk_per_trade', 0.02)
        self.max_position_pct = self.cfg.get('max_position_pct', 0.5)
        self.drawdown_penalty = self.cfg.get('drawdown_penalty', 2.0)
        self.invalid_action_penalty = self.cfg.get('invalid_action_penalty', -0.1)
        
        self.use_trailing = self.cfg.get('use_trailing', True)
        self.trailing_activation_pct = self.cfg.get('trailing_activation_pct', 0.002)
        self.trailing_callback_pct = self.cfg.get('trailing_callback_pct', 0.001)
        self.stop_loss_pct = self.cfg.get('stop_loss_pct', 0.0015)
        self.take_profit_pct = self.cfg.get('take_profit_pct', 0.003)
        
        self.df = None
        self.features = None
        self.close_idx = 0
        self.window = self.cfg.get('window', 120)
        self.n_bars = 0
        
        self.current_idx = 0
        self.capital = self.initial_capital
        self.margin_locked = 0.0
        self.position = 0.0          
        self.entry_price = 0.0
        self.entry_idx = 0
        self.position_side = 'long'  
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.invalid_scaling_count = 0
        
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0

    def reset(self, df: pd.DataFrame = None, scaled_features: np.ndarray = None, close_idx: int = None) -> np.ndarray:
        if df is not None:
            self.df = df.reset_index(drop=True)
            self.features = scaled_features
            self.close_idx = close_idx or 0
            self.n_bars = len(df)
        
        self.current_idx = self.window
        self.capital = self.initial_capital
        self.margin_locked = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.position_side = 'long'
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.invalid_scaling_count = 0
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        
        return self._get_ppo_state()

    def _get_expert_signals(self) -> Dict[str, float]:
        idx = min(self.current_idx, self.n_bars - 1)
        if idx < 0:
            return {'direction_prob': 0.5, 'entry_quality': 0.5, 'expected_move': 0.003, 'exit_distance': 0.02}
        try:
            row = self.df.iloc[idx]
            if 'expert_direction_prob' in row:
                return {
                    'direction_prob': float(row['expert_direction_prob']),
                    'entry_quality': float(row['expert_entry_quality']),
                    'expected_move': float(row['expert_expected_move']),
                    'exit_distance': float(row['expert_exit_distance'])
                }
            else:
                if not hasattr(self, '_warned'):
                    print("Warning: Expert columns not found in DataFrame. Using default values. Ensure run_part2 injects them.")
                    self._warned = True
                return {'direction_prob': 0.5, 'entry_quality': 0.5, 'expected_move': 0.003, 'exit_distance': 0.02}
        except:
            return {'direction_prob': 0.5, 'entry_quality': 0.5, 'expected_move': 0.003, 'exit_distance': 0.02}

    def _get_regime_probs(self) -> np.ndarray:
        idx = min(self.current_idx, self.n_bars - 1)
        regime_cols = ['regime_p_0', 'regime_p_1', 'regime_p_2']
        if all(col in self.df.columns for col in regime_cols) and idx < len(self.df):
            return self.df[regime_cols].iloc[idx].values.astype(np.float32)
        return np.array([0.33, 0.33, 0.34], dtype=np.float32)

    def _get_l2_proxy(self) -> float:
        idx = min(self.current_idx, self.n_bars - 1)
        if 'l2_proxy' in self.df.columns and idx < len(self.df):
            return float(np.clip(self.df.iloc[idx]['l2_proxy'], -1, 1))
        return 0.0

    def _get_portfolio_status(self) -> Tuple[float, float, float]:
        if self.position == 0:
            position_status, pnl_pct = 0.0, 0.0
        else:
            position_status = 1.0 if self.position_side == 'long' else -1.0
            price = self._current_price()
            pnl_pct = (price - self.entry_price) / self.entry_price if self.position_side == 'long' else (self.entry_price - price) / self.entry_price
            if self.trading_mode == 'future':
                pnl_pct *= self.leverage
        
        val = self.get_portfolio_value()
        available_margin = (self.capital / val) if val > 0 else 1.0
        return position_status, float(np.clip(pnl_pct, -0.5, 0.5)), float(np.clip(available_margin, 0.0, 1.0))

    def _get_ppo_state(self) -> np.ndarray:
        expert = self._get_expert_signals()
        regime_probs = self._get_regime_probs()
        l2_val = self._get_l2_proxy()
        pos_status, pnl_pct, margin_ratio = self._get_portfolio_status()
        
        state = np.array([
            float(expert['direction_prob']),
            float(expert['entry_quality']),
            float(expert['expected_move']),
            float(expert['exit_distance']),
            l2_val,
            regime_probs[0],
            regime_probs[1],
            regime_probs[2],
            pos_status,
            pnl_pct,
            margin_ratio
        ], dtype=np.float32)
        return state

    def _calculate_reward(self, pnl_pct: float, trade_closed: bool = False) -> float:
        reward = 0.0
        if trade_closed and pnl_pct != 0:
            reward = pnl_pct * 10.0
            if pnl_pct > 0:
                reward += 0.05 * min(self.consecutive_wins, 5)
            else:
                reward -= 0.05 * min(self.consecutive_losses, 5)
        
        val = self.get_portfolio_value()
        if val > self.peak_capital:
            self.peak_capital = val
        drawdown = (self.peak_capital - val) / (self.peak_capital + 1e-10)
        if drawdown > 0.05:
            reward -= self.drawdown_penalty * drawdown
        if self.capital < 0:
            reward -= 0.5
        return float(np.clip(reward, -1.0, 1.0))

    def _current_price(self) -> float:
        return float(self.df['close'].iloc[min(self.current_idx, self.n_bars - 1)])
    
    def _current_high(self) -> float:
        return float(self.df['high'].iloc[min(self.current_idx, self.n_bars - 1)])
    
    def _current_low(self) -> float:
        return float(self.df['low'].iloc[min(self.current_idx, self.n_bars - 1)])

    def get_portfolio_value(self) -> float:
        price = self._current_price()
        if self.position == 0:
            return self.capital
        if self.trading_mode == 'spot':
            return self.capital + (self.position * price)
        unrealized = self.position * (price - self.entry_price) if self.position_side == 'long' else self.position * (self.entry_price - price)
        return self.capital + self.margin_locked + unrealized

    def _force_close_position(self, exit_price: float, log_type: str) -> float:
        orig_margin = self.margin_locked
        if self.position_side == 'long':
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price

        if self.trading_mode == 'future':
            realized = self.position * (exit_price - self.entry_price) if self.position_side == 'long' else self.position * (self.entry_price - exit_price)
            fee = (self.position * exit_price) * self.fee_rate
            self.capital += (self.margin_locked + realized - fee)
            trade_pnl = realized / (orig_margin + 1e-10)
        else:
            gross = self.position * exit_price
            fee = gross * self.fee_rate
            self.capital += (gross - fee)
            trade_pnl = pnl_pct

        self.trades.append({
            'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
            'entry_price': self.entry_price, 'exit_price': exit_price,
            'pnl_pct': trade_pnl, 'side': self.position_side,
            'bars_held': self.current_idx - self.entry_idx, 'trigger': log_type
        })
        
        if trade_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.position = 0.0
        self.entry_price = 0.0
        self.margin_locked = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        return trade_pnl

    def _execute_order(self, action: int) -> Tuple[bool, float, str]:
        price = self._current_price()
        if action == 0:
            return True, 0.0, 'hold'
        
        if action in [1, 2, 3]:
            size_pct, action_type = [0.25, 0.50, 1.0][action - 1], 'buy'
        elif action in [4, 5, 6]:
            size_pct, action_type = [0.25, 0.50, 1.0][action - 4], 'sell'
        else:
            action_type = 'close_all'

        if self.position == 0:
            if action_type == 'close_all':
                return True, 0.0, 'hold'
            
            trade_value = self.capital * size_pct * self.max_position_pct
            
            if action_type == 'buy':
                entry_price = price * (1 + self.slippage)
                size = trade_value / entry_price
                if self.trading_mode == 'future':
                    margin, fee = trade_value / self.leverage, trade_value * self.fee_rate
                    if margin + fee <= self.capital:
                        self.capital -= (margin + fee)
                        self.margin_locked = margin
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = 'long'
                        self.dynamic_sl = entry_price * (1 - self.stop_loss_pct)
                        self.dynamic_tp = entry_price * (1 + self.take_profit_pct)
                        self.trailing_peak = entry_price
                        return True, size, 'buy_long_open'
                else:
                    cost = trade_value * (1 + self.fee_rate)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = 'long'
                        self.dynamic_sl = entry_price * (1 - self.stop_loss_pct)
                        self.dynamic_tp = entry_price * (1 + self.take_profit_pct)
                        return True, size, 'spot_buy_open'
                        
            elif action_type == 'sell' and self.trading_mode == 'future':
                entry_price = price * (1 - self.slippage)
                size = trade_value / entry_price
                margin, fee = trade_value / self.leverage, trade_value * self.fee_rate
                if margin + fee <= self.capital:
                    self.capital -= (margin + fee)
                    self.margin_locked = margin
                    self.position = size
                    self.entry_price = entry_price
                    self.entry_idx = self.current_idx
                    self.position_side = 'short'
                    self.dynamic_sl = entry_price * (1 + self.stop_loss_pct)
                    self.dynamic_tp = entry_price * (1 - self.take_profit_pct)
                    self.trailing_peak = entry_price
                    return True, size, 'sell_short_open'
                    
            return False, 0.0, 'insufficient_margin'

        if self.position > 0 and self.position_side == 'long':
            if action_type == 'buy':
                return False, 0.0, 'invalid_scaling'
            if action_type == 'sell' or action_type == 'close_all':
                sell_pct = 1.0 if action_type == 'close_all' else size_pct
                close_size = self.position * sell_pct
                exit_price = price * (1 - self.slippage)
                orig_margin = self.margin_locked
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
                
                if self.trading_mode == 'future':
                    realized = close_size * (exit_price - self.entry_price)
                    fee = (close_size * exit_price) * self.fee_rate
                    self.capital += ((self.margin_locked * sell_pct) + realized - fee)
                    self.margin_locked -= (self.margin_locked * sell_pct)
                    self.position -= close_size
                    trade_pnl = realized / (orig_margin * sell_pct + 1e-10)
                else:
                    gross = close_size * exit_price
                    self.capital += (gross - (gross * self.fee_rate))
                    self.position -= close_size
                    trade_pnl = pnl_pct

                if self.position < 1e-10:
                    self.trades.append({
                        'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
                        'entry_price': self.entry_price, 'exit_price': exit_price,
                        'pnl_pct': trade_pnl, 'side': 'long',
                        'bars_held': self.current_idx - self.entry_idx, 'trigger': 'agent_action'
                    })
                    if trade_pnl > 0:
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
                    self.position, self.entry_price, self.margin_locked = 0.0, 0.0, 0.0
                return True, close_size, 'long_reduced_or_closed'

        if self.position > 0 and self.position_side == 'short':
            if action_type == 'sell':
                return False, 0.0, 'invalid_scaling'
            if action_type == 'buy' or action_type == 'close_all':
                buy_pct = 1.0 if action_type == 'close_all' else size_pct
                close_size = self.position * buy_pct
                exit_price = price * (1 + self.slippage)
                orig_margin = self.margin_locked
                
                if self.trading_mode == 'future':
                    realized = close_size * (self.entry_price - exit_price)
                    fee = (close_size * exit_price) * self.fee_rate
                    self.capital += ((self.margin_locked * buy_pct) + realized - fee)
                    self.margin_locked -= (self.margin_locked * buy_pct)
                    self.position -= close_size
                    trade_pnl = realized / (orig_margin * buy_pct + 1e-10)
                    
                    if self.position < 1e-10:
                        self.trades.append({
                            'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
                            'entry_price': self.entry_price, 'exit_price': exit_price,
                            'pnl_pct': trade_pnl, 'side': 'short',
                            'bars_held': self.current_idx - self.entry_idx, 'trigger': 'agent_action'
                        })
                        if trade_pnl > 0:
                            self.consecutive_wins += 1
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1
                            self.consecutive_wins = 0
                        self.position, self.entry_price, self.margin_locked = 0.0, 0.0, 0.0
                return True, close_size, 'short_reduced_or_closed'

        return False, 0.0, 'hold'

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        reward = 0.0
        reward_computed = False
        trade_closed_this_step = False
        step_pnl = 0.0

        if self.position > 0 and self.use_trailing:
            price = self._current_price()
            if self.position_side == 'long':
                if price > self.trailing_peak:
                    self.trailing_peak = price
                profit_pct = (price - self.entry_price) / self.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    self.trailing_activated = True
                if self.trailing_activated:
                    new_sl = self.trailing_peak * (1 - self.trailing_callback_pct)
                    if new_sl > self.dynamic_sl:
                        self.dynamic_sl = new_sl
            elif self.position_side == 'short':
                if price < self.trailing_peak or self.trailing_peak == 0.0:
                    self.trailing_peak = price
                profit_pct = (self.entry_price - price) / self.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    self.trailing_activated = True
                if self.trailing_activated:
                    new_sl = self.trailing_peak * (1 + self.trailing_callback_pct)
                    if new_sl < self.dynamic_sl or self.dynamic_sl == 0.0:
                        self.dynamic_sl = new_sl

        if self.position > 0:
            high, low = self._current_high(), self._current_low()
            stopped_out = False
            exit_price, log_msg = 0.0, ''
            
            if self.position_side == 'long':
                if low <= self.dynamic_sl:
                    stopped_out = True
                    exit_price = self.dynamic_sl * (1 - self.slippage)
                    log_msg = 'sl_triggered'
                elif high >= self.dynamic_tp:
                    stopped_out = True
                    exit_price = self.dynamic_tp * (1 + self.slippage)
                    log_msg = 'tp_triggered'
            elif self.position_side == 'short':
                if high >= self.dynamic_sl:
                    stopped_out = True
                    exit_price = self.dynamic_sl * (1 + self.slippage)
                    log_msg = 'sl_triggered'
                elif low <= self.dynamic_tp:
                    stopped_out = True
                    exit_price = self.dynamic_tp * (1 - self.slippage)
                    log_msg = 'tp_triggered'
                    
            if stopped_out:
                step_pnl = self._force_close_position(exit_price, log_msg)
                reward = self._calculate_reward(step_pnl, trade_closed=True)
                reward_computed = True
                trade_closed_this_step = True
                action = 0

        if not trade_closed_this_step:
            success, _, order_status = self._execute_order(action)
            if order_status == 'invalid_scaling':
                reward = self.invalid_action_penalty
                reward_computed = True
                self.invalid_scaling_count += 1
            elif order_status in ['long_reduced_or_closed', 'short_reduced_or_closed'] and self.position == 0:
                trade_closed_this_step = True
                if self.trades:
                    step_pnl = self.trades[-1]['pnl_pct']

        if self.current_idx >= self.n_bars - 1:
            self.done = True
            if self.position > 0:
                price = self._current_price()
                exit_price = price * (1 - self.slippage) if self.position_side == 'long' else price * (1 + self.slippage)
                step_pnl = self._force_close_position(exit_price, 'episode_end')
                reward = self._calculate_reward(step_pnl, trade_closed=True)
                reward_computed = True

        if not reward_computed:
            if trade_closed_this_step:
                reward = self._calculate_reward(step_pnl, trade_closed=True)
            else:
                reward = self._calculate_reward(0.0, trade_closed=False)

        self.current_idx += 1
        next_state = self._get_ppo_state()
        
        return next_state, float(reward), self.done

    def get_trade_statistics(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'profit_factor': 0.0,
                'total_return': 0.0, 'invalid_scaling_attempts': self.invalid_scaling_count
            }
        wins = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(np.abs(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(np.abs(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        return {
            'total_trades': total_trades, 'win_rate': win_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital,
            'invalid_scaling_attempts': self.invalid_scaling_count
        }

    def set_expert_ensemble(self, expert_ensemble) -> None:
        self.expert_ensemble = expert_ensemble
