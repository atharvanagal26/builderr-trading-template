import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class InstitutionalAVSEEngine:
    """
    Adaptive Volatility Squeeze & Efficiency Parity Engine (AVSE)
    Customized for active paper trading challenges requiring high trade turnover,
    strict risk parity, and rapid exit execution.
    """

    @staticmethod
    def calculate_efficiency_ratio(closes: np.ndarray, period: int = 14) -> float:
        """Calculates Kaufman Efficiency Ratio (KER) to score directional momentum vs noise."""
        if len(closes) < period + 1:
            return 0.50
        net_change = abs(closes[-1] - closes[-period - 1])
        sum_changes = np.sum(np.abs(np.diff(closes[-period - 1:])))
        if sum_changes == 0:
            return 0.0
        return float(np.clip(net_change / sum_changes, 0.0, 1.0))

    @classmethod
    def evaluate_asset(cls, df: pd.DataFrame) -> dict:
        metrics = {
            "signal": "HOLD",
            "alpha_score": 0.0,
            "atr_pct": 0.015,
            "price": 0.0,
            "squeeze_active": False,
        }
        try:
            if df is None or len(df) < 25:
                return metrics

            # Normalize DataFrame column names
            col_map = {str(c).lower().strip(): c for c in df.columns}
            close_key = col_map.get("close", col_map.get("price", df.columns[-1]))
            high_key = col_map.get("high", close_key)
            low_key = col_map.get("low", close_key)
            vol_key = col_map.get("volume", col_map.get("vol", None))

            closes = df[close_key].to_numpy(dtype=float)
            highs = df[high_key].to_numpy(dtype=float)
            lows = df[low_key].to_numpy(dtype=float)

            volumes = df[vol_key].to_numpy(dtype=float) if vol_key and vol_key in df.columns else np.ones_like(closes)

            current_price = float(closes[-1])
            if current_price <= 0 or np.isnan(current_price):
                return metrics
            metrics["price"] = current_price

            prices_series = pd.Series(closes)

            # 1. Kaufman Efficiency Ratio
            ker = cls.calculate_efficiency_ratio(closes, period=14)

            # 2. Average True Range (ATR)
            hl = highs - lows
            prev_close = np.roll(closes, 1)
            prev_close[0] = closes[0]
            hc = np.abs(highs - prev_close)
            lc = np.abs(lows - prev_close)
            tr = np.maximum(hl, np.maximum(hc, lc))
            atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().to_numpy()[-1]
            if np.isnan(atr) or atr <= 0:
                atr = current_price * 0.015
            metrics["atr_pct"] = float(atr / current_price)

            # 3. TTM Squeeze Detection
            sma_20 = prices_series.rolling(20, min_periods=1).mean().to_numpy()[-1]
            std_20 = prices_series.rolling(20, min_periods=1).std().to_numpy()[-1]

            bb_upper = sma_20 + (2.0 * std_20)
            bb_lower = sma_20 - (2.0 * std_20)
            kc_upper = sma_20 + (1.5 * atr)
            kc_lower = sma_20 - (1.5 * atr)

            squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)
            metrics["squeeze_active"] = squeeze

            # 4. Relative Strength Index (RSI - 10 period)
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=10, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=10, min_periods=1).mean()
            rs = gain / (loss + 1e-6)
            rsi = float((100 - (100 / (1 + rs))).to_numpy()[-1])

            # 5. Volume Shock & Momentum
            vol_ma = pd.Series(volumes).rolling(14, min_periods=1).mean().to_numpy()[-1]
            vol_ratio = float(volumes[-1] / (vol_ma + 1e-6))
            roc_3 = (closes[-1] - closes[-4]) / (closes[-4] + 1e-6) if len(closes) >= 4 else 0.0

            # --- REGIME-BASED SIGNAL SELECTION ---
            if ker >= 0.40:  # Directional Trend / Micro-Breakout Regime
                is_breakout = (current_price > bb_upper) or (roc_3 > 0.012 and vol_ratio > 1.1)
                if is_breakout and rsi < 72:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(np.clip((ker * 40.0) + (roc_3 * 2000.0) + (vol_ratio * 10.0), 1.0, 100.0))
                elif current_price < sma_20 or rsi > 75:
                    metrics["signal"] = "SELL"
            else:  # Mean Reversion / Squeeze Release Regime
                if current_price <= (sma_20 - (1.2 * atr)) or rsi < 36:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(np.clip((40.0 - rsi) * 2.0 + (vol_ratio * 5.0), 1.0, 100.0))
                elif current_price >= sma_20 or rsi > 62:
                    metrics["signal"] = "SELL"

            return metrics
        except Exception as e:
            print(f"Error evaluating asset: {e}")
            return metrics


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list:
    """
    Main execution loop invoked by the Builderr paper trading platform.
    """
    try:
        orders = []
        if not market_state:
            return orders

        total_portfolio_value = float(cash)
        current_gross_exposure = 0.0
        current_positions = {}
        active_prices = {}

        # 1. Parse current portfolio state
        for ticker, p_val in portfolio_state.items():
            qty_held = p_val.get('quantity', p_val.get('qty', 0)) if isinstance(p_val, dict) else p_val
            if qty_held and float(qty_held) > 0:
                current_positions[ticker] = float(qty_held)

        # 2. Extract latest market prices
        for ticker, bars in market_state.items():
            if bars:
                try:
                    last_bar = bars[-1]
                    price = last_bar.get("close", last_bar.get("price", last_bar.get("open", 0)))
                    if price and float(price) > 0:
                        active_prices[ticker] = float(price)
                except Exception:
                    continue

        for ticker, qty in current_positions.items():
            if ticker in active_prices:
                pos_val = qty * active_prices[ticker]
                total_portfolio_value += pos_val
                current_gross_exposure += pos_val

        available_cash = float(cash)
        buy_candidates = []

        # 3. Process Sell Exits and Trailing Stops
        for ticker, bars in market_state.items():
            if not bars or len(bars) < 15:
                continue

            try:
                df = pd.DataFrame(bars)
                analysis = InstitutionalAVSEEngine.evaluate_asset(df)
                current_price = active_prices.get(ticker, analysis["price"])

                if current_price <= 0:
                    continue

                qty_held = current_positions.get(ticker, 0.0)

                if qty_held > 0:
                    col_map = {str(c).lower().strip(): c for c in df.columns}
                    close_key = col_map.get('close', col_map.get('price', df.columns[-1]))
                    closes_lookback = df[close_key].to_numpy(dtype=float)

                    recent_peak = float(np.max(closes_lookback[-10:]))
                    stop_price = recent_peak * (1.0 - (1.8 * analysis["atr_pct"]))

                    if current_price < stop_price or analysis["signal"] == "SELL":
                        orders.append({
                            "ticker": str(ticker),
                            "side": "sell",
                            "quantity": int(qty_held)
                        })
                        current_gross_exposure -= (qty_held * current_price)
                        available_cash += (qty_held * current_price)
                        current_positions[ticker] = 0.0
                        continue

                if analysis["signal"] == "BUY":
                    buy_candidates.append({
                        "ticker": str(ticker),
                        "price": float(current_price),
                        "score": float(analysis["alpha_score"]),
                        "atr_pct": float(analysis["atr_pct"]),
                        "qty_held": float(qty_held)
                    })
            except Exception as e:
                print(f"Error processing exits for {ticker}: {e}")
                continue

        # 4. Capital Allocation (Volatility Risk Parity)
        max_leverage_limit = 1.38
        max_allowed_exposure = total_portfolio_value * max_leverage_limit
        buy_candidates = sorted(buy_candidates, key=lambda x: x["score"], reverse=True)

        for candidate in buy_candidates:
            remaining_leverage = max_allowed_exposure - current_gross_exposure
            if remaining_leverage <= 0 or available_cash <= 10:
                break

            ticker = candidate["ticker"]
            price = candidate["price"]
            atr_pct = candidate["atr_pct"]
            qty_held = candidate["qty_held"]

            target_pct = min(0.28, max(0.12, 0.025 / (atr_pct + 1e-4)))
            target_alloc = total_portfolio_value * target_pct
            needed_spend = target_alloc - (qty_held * price)

            if needed_spend <= 0:
                continue

            spend_amount = min(needed_spend, available_cash * 0.90, remaining_leverage)

            # Lowered threshold from 0.04 to 0.005 to permit smaller valid allocations
            if spend_amount > (total_portfolio_value * 0.005):
                quantity = int(spend_amount / price)
                if quantity > 0:
                    orders.append({
                        "ticker": str(ticker),
                        "side": "buy",
                        "quantity": int(quantity)
                    })
                    available_cash -= (quantity * price)
                    current_gross_exposure += (quantity * price)

        return orders

    except Exception as e:
        print(f"Fatal error in decide loop: {e}")
        return []

# --- LOCAL TEST RUNNER ---
if __name__ == "__main__":
    # Mock data to test your bot locally
    sample_market_state = {
        "AAPL": [
            {"open": 150.0, "high": 152.0, "low": 149.0, "close": 151.5, "volume": 10000}
            for _ in range(30)
        ]
    }
    sample_portfolio_state = {}
    sample_cash = 10000.0

    print("Running local test simulation...")
    orders = decide(sample_market_state, sample_portfolio_state, sample_cash)
    print("Generated Orders:", orders)
