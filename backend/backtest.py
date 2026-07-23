"""
FinBrain 回测引擎 v2 — 信号提取 + OHLC 模拟 + 风险指标 + 基准对比 + 框架评估。
纯代码，不调LLM。
"""

import re, json, math
from datetime import datetime, timedelta


# ============================================================
#  信号提取
# ============================================================

def extract_signal(report: str) -> dict | None:
    """从报告中提取交易信号：建仓价、止损价、目标价、持仓周期、仓位比例。"""
    signal = {}
    # 建仓价
    m = re.search(r'[≤<=]\s*([\d.]+)\s*元.*建仓', report)
    if m: signal["entry_price"] = float(m.group(1))
    # 止损价
    m = re.search(r'止损\s*([\d.]+)\s*元', report)
    if m: signal["stop_loss"] = float(m.group(1))
    # 目标价（区间）
    m = re.search(r'目标\s*([\d.]+)\s*[-~至]\s*([\d.]+)\s*元', report)
    if m: signal["target_low"] = float(m.group(1)); signal["target_high"] = float(m.group(2))
    # 仓位比例
    m = re.search(r'仓位\s*(\d+)\s*%', report)
    if m: signal["position_pct"] = int(m.group(1)) / 100.0
    # 持仓周期
    m = re.search(r'持仓周期[：:]\s*(\d+)[-~至]?\d*\s*个?月', report)
    if m: signal["holding_months"] = int(m.group(1))
    # 股票代码
    m = re.search(r'(\d{6})', report)
    if m: signal["symbol"] = m.group(1)
    # 评级 + 加权总分
    m = re.search(r'(BUY|HOLD|SELL)', report)
    if m: signal["rating"] = m.group(1)
    m = re.search(r'加权总分:\s*([\d.]+)', report)
    if m: signal["weighted_score"] = float(m.group(1))

    if "entry_price" not in signal:
        return None
    signal.setdefault("holding_months", 6)
    signal.setdefault("stop_loss", signal["entry_price"] * 0.85)
    signal.setdefault("position_pct", 0.10)
    return signal


# ============================================================
#  基准数据
# ============================================================

def _get_benchmark_returns(symbol: str, bars: list, bench_symbol: str = "sh000300") -> list:
    """获取沪深300同期日收益率序列。失败时返回空列表。"""
    try:
        from backend.tools import fetch_stock_history
        first_day = bars[0].get("day", "") if bars else ""
        last_day = bars[-1].get("day", "") if bars else ""
        days = (datetime.strptime(last_day, "%Y-%m-%d") - datetime.strptime(first_day, "%Y-%m-%d")).days + 1
        bench_hist = fetch_stock_history(bench_symbol, scale=240, datalen=max(days, 60))
        if "error" in bench_hist:
            return []
        bench_bars = bench_hist.get("data", [])
        # 对齐日期，计算日收益率
        bench_map = {b["day"]: float(b["close"]) for b in bench_bars if b.get("day")}
        returns = []
        prev = None
        for bar in bars:
            day = bar.get("day", "")
            close = bench_map.get(day)
            if close and prev:
                returns.append((close - prev) / prev)
            prev = close
        return returns
    except Exception:
        return []


# ============================================================
#  风险指标计算
# ============================================================

def _calc_metrics(daily_returns: list, benchmark_returns: list = None,
                  risk_free_rate: float = 0.025) -> dict:
    """从日收益率序列计算风险指标。"""
    if not daily_returns or len(daily_returns) < 5:
        return {}

    n = len(daily_returns)
    total_return = math.prod(1 + r for r in daily_returns) - 1

    # 年化收益率（假设252个交易日）
    ann_return = (1 + total_return) ** (252 / n) - 1 if n > 0 else 0

    # 年化波动率
    mean_ret = sum(daily_returns) / n
    variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (n - 1) if n > 1 else 0
    ann_vol = math.sqrt(variance) * math.sqrt(252)

    # 夏普比率
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    dd_start = dd_end = 0
    for i, r in enumerate(daily_returns):
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
            dd_end = i

    # Calmar 比率（年化收益/最大回撤）
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # 胜率（按日）
    win_days = sum(1 for r in daily_returns if r > 0)
    daily_win_rate = win_days / n if n > 0 else 0

    # 盈亏比（日均盈利/日均亏损）
    gains = [r for r in daily_returns if r > 0]
    losses = [abs(r) for r in daily_returns if r < 0]
    profit_factor = (sum(gains) / sum(losses)) if losses and sum(losses) > 0 else 0

    metrics = {
        "累计收益": f"{total_return:+.1%}",
        "年化收益": f"{ann_return:+.1%}",
        "年化波动率": f"{ann_vol:.1%}",
        "夏普比率": round(sharpe, 2),
        "最大回撤": f"{max_dd:.1%}",
        "Calmar比率": round(calmar, 2),
        "日胜率": f"{daily_win_rate:.1%}",
        "盈亏比": round(profit_factor, 2),
        "交易天数": n,
    }

    # 基准对比
    if benchmark_returns and len(benchmark_returns) >= 5:
        bench_total = math.prod(1 + r for r in benchmark_returns) - 1
        bench_ann = (1 + bench_total) ** (252 / len(benchmark_returns)) - 1
        alpha = ann_return - bench_ann
        # 超额波动率
        excess_returns = [daily_returns[i] - benchmark_returns[i]
                         for i in range(min(len(daily_returns), len(benchmark_returns)))]
        if len(excess_returns) > 1:
            ex_mean = sum(excess_returns) / len(excess_returns)
            ex_var = sum((r - ex_mean) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
            tracking_error = math.sqrt(ex_var) * math.sqrt(252)
            info_ratio = alpha / tracking_error if tracking_error > 0 else 0
        else:
            tracking_error = 0
            info_ratio = 0
        metrics.update({
            "基准收益": f"{bench_ann:+.1%}",
            "超额收益(Alpha)": f"{alpha:+.1%}",
            "信息比率": round(info_ratio, 2),
        })

    return metrics


# ============================================================
#  核心回测
# ============================================================

# A股交易成本：印花税0.05%(卖出单边) + 佣金0.025%(双边) + 过户费0.001%(双边)
SLIPPAGE = 0.001   # 滑点 0.1%
STAMP_TAX = 0.0005  # 印花税（卖出）
COMMISSION = 0.00025  # 佣金（双边）
TRANSFER_FEE = 0.00001  # 过户费


def run_backtest(signal: dict, lookback_days: int = 365,
                 use_benchmark: bool = True) -> dict:
    """
    用历史K线回测单个交易信号。返回完整风险指标 + 权益曲线。

    Returns:
        {triggered, entry_date, entry_price, exit_date, exit_price, exit_reason,
         return_pct, holding_days, max_drawdown, sharpe, equity_curve, ...}
    """
    from backend.tools import fetch_stock_history

    symbol = signal.get("symbol", "")
    if not symbol:
        return {"error": "缺少股票代码"}

    entry_price = signal["entry_price"]
    stop_loss = signal.get("stop_loss", entry_price * 0.85)
    target_low = signal.get("target_low", entry_price * 1.2)
    target_high = signal.get("target_high", target_low * 1.1)
    position_pct = signal.get("position_pct", 0.10)
    holding_days = signal.get("holding_months", 6) * 22

    # 获取历史K线（日线）
    hist = fetch_stock_history(symbol, scale=240, datalen=max(lookback_days, 60))
    if "error" in hist:
        return {"error": hist["error"]}
    bars = hist.get("data", [])
    if len(bars) < 5:
        return {"error": "K线数据不足"}

    # 状态机
    result = {
        "triggered": False, "entry_date": "", "entry_price": 0,
        "exit_date": "", "exit_price": 0, "exit_reason": "未触发",
        "return_pct": 0, "holding_days": 0,
        "equity_curve": [], "daily_returns": [],
        "metrics": {}, "cost_pct": 0,
    }

    in_position = False
    entry_idx = -1
    daily_equity = []  # (day, equity_scaled)

    for i, bar in enumerate(bars):
        close = float(bar.get("close", 0))
        low = float(bar.get("low", close))
        high = float(bar.get("high", close))
        day = bar.get("day", "")

        if not in_position:
            if low <= entry_price and close > 0:
                in_position = True
                entry_idx = i
                result["triggered"] = True
                result["entry_date"] = day
                result["entry_price"] = entry_price
                # 入场成本
                entry_cost = entry_price * (1 + SLIPPAGE + COMMISSION + TRANSFER_FEE)
                daily_equity.append((day, 1.0))
        elif in_position:
            days_held = i - entry_idx
            exit_price = 0
            exit_reason = ""

            # 止损检查（以收盘价判断，避免盘中假突破）
            if close <= stop_loss:
                exit_price = stop_loss
                exit_reason = "止损"
            # 止盈检查
            elif high >= target_low:
                exit_price = target_low
                exit_reason = "止盈"
            # 超时
            elif days_held >= holding_days:
                exit_price = close
                exit_reason = "到期"

            if exit_price > 0:
                # 出场成本（卖出有印花税）
                exit_cost = exit_price * (1 - SLIPPAGE - COMMISSION - TRANSFER_FEE - STAMP_TAX)
                gross_return = (exit_cost - entry_cost) / entry_cost
                result["exit_date"] = day
                result["exit_price"] = exit_price
                result["exit_reason"] = exit_reason
                result["return_pct"] = round(gross_return * 100, 2)
                result["holding_days"] = days_held
                result["cost_pct"] = round((1 - (1 + gross_return) / (exit_price / entry_price)) * 100, 2)
                daily_equity.append((day, 1 + gross_return))
                break

            # 记录每日权益（用于风险指标）
            current_value = close / entry_price
            daily_equity.append((day, current_value))

    # 未出场：最后一天结算
    if in_position and not result["exit_price"]:
        last_bar = bars[-1]
        exit_price = float(last_bar.get("close", 0))
        exit_cost = exit_price * (1 - COMMISSION - TRANSFER_FEE - STAMP_TAX)
        gross_return = (exit_cost - entry_cost) / entry_cost
        result["exit_date"] = last_bar.get("day", "")
        result["exit_price"] = exit_price
        result["exit_reason"] = "数据到期"
        result["return_pct"] = round(gross_return * 100, 2)
        result["holding_days"] = len(bars) - entry_idx - 1
        result["cost_pct"] = round((1 - (1 + gross_return) / (exit_price / entry_price)) * 100, 2)
        daily_equity.append((result["exit_date"], 1 + gross_return))

    # 计算权益曲线和日收益率
    if daily_equity:
        result["equity_curve"] = [(d, round(v, 4)) for d, v in daily_equity]
        daily_rets = []
        prev_v = 1.0
        for _, v in daily_equity[1:]:
            daily_rets.append((v - prev_v) / prev_v)
            prev_v = v
        result["daily_returns"] = [round(r, 6) for r in daily_rets]

    # 风险指标
    if result["daily_returns"]:
        bench_rets = _get_benchmark_returns(symbol, bars) if use_benchmark else None
        result["metrics"] = _calc_metrics(result["daily_returns"], bench_rets)

    return result


# ============================================================
#  批量回测
# ============================================================

def batch_backtest(reports: list[str], lookback_days: int = 365) -> dict:
    """批量回测：输入多份报告文本，输出汇总统计 + 风险指标 + 信号质量分析。"""
    results = []
    for r in reports:
        signal = extract_signal(r)
        if signal:
            bt = run_backtest(signal, lookback_days)
            bt["symbol"] = signal.get("symbol", "?")
            bt["entry_price"] = signal["entry_price"]
            bt["rating"] = signal.get("rating", "?")
            bt["weighted_score"] = signal.get("weighted_score", 0)
            results.append(bt)

    if not results:
        return {"error": "无有效信号", "results": []}

    triggered = [r for r in results if r.get("triggered")]
    wins = [r for r in triggered if r.get("return_pct", 0) > 0]
    losses = [r for r in triggered if r.get("return_pct", 0) <= 0]

    summary = {
        "总信号数": len(results),
        "触发数": len(triggered),
        "触发率": f"{len(triggered)/len(results)*100:.0f}%" if results else "0%",
        "胜数": len(wins),
        "败数": len(losses),
        "胜率": round(len(wins) / len(triggered) * 100, 1) if triggered else 0,
        "平均收益": round(sum(r.get("return_pct", 0) for r in triggered) / len(triggered), 1) if triggered else 0,
        "最大收益": round(max(r.get("return_pct", 0) for r in triggered), 1) if triggered else 0,
        "最大亏损": round(min(r.get("return_pct", 0) for r in triggered), 1) if triggered else 0,
        "盈亏比": round(
            sum(r.get("return_pct", 0) for r in wins) / abs(sum(r.get("return_pct", 0) for r in losses))
            if wins and losses and sum(r.get("return_pct", 0) for r in losses) != 0 else 0, 2),
        "明细": results,
    }

    # 按评级分组统计
    by_rating = {}
    for r in results:
        rating = r.get("rating", "?")
        if rating not in by_rating:
            by_rating[rating] = {"count": 0, "triggered": 0, "wins": 0, "returns": []}
        by_rating[rating]["count"] += 1
        if r.get("triggered"):
            by_rating[rating]["triggered"] += 1
            by_rating[rating]["returns"].append(r.get("return_pct", 0))
            if r.get("return_pct", 0) > 0:
                by_rating[rating]["wins"] += 1

    rating_summary = {}
    for rating, data in by_rating.items():
        if data["triggered"] > 0:
            rating_summary[rating] = {
                "信号数": data["count"],
                "触发数": data["triggered"],
                "胜率": f"{data['wins']/data['triggered']*100:.0f}%",
                "平均收益": f"{sum(data['returns'])/len(data['returns']):+.1f}%",
            }
    summary["按评级"] = rating_summary

    return summary


# ============================================================
#  框架评估：评分/评级 vs 实际收益
# ============================================================

def evaluate_framework(reports_and_prices: list[tuple[str, float]]) -> dict:
    """
    输入 [(报告文本, 报告生成时的股价), ...]，
    输出框架信度分析：高评分是否真跑赢？BUY vs HOLD vs SELL 区分度？

    核心问题：FinBrain 的评分/评级是否具有预测能力？
    """
    signals = []
    for report, gen_price in reports_and_prices:
        sig = extract_signal(report)
        if sig:
            # 先跑回测，获取实际收益
            bt = run_backtest(sig, lookback_days=365)
            sig["return_pct"] = bt.get("return_pct", 0)
            sig["triggered"] = bt.get("triggered", False)
            sig["gen_price"] = gen_price
            signals.append(sig)

    if len(signals) < 5:
        return {"error": f"有效信号不足({len(signals)}<5)，无法评估框架信度"}

    # 按加权总分组
    high_score = [s for s in signals if s.get("weighted_score", 0) >= 55]
    mid_score = [s for s in signals if 40 <= s.get("weighted_score", 0) < 55]
    low_score = [s for s in signals if s.get("weighted_score", 0) < 40]

    # 按评级分组
    buy_signals = [s for s in signals if s.get("rating") == "BUY"]
    hold_signals = [s for s in signals if s.get("rating") == "HOLD"]
    sell_signals = [s for s in signals if s.get("rating") == "SELL"]

    def _avg_return(sig_list):
        if not sig_list: return 0, 0
        rets = [s.get("return_pct", 0) for s in sig_list]
        return sum(rets) / len(rets), sum(1 for r in rets if r > 0) / len(rets)

    high_ret, high_wr = _avg_return(high_score)
    mid_ret, mid_wr = _avg_return(mid_score)
    low_ret, low_wr = _avg_return(low_score)
    buy_ret, buy_wr = _avg_return(buy_signals)
    hold_ret, hold_wr = _avg_return(hold_signals)
    sell_ret, sell_wr = _avg_return(sell_signals)

    return {
        "信号总数": len(signals),
        "按评分": {
            "高分组(≥55)": {"数量": len(high_score), "平均收益": f"{high_ret:+.1f}%", "胜率": f"{high_wr:.0%}"},
            "中分组(40-54)": {"数量": len(mid_score), "平均收益": f"{mid_ret:+.1f}%", "胜率": f"{mid_wr:.0%}"},
            "低分组(<40)": {"数量": len(low_score), "平均收益": f"{low_ret:+.1f}%", "胜率": f"{low_wr:.0%}"},
        },
        "按评级": {
            "BUY": {"数量": len(buy_signals), "平均收益": f"{buy_ret:+.1f}%", "胜率": f"{buy_wr:.0%}"},
            "HOLD": {"数量": len(hold_signals), "平均收益": f"{hold_ret:+.1f}%", "胜率": f"{hold_wr:.0%}"},
            "SELL": {"数量": len(sell_signals), "平均收益": f"{sell_ret:+.1f}%", "胜率": f"{sell_wr:.0%}"},
        },
        "评分区分度": "有效" if (high_ret > mid_ret > low_ret) else "无效或反向",
        "评级区分度": "有效" if (buy_ret > hold_ret > sell_ret) else "无效或反向",
    }


# ============================================================
#  兼容旧 API
# ============================================================

def _legacy_batch_backtest(reports: list[str], lookback_days: int = 180) -> dict:
    """旧格式兼容：返回扁平汇总，不包含风险指标。"""
    result = batch_backtest(reports, lookback_days)
    # 扁平化，去掉嵌套结构
    flat = {k: v for k, v in result.items() if k in
            ("总信号数", "触发数", "胜数", "败数", "胜率", "平均收益", "最大收益", "最大亏损", "盈亏比", "明细")}
    return flat
