"""
FinBrain 回测引擎 — 根据分析报告的买卖建议，用历史K线模拟交易，计算胜率。
纯代码，不调LLM。
"""

import re, json
from datetime import datetime, timedelta


def extract_signal(report: str) -> dict | None:
    """从报告中提取交易信号：建仓价、止损价、目标价、持仓周期。"""
    signal = {}
    # 建仓价
    m = re.search(r'[≤<=]\s*([\d.]+)\s*元.*建仓', report)
    if m: signal["entry_price"] = float(m.group(1))
    # 止损价
    m = re.search(r'止损\s*([\d.]+)\s*元', report)
    if m: signal["stop_loss"] = float(m.group(1))
    # 目标价
    m = re.search(r'目标\s*([\d.]+)\s*[-~至]\s*([\d.]+)\s*元', report)
    if m: signal["target_low"] = float(m.group(1)); signal["target_high"] = float(m.group(2))
    # 持仓周期
    m = re.search(r'持仓周期[：:]\s*(\d+)[-~至]?\d*\s*个?月', report)
    if m: signal["holding_months"] = int(m.group(1))
    # 股票代码
    m = re.search(r'(\d{6})', report)
    if m: signal["symbol"] = m.group(1)
    # 评级
    m = re.search(r'投资决策.*?(BUY|HOLD|SELL)', report)
    if m: signal["rating"] = m.group(1)

    if "entry_price" not in signal:
        return None
    signal.setdefault("holding_months", 6)
    signal.setdefault("stop_loss", signal["entry_price"] * 0.85)
    return signal


def run_backtest(signal: dict, lookback_days: int = 180) -> dict:
    """用历史K线回测单个交易信号。返回 {触发, 入场价, 出场价, 出场原因, 收益率, 持仓天数}"""
    from backend.tools import fetch_stock_history

    symbol = signal.get("symbol", "")
    if not symbol:
        return {"error": "缺少股票代码"}

    entry_price = signal["entry_price"]
    stop_loss = signal.get("stop_loss", entry_price * 0.85)
    target_low = signal.get("target_low", entry_price * 1.2)
    holding_days = signal.get("holding_months", 6) * 22  # 每月约22个交易日

    # 获取历史K线（日线，lookback_days根）
    hist = fetch_stock_history(symbol, scale=240, datalen=lookback_days)
    if "error" in hist:
        return {"error": hist["error"]}
    bars = hist.get("data", [])
    if len(bars) < 5:
        return {"error": "K线数据不足"}

    # 从最早到最新遍历，模拟入场→持有→出场
    result = {"triggered": False, "entry_date": "", "entry_price": 0,
              "exit_date": "", "exit_price": 0, "exit_reason": "未触发",
              "return_pct": 0, "holding_days": 0}

    in_position = False
    entry_idx = -1
    for i, bar in enumerate(bars):
        close = float(bar.get("close", 0))
        low = float(bar.get("low", close))
        high = float(bar.get("high", close))
        day = bar.get("day", "")

        if not in_position:
            # 检查是否触及建仓价（当日最低价≤建仓价）
            if low <= entry_price and close > 0:
                in_position = True
                entry_idx = i
                result["triggered"] = True
                result["entry_date"] = day
                result["entry_price"] = entry_price
        else:
            days_held = i - entry_idx
            # 止损检查
            if close <= stop_loss:
                result["exit_date"] = day
                result["exit_price"] = stop_loss
                result["exit_reason"] = "止损"
                result["return_pct"] = round((stop_loss - entry_price) / entry_price * 100, 1)
                result["holding_days"] = days_held
                return result
            # 止盈检查
            if high >= target_low:
                result["exit_date"] = day
                result["exit_price"] = target_low
                result["exit_reason"] = "止盈"
                result["return_pct"] = round((target_low - entry_price) / entry_price * 100, 1)
                result["holding_days"] = days_held
                return result
            # 超时
            if days_held >= holding_days:
                result["exit_date"] = day
                result["exit_price"] = close
                result["exit_reason"] = "到期"
                result["return_pct"] = round((close - entry_price) / entry_price * 100, 1)
                result["holding_days"] = days_held
                return result

    # 遍历完未出场：以最后一天收盘价结算
    if in_position:
        last_bar = bars[-1]
        result["exit_date"] = last_bar.get("day", "")
        result["exit_price"] = float(last_bar.get("close", 0))
        result["exit_reason"] = "数据到期"
        if result["entry_price"] > 0:
            result["return_pct"] = round((result["exit_price"] - result["entry_price"]) / result["entry_price"] * 100, 1)
        result["holding_days"] = len(bars) - entry_idx - 1

    return result


def batch_backtest(reports: list[str], lookback_days: int = 180) -> dict:
    """批量回测：输入多份报告文本，输出汇总统计。"""
    results = []
    for r in reports:
        signal = extract_signal(r)
        if signal:
            bt = run_backtest(signal, lookback_days)
            bt["symbol"] = signal.get("symbol", "?")
            bt["entry_price"] = signal["entry_price"]
            bt["rating"] = signal.get("rating", "?")
            results.append(bt)

    if not results:
        return {"error": "无有效信号", "results": []}

    triggered = [r for r in results if r.get("triggered")]
    wins = [r for r in triggered if r.get("return_pct", 0) > 0]
    losses = [r for r in triggered if r.get("return_pct", 0) <= 0]

    return {
        "总信号数": len(results),
        "触发数": len(triggered),
        "胜数": len(wins),
        "败数": len(losses),
        "胜率": round(len(wins) / len(triggered) * 100, 1) if triggered else 0,
        "平均收益": round(sum(r.get("return_pct", 0) for r in triggered) / len(triggered), 1) if triggered else 0,
        "最大收益": round(max(r.get("return_pct", 0) for r in triggered), 1) if triggered else 0,
        "最大亏损": round(min(r.get("return_pct", 0) for r in triggered), 1) if triggered else 0,
        "明细": results,
    }
