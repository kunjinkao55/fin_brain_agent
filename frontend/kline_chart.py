"""
FinBrain K线图模块 — 分时/五日/日K/周K/月K
数据粒度: 分时5min | 五日5min | 日K daily | 周K weekly(聚合) | 月K monthly(聚合)
"""
from datetime import datetime, timedelta
from collections import OrderedDict

import plotly.graph_objects as go


def _calc_sma(values: list[float], period: int) -> list[float | None]:
    """简单移动平均。前 period-1 位填 None 使长度对齐。"""
    if len(values) < period:
        return [None] * len(values)
    result = [None] * (period - 1)
    for i in range(period - 1, len(values)):
        result.append(sum(values[i - period + 1 : i + 1]) / period)
    return result


def _aggregate_weekly(daily_data: list[dict]) -> list[dict]:
    """日线→周线: 周一开盘, 周五收盘, 周最高/最低/总成交量"""
    weeks = OrderedDict()
    for d in daily_data:
        day_str = d["day"]
        try:
            dt = datetime.strptime(day_str, "%Y-%m-%d")
            week_key = dt.strftime("%Y-W%W")
        except ValueError:
            continue
        o, h, l, c = float(d["open"]), float(d["high"]), float(d["low"]), float(d["close"])
        vol = int(d["volume"])
        if week_key not in weeks:
            weeks[week_key] = {"day": day_str, "open": o, "high": h, "low": l, "close": c, "volume": vol}
        else:
            w = weeks[week_key]
            w["high"] = max(w["high"], h)
            w["low"] = min(w["low"], l)
            w["close"] = c
            w["day"] = day_str
            w["volume"] += vol
    return list(weeks.values())


def _aggregate_monthly(daily_data: list[dict]) -> list[dict]:
    """日线→月线: 月首开盘, 月末收盘, 月最高/最低/总成交量"""
    months = OrderedDict()
    for d in daily_data:
        day_str = d["day"]
        month_key = day_str[:7]
        o, h, l, c = float(d["open"]), float(d["high"]), float(d["low"]), float(d["close"])
        vol = int(d["volume"])
        if month_key not in months:
            months[month_key] = {"day": day_str, "open": o, "high": h, "low": l, "close": c, "volume": vol}
        else:
            m = months[month_key]
            m["high"] = max(m["high"], h)
            m["low"] = min(m["low"], l)
            m["close"] = c
            m["day"] = day_str
            m["volume"] += vol
    return list(months.values())


def build_kline_chart(symbol: str, name: str, timeframe: str) -> go.Figure | None:
    """
    根据时间粒度拉取数据并构建 Plotly 图表。
    返回 Figure 对象, 调用方用 st.plotly_chart(fig) 渲染。
    """
    from backend.tools import fetch_stock_history, get_intraday

    # ---- 数据获取 ----
    if timeframe == "分时":
        raw = get_intraday(symbol)
        if not raw or "error" in raw or not raw.get("bars"):
            raw = fetch_stock_history(symbol, scale=5, datalen=240)
            today = datetime.now().strftime("%Y-%m-%d")
            raw_bars = [b for b in raw.get("data", []) if b["day"].startswith(today)]
            raw = {"bars": [{"time": b["day"][-8:], "close": float(b["close"]), "open": float(b["open"]),
                             "high": float(b["high"]), "low": float(b["low"])} for b in raw_bars],
                   "date": today, "count": len(raw_bars)}
        if not raw.get("bars"):
            return None
        bars = raw["bars"]
        today_str = raw.get("date", datetime.now().strftime("%Y-%m-%d"))

        closes = [b["close"] for b in bars]
        # 手动裁掉午休11:30-13:00：x轴用"交易分钟数"，早盘和下午盘无缝拼接
        trading_min = []
        tick_positions = []
        tick_texts = []
        for b in bars:
            t = b["time"]
            hh, mm = int(t[:2]), int(t[3:5])
            total = hh * 60 + mm
            if 11 * 60 + 30 < total < 13 * 60:
                continue  # 跳过午休（不应有数据）
            if total >= 13 * 60:
                total -= 90  # 下午盘减掉午休90分钟
            trading_min.append(total)

        # 生成完整交易时间刻度
        for h in range(9, 12):
            for m in [0, 30]:
                if h == 9 and m == 0: continue
                if h == 11 and m == 30: continue
                tick_positions.append(h * 60 + m)
                tick_texts.append(f"{h:02d}:{m:02d}")
        for h in range(13, 16):
            for m in [0, 30]:
                tick_positions.append(h * 60 + m - 90)
                tick_texts.append(f"{h:02d}:{m:02d}")
                if h == 15 and m == 0: break

        closes_plot = closes[:len(trading_min)]
        title = f"{name} 分时图 ({today_str})"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trading_min, y=closes_plot, mode="lines",
            line=dict(color="#cc3333", width=1.5),
            name="价格", fill="tozeroy", fillcolor="rgba(204,51,51,0.08)",
        ))
        y_lo, y_hi = min(closes_plot), max(closes_plot)
        y_pad = max((y_hi - y_lo) * 0.1, 0.02)
        fig.update_layout(
            yaxis=dict(range=[y_lo - y_pad, y_hi + y_pad]),
            xaxis=dict(
                tickmode="array", tickvals=tick_positions, ticktext=tick_texts,
                range=[tick_positions[0], tick_positions[-1]],
            ),
        )

    elif timeframe == "五日":
        # 5日 5分钟线 — 每根5min, 240根覆盖近5个交易日
        raw = fetch_stock_history(symbol, scale=5, datalen=240)
        if "error" in raw or not raw.get("data"):
            return None
        data = raw["data"]
        x = [d["day"] for d in data]  # "2026-07-15 09:35:00" → 取时间部分
        closes = [float(d["close"]) for d in data]
        title = f"{name} 五日分时"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=closes, mode="lines", line=dict(color="#cc3333", width=1.2),
            name="收盘价",
        ))
        # 简化x轴标签: 只显示日期
        tickvals = [d["day"] for d in data if "09:35" in d["day"] or "10:00" in d["day"]][::12]
        if tickvals:
            fig.update_layout(xaxis=dict(tickvals=tickvals, ticktext=[v[:10] for v in tickvals]))

    else:
        # 日K/周K/月K — 蜡烛图 + 均线
        tf_config = {
            "日K": {"scale": 240, "datalen": 30, "label": "日K线(1个月)"},
            "周K": {"scale": 240, "datalen": 130, "label": "周K线(6个月)"},
            "月K": {"scale": 240, "datalen": 500, "label": "月K线(2年)"},
        }
        cfg = tf_config[timeframe]
        raw = fetch_stock_history(symbol, scale=cfg["scale"], datalen=cfg["datalen"])
        if "error" in raw or not raw.get("data"):
            return None
        data = raw["data"]

        if timeframe == "周K":
            data = _aggregate_weekly(data)
        elif timeframe == "月K":
            data = _aggregate_monthly(data)

        dates = [d["day"] for d in data]
        opens = [float(d["open"]) for d in data]
        highs = [float(d["high"]) for d in data]
        lows = [float(d["low"]) for d in data]
        closes = [float(d["close"]) for d in data]
        volumes = [int(d["volume"]) for d in data]

        # 成交量柱(下半部分) + 蜡烛图(上半部分) — 双y轴
        fig = go.Figure()

        # 蜡烛图
        fig.add_trace(go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name="K线", yaxis="y",
            increasing=dict(line=dict(color="#cc3333", width=1), fillcolor="#cc3333"),
            decreasing=dict(line=dict(color="#2e7d32", width=1), fillcolor="#2e7d32"),
        ))

        # 均线
        ma_configs = [
            (5, "#f4a261"), (10, "#e9c46a"),
            (20, "#457b9d"), (60, "#a855f7"),
        ]
        for period, color in ma_configs:
            if len(closes) >= period:
                ma = _calc_sma(closes, period)
                fig.add_trace(go.Scatter(
                    x=dates, y=ma, mode="lines", yaxis="y",
                    line=dict(color=color, width=1.0),
                    name=f"MA{period}",
                ))

        # 成交量柱
        vol_colors = ["#cc3333" if closes[i] >= opens[i] else "#2e7d32" for i in range(len(closes))]
        fig.add_trace(go.Bar(
            x=dates, y=volumes, yaxis="y2",
            marker_color=vol_colors, opacity=0.35,
            name="成交量", showlegend=False,
        ))

        title = f"{name} {cfg['label']}"
        fig.update_layout(
            yaxis=dict(title="价格", domain=[0.25, 1.0], showgrid=True, gridcolor="#333"),
            yaxis2=dict(title="成交量", domain=[0, 0.20], showgrid=False),
            xaxis=dict(rangeslider=dict(visible=False)),
        )

    # ---- 公共样式 ----
    fig.update_layout(
        title=title,
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="#ddd", size=11),
        xaxis=dict(showgrid=True, gridcolor="#333"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
    )
    return fig
