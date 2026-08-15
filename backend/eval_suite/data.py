"""
Layer A — 数据管道解耦评估（data_collector / tools 抓取正确性）。

目的（开发期调试定位）：把「数据管道正确性」和「agent 推理正确性」分开评估。
免费 API 数据错（股本未除权、财报滞后、字段缺失）不算 agent 幻觉——
本模块直接对抓取结果做黄金事实比对，定位数据层错误。

指标：
    1. 字段抓取成功/失败率  — 每类数据源是否拿到
    2. 黄金数字误差         — 总股本/BPS/归母净利润/TTM EPS vs 上市公告书口径
    3. 财报时效             — 最新报告期是否已达预期（快报双通道是否生效）
    4. 跨源一致性           — 估值接口 vs 财报接口的同字段比对

黄金事实来自 golden_pool：
    real 股（301583/600131）做精确数值比对；
    synthetic 股做宽松校验（结构完整性 + 内部自洽），不做精确值断言。

设计：
    extract_metrics(collected)      纯函数 —— 从抓取 dict 提取关键数字，可离线 CI
    check_against_golden(metrics)   纯函数 —— 与黄金事实比对，产出违规列表
    collect_data(symbol)            真实运行器 —— 调 data_collector 同款工具（网络）
    evaluate_data_pipeline()        真实评估入口 —— 供 CLI / 周度监控
"""

from backend.eval_suite.golden_pool import build_golden_data_facts, get_stock

# 各字段容差（相对偏差）：总股本/归母为硬口径严查；BPS/TTM 允许一定舍入
_GOLDEN_TOL = {
    "总股本(股)": 0.005,      # 0.5%：发行后股本为公告口径，应精确
    "每股净资产(元)": 0.05,    # 5%：发行后 BPS 9.87
    "2025年报归母(元)": 0.05,  # 5%：0.9818亿
    "2026Q1归母(元)": 0.05,    # 5%：0.1464亿
    "TTM归母(元)": 0.05,       # 5%：0.9682亿
    "TTM EPS": 0.05,           # 5%：0.522
}

def _pool_symbols() -> list[str]:
    from backend.eval_suite.golden_pool import GOLDEN_STOCKS
    return list(GOLDEN_STOCKS.keys())


GOLDEN_FACTS = {s: build_golden_data_facts(s) for s in _pool_symbols()}

def _golden_for(symbol: str) -> dict | None:
    """取某只股票的 Layer A 黄金事实。synthetic 股返回宽松标记。"""
    from backend.eval_suite.golden_pool import GOLDEN_STOCKS
    stock = GOLDEN_STOCKS.get(symbol)
    if not stock:
        return None
    facts = build_golden_data_facts(symbol)
    facts["_source"] = stock.source
    return facts

# 应成功抓取的字段（缺任一即数据管道失败）
REQUIRED_KEYS = ["总股本(股)", "每股净资产(元)", "2025年报归母(元)", "TTM EPS"]


# ============================================================
#  提取器（纯函数，不调 LLM / 不访问网络）
# ============================================================

def _num(v):
    """宽松转 float；None/''/非数字返回 None。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_metrics(collected: dict) -> dict:
    """从 data_collector 抓取结构提取关键数字。

    collected 结构（与 data_collector 输出一致）：
        {代码, 行情:{当前价格,...}, 估值:{data:[{总股本,每股净资产,每股收益,...}]},
         财报:{利润表:[{报告期,归母净利润,...}]}, 公告:{列表:[{快报数据}]}}
    """
    metrics = {"代码": collected.get("代码", "?")}

    # 行情
    price = _num((collected.get("行情") or {}).get("当前价格"))
    metrics["当前价格(元)"] = price

    # 估值：取最新一条
    val_rows = (collected.get("估值") or {}).get("data") or []
    if val_rows:
        r0 = val_rows[0]
        metrics["总股本(股)"] = _num(r0.get("总股本"))
        metrics["每股净资产(元)"] = _num(r0.get("每股净资产"))
        metrics["每股收益(元)"] = _num(r0.get("每股收益"))
        metrics["最新估值报告期"] = r0.get("报告期", "")
    else:
        metrics["总股本(股)"] = None
        metrics["每股净资产(元)"] = None
        metrics["每股收益(元)"] = None
        metrics["最新估值报告期"] = ""

    # 财报利润表：归母净利润（最新年报 + 最新一季报）+ 最新报告期
    profit_rows = (collected.get("财报") or {}).get("利润表") or []
    metrics["最新报告期"] = ""
    metrics["2025年报归母(元)"] = None
    metrics["2026Q1归母(元)"] = None
    for r in profit_rows:
        period = r.get("报告期", "")
        if not metrics["最新报告期"] and period:
            metrics["最新报告期"] = period
        ni = _num(r.get("归母净利润"))
        if ni is None:
            continue
        if period == "年报":
            metrics["2025年报归母(元)"] = ni
        elif period == "一季报" and metrics["2026Q1归母(元)"] is None:
            metrics["2026Q1归母(元)"] = ni

    # TTM 归母 = 最新年报 + 最新一季报 - 上年一季报（需 3 行）
    q_last, q_prev, annual = None, None, None
    for r in profit_rows:
        ni = _num(r.get("归母净利润"))
        if ni is None:
            continue
        period = r.get("报告期", "")
        if period == "年报":
            annual = ni
        elif period == "一季报":
            if q_last is None:
                q_last = ni
            elif q_prev is None:
                q_prev = ni
    if annual is not None and q_last is not None and q_prev is not None:
        metrics["TTM归母(元)"] = annual + q_last - q_prev
    else:
        metrics["TTM归母(元)"] = None
    if metrics.get("总股本(股)") and metrics.get("TTM归母(元)"):
        metrics["TTM EPS"] = metrics["TTM归母(元)"] / metrics["总股本(股)"]
    else:
        metrics["TTM EPS"] = None

    # 公告：是否含业绩快报
    anns = (collected.get("公告") or {}).get("列表") or []
    metrics["含快报"] = any(isinstance(a, dict) and a.get("快报数据") for a in anns)
    return metrics


def check_against_golden(metrics: dict, symbol: str) -> list[str]:
    """与黄金事实比对，返回违规描述列表（空 = 数据管道通过）。

    real 股做精确数值比对；synthetic 股只做结构完整性 + 内部自洽校验。
    """
    golden = _golden_for(symbol)
    if not golden:
        return [f"无黄金事实定义: {symbol}"]
    source = golden.pop("_source", "real")
    violations = []

    if source == "real":
        for key, expected in golden.items():
            if key == "最新报告期":
                continue
            tol = _GOLDEN_TOL.get(key, 0.05)
            actual = metrics.get(key)
            if actual is None:
                violations.append(f"[{key}] 抓取缺失（应≈{expected:.4g}）")
                continue
            denom = max(abs(actual), abs(expected))
            if denom > 0 and abs(actual - expected) / denom > tol:
                violations.append(
                    f"[{key}] 偏差 {abs(actual - expected)/denom:.1%}（应 {expected:.4g}，实际 {actual:.4g}，容差 {tol:.0%}）")
    else:
        # synthetic 股：仅校验结构完整 + TTM 自洽
        for key in REQUIRED_KEYS:
            if metrics.get(key) is None:
                violations.append(f"[必需字段] {key} 抓取失败")
        ttm = metrics.get("TTM归母(元)")
        if ttm is None:
            violations.append("[TTM自洽] 无法计算 TTM 归母（利润表缺年报或两期一季报）")
        elif ttm < 0:
            violations.append(f"[TTM自洽] TTM 归母为负（{ttm/1e8:.2f}亿），数据口径可疑")

    # 财报时效：最新报告期应达到预期（快报双通道）
    expected_periods = golden.get("最新报告期", [])
    actual_period = metrics.get("最新报告期", "")
    if actual_period not in expected_periods and actual_period:
        violations.append(
            f"[财报时效] 最新报告期「{actual_period}」未达预期（应含 {'/'.join(expected_periods)}）——"
            "快报双通道或财报接口可能滞后")

    # 必需字段缺失（real 股补充：数值已比对，仍需全字段）
    if source == "real":
        for key in REQUIRED_KEYS:
            if metrics.get(key) is None:
                violations.append(f"[必需字段] {key} 抓取失败")

    return violations


# ============================================================
#  真实运行器（调 data_collector 同款工具，需网络，供 CLI / 周度监控）
# ============================================================

def collect_data(symbol: str) -> dict:
    """调用与 data_collector_node 相同的工具，构造抓取结构（网络模式）。"""
    from backend.tools import (get_financial_statements, get_valuation,
                               fetch_stock_price, get_recent_announcements)
    fin = get_financial_statements(symbol)
    val = get_valuation(symbol)
    price = fetch_stock_price(symbol)
    anns = get_recent_announcements(symbol, 20)
    return {
        "代码": symbol,
        "行情": price if isinstance(price, dict) else {},
        "估值": val if isinstance(val, dict) else {},
        "财报": {"利润表": (fin.get("profit") or []) if isinstance(fin, dict) else []},
        "公告": anns if isinstance(anns, dict) else {},
        "抓取错误": {
            "财报": fin.get("error", "") if isinstance(fin, dict) else "",
            "估值": val.get("error", "") if isinstance(val, dict) else "",
            "行情": price.get("error", "") if isinstance(price, dict) else "",
            "公告": anns.get("error", "") if isinstance(anns, dict) else "",
        },
    }


def evaluate_data_pipeline(symbols: list[str] | None = None) -> dict:
    """对多只股票跑数据管道黄金比对，聚合报告。"""
    symbols = symbols or ["301583"]
    per_stock = []
    for sym in symbols:
        try:
            collected = collect_data(sym)
            metrics = extract_metrics(collected)
            violations = check_against_golden(metrics, sym)
            errors = {k: v for k, v in (collected.get("抓取错误") or {}).items() if v}
            per_stock.append({
                "代码": sym,
                "抓取错误": errors,
                "指标": metrics,
                "违规": violations,
                "通过": not violations and not errors,
            })
        except Exception as e:
            per_stock.append({"代码": sym, "通过": False, "违规": [f"采集异常: {e}"], "抓取错误": {}})

    passed = sum(1 for r in per_stock if r.get("通过"))
    return {
        "股票数": len(per_stock),
        "通过数": passed,
        "通过率": round(passed / len(per_stock), 3) if per_stock else 0.0,
        "逐只": per_stock,
    }


def format_pipeline_report(report: dict) -> str:
    """人类可读报告。"""
    lines = []
    lines.append("=" * 64)
    lines.append("FinBrain Layer A — 数据管道黄金比对报告")
    lines.append("=" * 64)
    lines.append(f"通过 {report['通过数']}/{report['股票数']} = {report['通过率']:.0%}")
    for r in report["逐只"]:
        status = "通过" if r.get("通过") else "违规 !!"
        lines.append(f"\n[{r['代码']}] {status}")
        if r.get("抓取错误"):
            lines.append(f"  抓取错误: {' | '.join(f'{k}:{v}' for k, v in r['抓取错误'].items())}")
        if r.get("违规"):
            for v in r["违规"]:
                lines.append(f"  ✗ {v}")
        m = r.get("指标") or {}
        if m:
            lines.append(f"  总股本: {m.get('总股本(股)')}  BPS: {m.get('每股净资产(元)')}")
            lines.append(f"  TTM归母: {m.get('TTM归母(元)')}  TTM EPS: {m.get('TTM EPS')}")
            lines.append(f"  最新报告期: {m.get('最新报告期')}  含快报: {m.get('含快报')}")
    return "\n".join(lines)
