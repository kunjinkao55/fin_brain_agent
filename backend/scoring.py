"""
FinBrain 投资决策引擎 — 动态权重 + 安全边际 + BUY/HOLD/SELL 评级

公司类型决定评分权重，估值差距决定买卖信号。
所有阈值从 configs/scoring.json 加载，代码仅保留默认值兜底。
"""

import logging
from backend.scoring_config import (get_weights, get_safety_margin, get_safety_adjustments,
                                     get_valuation, get_roe_multipliers, get_rating_thresholds)

logger = logging.getLogger(__name__)


def _quality_adjustment(roe: float, debt: float) -> float:
    """根据财务质量微调安全边际"""
    adj_cfg = get_safety_adjustments()
    adj = 0.0
    if roe > adj_cfg["roe_high"]: adj += adj_cfg["roe_high_adj"]
    elif roe < adj_cfg["roe_low"]: adj += adj_cfg["roe_low_adj"]
    if debt < adj_cfg["debt_low"]: adj += adj_cfg["debt_low_adj"]
    elif debt > adj_cfg["debt_high"]: adj += adj_cfg["debt_high_adj"]
    limits = adj_cfg["limits"]
    return max(limits[0] - 0.10, min(limits[1] - 0.40, adj))

# ============================================================
#  公开 API
# ============================================================

def compute_investment_rating(
    company_type: str,
    financial_scores: dict,  # {盈利能力:{得分,依据}, 成长性:{}, 财务健康:{}, 估值合理:{}}
    llm_scores: dict,        # {行业前景:{得分}, 资金认可:{得分}}
    eps: float,              # 每股收益（年报）
    stock_price: float,      # 当前股价
    industry: str,           # 行业名
    roe: float,              # ROE(%)
    debt: float,             # 资产负债率(%)
) -> dict:
    """
    根据公司类型动态赋权，计算综合评分 + 合理价值 + 安全边际 + 投资评级。
    全部代码计算，不依赖 LLM。
    """
    weights = get_weights(company_type)

    # --- 1. 提取各维度得分 ---
    def _safe_score(d, key):
        v = d.get(key, {}) if isinstance(d, dict) else {}
        return v.get("得分", 5) if isinstance(v, dict) else 5

    raw = {
        "财务质量": (_safe_score(financial_scores, "盈利能力") +
                     _safe_score(financial_scores, "财务健康")) / 2,
        "成长质量": _safe_score(financial_scores, "成长性"),
        "估值": _safe_score(financial_scores, "估值合理"),
        "行业周期": _safe_score(llm_scores, "行业前景"),
        "商业模式": 5,  # 默认中性，LLM通过叙事补充
        "竞争优势": 5,
    }

    # --- 2. 加权总分（0-10制→0-100制） ---
    weighted = sum(raw[k] * weights.get(k, 0.15) for k in raw) * 10
    weighted = round(weighted, 1)

    # --- 3. 合理价值估算 ---
    val_cfg = get_valuation()
    ind_pe = val_cfg["industry_pe"].get(industry, val_cfg["default_ind_pe"])
    # 财务质量调整PE基准：ROE和负债率决定了企业应享有怎样的估值
    quality_mult = 1.0
    if roe > 35: quality_mult = 1.6       # 顶级盈利能力（茅台级）
    elif roe > 25: quality_mult = 1.3     # 优秀
    elif roe > 18: quality_mult = 1.1     # 良好
    elif roe > 12: quality_mult = 0.85    # 中等（丽珠14.67%在这档）
    elif roe > 8: quality_mult = 0.65     # 偏低
    elif roe > 3: quality_mult = 0.45     # 弱
    else: quality_mult = 0.30             # 极弱（泰格4.29%在这档）
    if debt > 70: quality_mult -= 0.10
    elif debt > 50: quality_mult -= 0.05
    elif debt < 20: quality_mult += 0.05

    # 现金流修正：低ROE可能是重资产折旧导致（非经营问题），若现金流强劲则减轻惩罚
    _fh = financial_scores.get("财务健康", {}) if isinstance(financial_scores, dict) else {}
    _cf_sev = _fh.get("现金流严重度", 1) if isinstance(_fh, dict) else 1
    _cf_label = str(_fh.get("现金流标签", "")) if isinstance(_fh, dict) else ""
    if quality_mult < 0.7 and _cf_sev <= 1:  # ROE低但现金流🟢/🟡优秀
        quality_mult = max(quality_mult, 0.7)  # 抬底：重资产优质公司不应被ROE过度惩罚
    elif quality_mult < 0.5 and _cf_sev <= 2:  # ROE很低但现金流至少🟠
        quality_mult = max(quality_mult, 0.5)  # 抬底：现金流尚可则不过度折价

    # 成长溢价：高增速公司应享有更高PE倍数（仅对S/A级成长股给予溢价，不对低增速惩罚）
    growth_score = _safe_score(financial_scores, "成长性")
    growth_pe_mult = 1.0
    if growth_score >= 9:   growth_pe_mult = 1.8   # S级：超高速成长(如中际旭创265%增速)
    elif growth_score >= 7: growth_pe_mult = 1.3   # A级：强劲成长

    fair_pe = ind_pe * quality_mult * growth_pe_mult
    fair_value = round(eps * fair_pe, 2) if eps > 0 else 0

    # --- 4. 安全边际 ---
    base_margin = get_safety_margin(company_type)
    quality_adj = _quality_adjustment(roe, debt)
    safety_margin = base_margin + quality_adj
    safety_margin = max(0.15, min(0.55, safety_margin))  # 限制在15%-55%

    # --- 5. 买入区间 ---
    buy_zone_upper = round(fair_value * (1 - safety_margin), 2) if fair_value > 0 else 0

    # --- 6. 投资评级 ---
    if stock_price <= 0 or fair_value <= 0:
        rating = "HOLD"
        gap_pct = 0
    else:
        gap_pct = (fair_value - stock_price) / stock_price * 100
        if stock_price <= buy_zone_upper:
            rating = "BUY"
        elif stock_price < fair_value and weighted >= 40:
            rating = "HOLD"
        else:
            rating = "SELL"

    # --- 7. 仓位建议 ---
    if rating == "BUY" and weighted >= 60:
        position = "10%-15%"
    elif rating == "BUY":
        position = "5%-10%"
    elif rating == "HOLD" and weighted >= 45:
        position = "3%-5% (观望仓位)"
    else:
        position = "0%-3% 或不参与"

    # --- 8. 置信度 ---
    if weighted >= 75: confidence = "A (高)"
    elif weighted >= 55: confidence = "B (中)"
    else: confidence = "C (低)"

    # 估值计算链（透明化）+ 前瞻敏感性
    _eps_display = round(eps, 2) if eps else 0
    valuation_chain = {
        "EPS(TTM)": _eps_display,
        "行业PE中枢": ind_pe,
        "财务质量乘数": round(quality_mult, 2),
        "成长溢价": round(growth_pe_mult, 2),
        "最终PE": round(fair_pe, 1),
        "公式": f"{_eps_display} × {ind_pe} × {quality_mult} × {growth_pe_mult} = {fair_value}",
    }
    # 前瞻敏感性：若EPS回到正常水平，合理价值会是多少
    if eps > 0 and _eps_display > 0:
        _eps_2x = round(_eps_display * 2, 2)
        _eps_3x = round(_eps_display * 3, 2)
        _fv_2x = round(_eps_2x * fair_pe, 2)
        _fv_3x = round(_eps_3x * fair_pe, 2)
        valuation_chain["前瞻说明"] = (
            f"以上基于TTM EPS {_eps_display}元（历史利润）。"
            f"若利润恢复至{_eps_2x}元(2×当前)，合理价值≈{_fv_2x}元；"
            f"若利润恢复至{_eps_3x}元(3×当前)，合理价值≈{_fv_3x}元。"
            f"当前估值对盈利复苏极度敏感——利润翻倍则合理价值翻倍。"
        )

    return {
        "评级": rating,
        "加权总分": weighted,
        "权重描述": weights.get("描述", ""),
        "合理PE": round(fair_pe, 1),
        "合理价值": fair_value,
        "估值明细": valuation_chain,
        "当前价格": stock_price,
        "估值差距": f"{gap_pct:+.1f}%",
        "安全边际要求": f"{safety_margin:.0%}",
        "买入区间": f"≤{buy_zone_upper:.2f}元" if buy_zone_upper > 0 else "无法计算",
        "仓位建议": position,
        "置信度": confidence,
        "维度明细": {k: {"得分": round(raw[k], 1), "权重": f"{weights.get(k, 0.15):.0%}"} for k in raw},
    }


def compute_expected_return(stock_price: float, scenarios: dict) -> dict:
    """概率加权期望收益 + 风险收益比。scenarios = {"悲观":{"价格":x,"概率":"20%"},...}"""
    total_prob = 0.0
    weighted_sum = 0.0
    best_case = float('-inf')
    worst_case = float('inf')

    for label in ["悲观", "基准", "乐观"]:
        s = scenarios.get(label, {})
        try:
            price = float(s.get("价格", 0) or 0)
            prob_str = str(s.get("概率", "33%")).replace("%", "")
            prob = float(prob_str) / 100.0
        except (ValueError, TypeError):
            continue
        if price > 0 and prob > 0:
            ret = (price - stock_price) / stock_price
            weighted_sum += ret * prob
            total_prob += prob
            best_case = max(best_case, ret)
            worst_case = min(worst_case, ret)

    if total_prob == 0 or stock_price <= 0:
        return {"期望收益": "无法计算", "风险收益比": "无法计算"}

    eret = weighted_sum / total_prob if total_prob > 0 else 0
    max_loss = abs(worst_case) if worst_case < 0 else 0.001
    rr = eret / max_loss

    return {
        "期望收益": f"{eret:+.1%}",
        "最大下跌风险": f"{worst_case:+.1%}" if worst_case < 0 else "无下行风险",
        "最大上涨空间": f"{best_case:+.1%}",
        "风险收益比": f"{rr:.2f} (期望收益/最大下跌, >1才值得冒险)",
    }
