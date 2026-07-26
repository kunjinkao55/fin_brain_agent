"""
FinBrain 计算验证层（FIX-04 + FIX-11 + FIX-12）

背景：LLM 在报告中直接书写未经计算的数值结论，导致事实性错误
（托伦斯案例：写"隐含3年增速40%"而正确值≈102%；写"30元≈PE45倍"实为58倍；
写"100元=跌破PB15倍"实为10.1倍）。

本模块提供三层防线：
    1. CalcTable 计算登记表：所有写入报告的派生数值必须先登记（含人类可读
       公式与输入快照），渲染层只引用登记值，杜绝 LLM 自由书写计算结果。
    2. 纯函数计算 helpers + 单位/舍入规范（FIX-11）：亿/万/百分比/价格统一渲染，
       禁止"1.89亿"被进位成"2亿"这类舍入失真。
    3. verify_report_text 报告数字回溯验证（FIX-04）：标签锚定式扫描报告文本，
       出现值与登记值、以及同一指标多处出现值之间做容差比对。
    另含 cross_validate_fields 跨源字段比对（FIX-12）。

仅使用标准库。

用法：
    from backend.calc_engine import CalcTable, implied_cagr, verify_report_text
    ct = CalcTable()
    ct.register("市值", 322.7, formula="174.00 × 1.8547亿股 = 322.7亿",
                inputs={"price": 174.0, "total_shares": 1.8547e8}, source="quote")
    violations = verify_report_text(report_text, ct)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
#  计算登记表
# ============================================================

@dataclass
class CalcEntry:
    """一条已登记的计算结果（含公式与输入快照，可回溯血缘）。"""
    name: str                          # 指标名，如 "市值"、"PE"、"PB"、"合理价值"、"隐含增速"
    value: float
    formula: str = ""                  # 人类可读公式，如 "174.00 × 1.8547亿股 = 322.7亿"
    inputs: dict = field(default_factory=dict)  # 输入快照 {参数名: 值}
    source: str = ""                   # 血缘（数据来源/计算步骤标识）


class CalcTable:
    """计算登记表：name -> CalcEntry。同名重复注册将覆盖为最新值。"""

    def __init__(self) -> None:
        self._entries: dict[str, CalcEntry] = {}

    def register(self, name: str, value: float, formula: str = "",
                 inputs: Optional[dict] = None, source: str = "") -> None:
        """登记一个计算指标。"""
        self._entries[name] = CalcEntry(
            name=name,
            value=float(value),
            formula=formula,
            inputs=dict(inputs or {}),
            source=source,
        )

    def get(self, name: str) -> Optional[CalcEntry]:
        """按指标名取登记条目，未注册返回 None。"""
        return self._entries.get(name)

    def names(self) -> list[str]:
        """全部已注册指标名（注册顺序）。"""
        return list(self._entries.keys())

    def as_dict(self) -> dict:
        """{name: value} 扁平字典，供渲染层引用。"""
        return {k: e.value for k, e in self._entries.items()}


# ============================================================
#  纯函数计算 helpers
# ============================================================

def implied_cagr(price: float, total_shares: float, base_net_profit: float,
                 target_pe: float, years: int = 3) -> float:
    """市场隐含净利润年复合增速（返回小数，1.02 表示 102%）。

    逻辑：现价市值 M = price × total_shares；假设 years 年后 PE 回归 target_pe
    且股价不变，则需净利润 N = M / target_pe；
    CAGR = (N / base_net_profit)^(1/years) - 1。

    base_net_profit <= 0（亏损）等非法输入返回 0.0。

    托伦斯验证：price=174, total_shares=1.8547e8, base=0.9818e8,
    target_pe=40, years=3 → ≈1.02（即 102%）。
    """
    if base_net_profit <= 0 or price <= 0 or total_shares <= 0 \
            or target_pe <= 0 or years <= 0:
        return 0.0
    market_cap = price * total_shares
    required_profit = market_cap / target_pe
    return (required_profit / base_net_profit) ** (1.0 / years) - 1.0


def pe_at_price(price: float, eps: float) -> float:
    """价格对应的 PE 倍数；eps <= 0（亏损）返回 0.0，此时 PE 无意义。"""
    return price / eps if eps > 0 else 0.0


def pb_at_price(price: float, bps: float) -> float:
    """价格对应的 PB 倍数；bps <= 0（资不抵债）返回 0.0。"""
    return price / bps if bps > 0 else 0.0


# ============================================================
#  单位与舍入规范（FIX-11）
# ============================================================

def fmt_yi(x: float, digits: int = 2) -> str:
    """元 → 中文单位字符串。

    |x| >= 1e8 → "X.XX亿"；|x| >= 1e4 → "X万"（整数）；否则 "X"（整数）。
    """
    ax = abs(x)
    if ax >= 1e8:
        return f"{x / 1e8:.{digits}f}亿"
    if ax >= 1e4:
        return f"{x / 1e4:.0f}万"
    return f"{x:.0f}"


def fmt_yi_amount(x_yi: float, digits: int = 2) -> str:
    """入参已是"亿"为单位的金额 → 中文单位字符串。

    >= 1 亿 → "X.XX亿"；0.01~1 亿 → 万级整数（如 0.59 亿 → "5900万"）；
    < 0.01 亿 → 万级（数值 < 100，如 "50万"）。
    关键约束：1.89 亿必须渲染为 "1.89亿"，不得进位为 "2亿"——因此保留
    digits 位小数而非取整。
    """
    if abs(x_yi) >= 1:
        return f"{x_yi:.{digits}f}亿"
    return f"{x_yi * 1e4:.0f}万"


def fmt_pct(x: float, digits: int = 1, with_fraction: bool = True) -> str:
    """百分比格式化。

    with_fraction=True（默认）：入参为小数，0.176 → "17.6%"。
    with_fraction=False：入参已是百分数（如 17.6），直接格式化 → "17.6%"，
    不再乘以 100。调用方必须分清两种入参口径，避免 "1760%" 类错误。
    """
    v = x * 100 if with_fraction else x
    return f"{v:.{digits}f}%"


def fmt_price(x: float) -> str:
    """价格格式化：固定保留 2 位小数。"""
    return f"{x:.2f}"


# ============================================================
#  报告数字回溯验证（FIX-04）
# ============================================================

# 指标名 → 标签锚定正则列表。
# 设计原则：只在"指标标签"附近抓数字，不盲抓全文——报告中存在大量非计算
# 数字（年份、评分、代码等），盲抓必然误报。
# 注意："隐含增速"的注册值以百分数存储（102.0 表示 102%），与文本中
# "%" 前的数值直接比较；其余指标注册值与文本值同单位。
_LABEL_PATTERNS: dict[str, list[str]] = {
    "市值": [r"市值[：:]?\s*(\d+\.?\d*)\s*亿"],
    "PE": [r"PE\(TTM\)[：:]?\s*(\d+\.?\d*)",
           r"PE[：:]\s*(\d+\.?\d*)"],
    "PB": [r"PB[：:]?\s*(\d+\.?\d*)"],
    "合理价值": [r"合理价值[：:]?\s*\**(\d+\.?\d*)"],
    "安全买入价": [r"安全买入价[：:]?\s*≤?\s*\**(\d+\.?\d*)"],
    "隐含增速": [r"(?:年复合增速|年化增速|复合增速|年复合增长率)[^0-9]{0,12}(\d+\.?\d*)\s*%"],
    "概率加权价值": [r"概率加权价值[：:]?\s*\**(\d+\.?\d*)"],
}

def _verify_cfg() -> dict:
    """回溯验证容差（configs/scoring.json [估值守卫.回溯验证]，禁止硬编码）。"""
    from backend.scoring_config import get_valuation_guards
    return get_valuation_guards()["verify"]


def _label_tol() -> dict:
    """分指标容差：估值锚点类指标在 LLM 叙述/框架分歧段中常按整数取整
    （29.2 → "合理价值30元"），四舍五入不构成矛盾，用[锚点容差]；
    PE/PB/市值保持[默认容差]（B2/B3 类错误均为 20%+ 偏差，不受影响）。"""
    anchor = _verify_cfg()["锚点容差"]
    return {"合理价值": anchor, "安全买入价": anchor, "概率加权价值": anchor, "隐含增速": anchor}


def _rel_dev(actual: float, expected: float) -> float:
    """相对偏差 |a-b|/|b|；expected 为 0 时退化为绝对差（避免除零）。"""
    if expected == 0:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


def verify_report_text(text: str, calc_table: CalcTable, tol: float = None) -> list[str]:
    """报告数字回溯验证（FIX-04 核心）。

    对 calc_table 中每个已注册且在 _LABEL_PATTERNS 中有锚定正则的指标：
      1. 在 text 中找出该指标标签附近的所有数值出现处；
      2. 每个出现值与注册值做相对偏差比对，> tol → 违规；
      3. 同一指标多处出现的值做两两一致性检查（> tol → 违规），
         同时覆盖"同一指标全文唯一值"约束。

    text 中找不到某指标 → 不算违规（该指标可能未被渲染）；
    无锚定正则的注册指标 → 跳过，无法回溯。

    返回违规描述字符串列表，每条含指标名/期望值/实际值/偏差。
    """
    if tol is None:
        tol = _verify_cfg()["默认容差"]
    _ltol = _label_tol()
    violations: list[str] = []
    for name in calc_table.names():
        patterns = _LABEL_PATTERNS.get(name)
        if not patterns:
            continue
        entry = calc_table.get(name)
        tol_i = _ltol.get(name, tol)  # 分指标容差（叙述取整放宽）
        occurrences: list[float] = []
        for pat in patterns:
            for m in re.findall(pat, text):
                try:
                    occurrences.append(float(m))
                except (ValueError, TypeError):
                    continue
        if not occurrences:
            continue

        # 出现值 vs 注册值
        for occ in occurrences:
            dev = _rel_dev(occ, entry.value)
            if dev > tol_i:
                violations.append(
                    f"[{name}] 文本值 {occ} 与登记值 {entry.value:.2f} "
                    f"偏差 {dev:.1%} 超过容差 {tol_i:.0%}"
                )

        # 同一指标多处出现值两两一致性
        for i in range(len(occurrences)):
            for j in range(i + 1, len(occurrences)):
                dev = _rel_dev(occurrences[i], occurrences[j])
                if dev > tol_i:
                    violations.append(
                        f"[{name}] 全文多处出现值不一致："
                        f"{occurrences[i]} vs {occurrences[j]}，偏差 {dev:.1%}"
                    )
    return violations


# ============================================================
#  跨源字段比对（FIX-12）
# ============================================================

def cross_validate_fields(primary: dict, secondary: dict, tol: float = None) -> list[str]:
    """对两个字典的共同键做相对偏差比对（FIX-12）。

    相对偏差 = |a-b| / max(|a|,|b|)，> tol → 返回差异描述（键名/两值/偏差）。
    None 值跳过（缺数据不构成矛盾）；0 值跳过（0 作分母无意义）；
    非数值字段跳过（本函数只做数值比对）。
    """
    if tol is None:
        tol = _verify_cfg()["跨源容差"]
    diffs: list[str] = []
    for key in primary.keys() & secondary.keys():
        v1, v2 = primary[key], secondary[key]
        if v1 is None or v2 is None:
            continue
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        if v1 == 0 or v2 == 0:
            continue
        dev = abs(v1 - v2) / max(abs(v1), abs(v2))
        if dev > tol:
            diffs.append(
                f"[{key}] 跨源不一致：{v1:.2f} vs {v2:.2f}，"
                f"偏差 {dev:.1%} 超过容差 {tol:.0%}"
            )
    return diffs
