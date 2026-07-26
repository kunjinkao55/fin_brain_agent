"""跨模块一致性不变量校验器（FIX-05）+ 指标生命周期注册表（FIX-07）。

针对人工核验发现的真实报告矛盾（合理区间高于合理价值、SELL 与买入建议并存、
情景增速与文字假设不匹配、已禁用指标仍被引用、同一指标多处数值打架、
情绪描述与市场广度背离），提供 6 条不变量检查（INV-1 ~ INV-6），
优先从 item 结构化字段取证，report_text 用于全文扫描类检查。

指标生命周期注册表（FIX-07）：指标可因数据失真（如次新股前瞻 PE 季节性失真）
被标记为 disabled，后续校验（INV-4）可拦截报告中对该指标的继续引用。

本模块不 import backend.agent，可被任意模块安全引入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


def _breakers() -> dict:
    """一致性断路器阈值（configs/scoring.json [估值守卫.一致性断路器]，禁止硬编码）。"""
    from backend.scoring_config import get_valuation_guards
    return get_valuation_guards()["breakers"]


def _grounding_tol() -> float:
    """grounding 校验取整容差（[估值守卫.回溯验证.grounding容差]）。"""
    from backend.scoring_config import get_valuation_guards
    return get_valuation_guards()["verify"]["grounding容差"]


@dataclass
class Issue:
    """一条一致性违例。"""

    rule: str        # 规则编号 "INV-1"~"INV-6"
    severity: str    # "blocker" | "warning"
    detail: str      # 人类可读描述


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _num(s) -> float | None:
    """宽松数字解析：去逗号/元/百分号，失败返回 None。"""
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    m = re.search(r'-?[\d,]+(?:\.\d+)?', s.replace(',', ''))
    try:
        return float(m.group(0)) if m else None
    except (ValueError, TypeError):
        return None


def _safe_buy_price(item: dict) -> float | None:
    """从 投资评级.买入区间（形如 "≤20.9元"）解析安全买入价。"""
    rating = item.get("投资评级", {})
    if not isinstance(rating, dict):
        return None
    return _num(rating.get("买入区间", ""))


def _fair_value(item: dict) -> float | None:
    rating = item.get("投资评级", {})
    if not isinstance(rating, dict):
        return None
    return _num(rating.get("合理价值"))


def _texts(item: dict) -> str:
    """操作建议 + 综合结论 拼接文本。"""
    return "\n".join(str(item.get(k, "") or "") for k in ("操作建议", "综合结论"))


# ---------------------------------------------------------------------------
# INV-1 ~ INV-6
# ---------------------------------------------------------------------------

def _inv1(item: dict, issues: list[Issue]) -> None:
    """合理区间 ⊆ [安全买入价×0.98, 合理价值×1.1]。"""
    safe = _safe_buy_price(item)
    fair = _fair_value(item)
    if safe is None and fair is None:
        return
    text = _texts(item)
    # 两种语序："合理区间(约)30-60元" 与 "30-60元合理区间"；兼容全角括号与加粗标记
    patterns = [
        r'合理区间[（(]?\*{0,2}[约为：:\s]*([\d.]+)\s*[-~—至–]\s*([\d.]+)\s*元',
        r'([\d.]+)\s*[-~—至–]\s*([\d.]+)\s*元[^。；\n]{0,6}?合理区间',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            _b = _breakers()
            if safe is not None and lo < safe * (1 - _b["INV1下限容差"]):
                issues.append(Issue(
                    "INV-1", "blocker",
                    f"合理区间下限{lo}元低于安全买入价{safe}元（容差{_b['INV1下限容差']*100:.0f}%），区间与安全边际口径冲突"))
                return
            if fair is not None and hi > fair * _b["INV1上限倍数"]:
                issues.append(Issue(
                    "INV-1", "blocker",
                    f"合理区间上限{hi}元高于合理价值{fair}元的{_b['INV1上限倍数']*100:.0f}%（{fair * _b['INV1上限倍数']:.2f}元），区间自相矛盾"))
                return


def _inv2(item: dict, issues: list[Issue]) -> None:
    """评级 SELL/AVOID 时，买入语境价格不得高于安全买入价。"""
    rating = item.get("投资评级", {})
    level = str(rating.get("评级", "")).upper() if isinstance(rating, dict) else ""
    if level not in ("SELL", "AVOID"):
        return
    safe = _safe_buy_price(item)
    if safe is None:
        return
    text = _texts(item)
    # 明确的买入动词；"可/再" 等弱词不算（"再考虑"并非买入建议）
    pat = r'(\d+\.?\d*)\s*元(?:以下|之下)?[^。；\n]{0,12}(?:买入|建仓|轻仓|试探|抄底|参与|加仓)'
    # 否定语境排除："当前174元严重高估，不建议买入" 不是买入建议
    neg = re.compile(r'(?:不|勿|暂不|禁止|避免|未|别)[^。；\n]{0,4}(?:买入|建仓|参与|加仓|抄底)')
    for m in re.finditer(pat, text):
        if neg.search(m.group(0)):
            continue
        price = float(m.group(1))
        if price > safe:
            issues.append(Issue(
                "INV-2", "blocker",
                f"评级{level}但文本含买入建议价位{price}元，高于安全买入价{safe}元，决策与建议矛盾"))
            return


# 假设文本增速描述 → 小数增速
_GROWTH_PATTERNS = [
    (r'(?:下[滑降跌]|回落|减少)\s*(?:约|近)?\s*([\d.]+)\s*%', -1),
    (r'(?:增长|上升|提升|增加|上涨)\s*(?:约|近)?\s*([\d.]+)\s*%', 1),
]
_FLAT_WORDS = ("持平", "维持", "基本稳定", "不增不减", "保持不变")


def _assumption_growth(text: str) -> float | None:
    """从情景假设文本解析增速描述（小数），无法识别返回 None。"""
    for pat, sign in _GROWTH_PATTERNS:
        m = re.search(pat, text)
        if m:
            return sign * float(m.group(1)) / 100.0
    if any(w in text for w in _FLAT_WORDS):
        return 0.0
    return None


def _inv3(item: dict, issues: list[Issue]) -> None:
    """情景估值：价格单调性（blocker）+ 假设增速与 EPS 隐含增速对齐（warning）。"""
    sc = item.get("情景估值", {})
    if not isinstance(sc, dict):
        return
    names = ["悲观", "基准", "乐观"]
    prices = []
    for n in names:
        s = sc.get(n, {})
        prices.append(_num(s.get("价格")) if isinstance(s, dict) else None)
    if all(p is not None for p in prices):
        if not (prices[0] <= prices[1] <= prices[2]):
            issues.append(Issue(
                "INV-3", "blocker",
                f"情景价格倒挂：悲观{prices[0]} / 基准{prices[1]} / 乐观{prices[2]}，"
                f"应满足 悲观≤基准≤乐观"))

    # 假设文本增速 vs EPS 隐含增速（相对 TTM EPS），偏差>5pp 视为不一致
    eps_ttm = _num(item.get("_eps_ttm"))
    if not eps_ttm or eps_ttm <= 0:
        return  # 取不到 TTM EPS 则跳过增速对齐检查
    for n in names:
        s = sc.get(n, {})
        if not isinstance(s, dict):
            continue
        eps = _num(s.get("EPS"))
        text_growth = _assumption_growth(str(s.get("假设", "") or ""))
        if eps is None or text_growth is None:
            continue
        implied = eps / eps_ttm - 1.0
        _tol3 = _breakers()["INV3增速偏差pp"]
        if abs(implied - text_growth) > _tol3 - 1e-6:
            issues.append(Issue(
                "INV-3", "warning",
                f"{n}情景假设文本增速{text_growth:+.0%}与EPS隐含增速{implied:+.1%}"
                f"（EPS {eps} vs TTM {eps_ttm}）偏差超5pp"))


def _inv4(item: dict, report_text: str, issues: list[Issue]) -> None:
    """已禁用指标不得在报告中带数值出现。"""
    if not report_text:
        return
    for metric in disabled_metrics(item):
        pat = re.escape(metric) + r'[^\n]{0,20}?[\d,.]+\s*倍'
        m = re.search(pat, report_text)
        if m:
            issues.append(Issue(
                "INV-4", "blocker",
                f"指标「{metric}」已禁用（{disabled_metrics(item)[metric]}），"
                f"但报告中仍出现数值引用：…{m.group(0)}…"))


# 全文唯一值检查的标签（同标签多处数值须一致）
# 注意：不含"经营现金流"——Q1(-0.8亿)与年报(0.59亿)是不同期次的合法并存值，
# 标签级比对无法区分期次，必然误报（现金流类矛盾由跨源比对/FCF预警覆盖）。
_UNIQUE_LABELS = {
    "市值": r'市值[^。\n]{0,15}?([\d,]+(?:\.\d+)?)\s*亿',
    "归母净利": r'归母净利[润]?[^。\n]{0,15}?([\d,]+(?:\.\d+)?)\s*亿',
}


def _inv5(item: dict, report_text: str, issues: list[Issue]) -> None:
    """同一指标全文唯一值：同标签多处出现值两两偏差>1% → blocker。"""
    if not report_text:
        return  # 调用方不传 report_text 时跳过全文类检查
    # item 估值水位中的市值作为基准值之一参与比对
    level = item.get("估值水位", {})
    for label, pat in _UNIQUE_LABELS.items():
        values: list[float] = []
        if isinstance(level, dict) and label == "市值":
            base = _num(str(level.get("市值", "")))
            if base is not None:
                values.append(base)
        for m in re.finditer(pat, report_text):
            try:
                values.append(float(m.group(1).replace(',', '')))
            except (ValueError, TypeError):
                continue
        bad = False
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                denom = max(abs(a), abs(b))
                if denom > 0 and abs(a - b) / denom > 0.01:
                    bad = True
        if bad:
            uniq = sorted({round(v, 4) for v in values})
            issues.append(Issue(
                "INV-5", "blocker",
                f"「{label}」全文取值不唯一：{uniq}（亿元口径），两两偏差超1%"))


def _inv6(item: dict, issues: list[Issue]) -> None:
    """情绪一致性：上涨比例<30% 时综合情绪不得为偏暖/乐观/高涨。"""
    sent = item.get("市场情绪", {})
    if not isinstance(sent, dict):
        return
    # 市场广度兼容 str（"上涨比例 8.5%"）与 dict（{"上涨比例": "8.5%"}）两种形态
    breadth = sent.get("市场广度", "")
    if isinstance(breadth, dict):
        breadth_text = str(breadth.get("上涨比例", ""))
    else:
        breadth_text = str(breadth)
    m = re.search(r'([\d.]+)\s*%', breadth_text)
    if not m:
        return
    up_ratio = float(m.group(1))
    mood_text = str(sent.get("综合情绪", "")) + str(sent.get("情绪标签", ""))
    if up_ratio < _breakers()["INV6广度阈值"] and any(w in mood_text for w in ("偏暖", "乐观", "高涨")):
        issues.append(Issue(
            "INV-6", "warning",
            f"市场广度上涨比例仅{up_ratio}%，与综合情绪「{mood_text}」的偏暖表述背离"))


# ---------------------------------------------------------------------------
# INV-7 ~ INV-10（支柱五：逻辑一致性断路器）
# ---------------------------------------------------------------------------

def _inv7(item: dict, issues: list[Issue]) -> None:
    """评级-增速一致性：SELL 但营收增速>30% → 高增长转型期静态PE卖出可能犯错。"""
    rating = item.get("投资评级", {})
    level = str(rating.get("评级", "")).upper() if isinstance(rating, dict) else ""
    if level not in ("SELL", "AVOID"):
        return
    g = item.get("_rev_growth")
    if g is None:
        return
    try:
        g = float(g)
    except (TypeError, ValueError):
        return
    _th7 = _breakers()["INV7增速阈值"]
    if g > _th7:
        issues.append(Issue(
            "INV-7", "warning",
            f"评级{level}但营收增速{g:.0f}%>{_th7:.0f}%：高增长转型期使用静态PE卖出可能犯下致命错误，"
            "建议人工复核（若为第二曲线/新业务驱动，应先做SOTP拆分再定评级）"))


def _inv9(item: dict, issues: list[Issue]) -> None:
    """估值-现金流匹配：合理价值隐含市值 > 10×经营现金流 且 ROIC<15% → 依赖再投资假设。"""
    fair = _fair_value(item)
    ctx = item.get("_share_ctx", {})
    shares = ctx.get("总股本") if isinstance(ctx, dict) else None
    cfo = item.get("_cfo_annual")
    roic = item.get("_roic_proxy")
    if not fair or not shares or not cfo or roic is None:
        return
    try:
        shares = float(shares)
        cfo = float(cfo)
        roic = float(roic)
    except (TypeError, ValueError):
        return
    if cfo <= 0 or shares <= 0:
        return
    _b9 = _breakers()
    implied_mcap = fair * shares
    if implied_mcap > _b9["INV9现金流倍数"] * cfo and roic < _b9["INV9_ROIC阈值"]:
        issues.append(Issue(
            "INV-9", "warning",
            f"合理价值隐含市值{implied_mcap/1e8:.0f}亿 > {_b9['INV9现金流倍数']:.0f}×经营现金流({cfo/1e8:.2f}亿)，"
            f"且ROIC仅{roic*100:.1f}%：估值严重依赖再投资假设，资本回报下降时价值将大幅缩水"))


def _inv10(item: dict, issues: list[Issue]) -> None:
    """历史锚定陷阱：周期属性强（cyclical>0.4）且合理价值基于 TTM EPS → 顶部/底部失真。"""
    try:
        cyc = float(item.get("_cyclical", 0) or 0)
    except (TypeError, ValueError):
        return
    if cyc > _breakers()["INV10周期阈值"] and _fair_value(item):
        issues.append(Issue(
            "INV-10", "warning",
            f"周期属性({cyc*100:.0f}%)公司以TTM EPS锚定合理价值："
            "周期顶部/底部利润失真将直接传导至估值，建议结合前瞻利润与PB底部复核"))


def check_invariants(item: dict, report_text: str = "") -> list[Issue]:
    """对报告 item 执行 10 条跨模块一致性不变量检查（INV-1~6 + 支柱五断路器 INV-7~10）。

    尽量从 item 结构化字段取证；report_text 用于全文扫描类检查
    （INV-4 禁用指标引用、INV-5 全文唯一值），不传则跳过这两类。
    """
    issues: list[Issue] = []
    if not isinstance(item, dict):
        return issues
    for fn in (_inv1, _inv2, _inv3, _inv6, _inv7, _inv9, _inv10):
        try:
            fn(item, issues)
        except Exception:
            pass
    for fn in (_inv4, _inv5):
        try:
            fn(item, report_text, issues)
        except Exception:
            pass
    return issues


# ---------------------------------------------------------------------------
# LLM 输出 grounding 校验（Critic/Repair 幻觉拦截）
# ---------------------------------------------------------------------------

# 硬数字：数字紧邻单位（元/亿/万/%/倍）。年份(19xx/20xx)与序号类小数字不构成事实主张，不抓
_HARD_NUM_PAT = re.compile(r'(-?\d+\.?\d*)\s*(亿|万|元|%|倍)')
_YEAR_PAT = re.compile(r'^(19|20)\d{2}$')
# 裸数字池（辅助）：报告中常写 "PE:337"（无"倍"字），若股评写"337倍"，
# 仅按单位池匹配会误杀。裸数字要求非小数点/逗号/百分号邻接，降低误配。
_BARE_NUM_PAT = re.compile(r'(?<![\d.,(（])(-?\d+\.?\d*)(?![\d.,%）)])')


def extract_hard_numbers(text: str) -> list[tuple[float, str]]:
    """从文本提取带单位的事实型数字：[(归一值, 单位类)]。
    金额统一归一到"亿"（万→/1e4）；元/倍/% 保持原值。"""
    out: list[tuple[float, str]] = []
    if not text:
        return out
    for m in _HARD_NUM_PAT.finditer(str(text)):
        raw = m.group(1)
        if _YEAR_PAT.match(raw):
            continue
        v = float(raw)
        unit = m.group(2)
        if unit == "万":
            out.append((v / 1e4, "亿"))
        else:
            out.append((v, unit))
    return out


def _extract_bare_numbers(text: str) -> list[float]:
    """提取全部裸数字（不限单位），作为 traceability 的辅助池。"""
    out: list[float] = []
    if not text:
        return out
    for m in _BARE_NUM_PAT.finditer(str(text)):
        raw = m.group(1)
        if _YEAR_PAT.match(raw):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _traceable(value: float, unit: str, pool: list[tuple[float, str]],
               bare_pool: list[float], tol: float) -> bool:
    """value 是否可溯源：先按单位类匹配 pool；失败则按裸数字匹配 bare_pool。
    0 值宽松放过。"""
    if value == 0:
        return True
    for pv, pu in pool:
        if pu != unit:
            continue
        denom = max(abs(value), abs(pv))
        if denom > 0 and abs(value - pv) / denom <= tol:
            return True
    for bv in bare_pool:
        denom = max(abs(value), abs(bv))
        if denom > 0 and abs(value - bv) / denom <= tol:
            return True
    return False


def _make_pools(source_text: str) -> tuple[list[tuple[float, str]], list[float]]:
    return extract_hard_numbers(source_text), _extract_bare_numbers(source_text)


def filter_untraceable_issues(
    issues: list[str],
    source_text: str,
    tol: float = None,
) -> tuple[list[str], list[str]]:
    """Critic 发现的 grounding 过滤：发现中的每个硬数字都必须能在 source_text
    （原始分析+采集数据）找到出处（±tol 取整误差）。

    - 发现不含硬数字 → 保留（无法校验，存疑从宽）
    - 全部硬数字可溯源 → 保留
    - 含 ≥1 个不可溯源硬数字 → 判定疑似幻觉（过时数据/编造），丢弃

    :return: (保留的发现, 丢弃的发现)
    """
    if tol is None:
        tol = _grounding_tol()
    pool, bare = _make_pools(source_text)
    kept: list[str] = []
    dropped: list[str] = []
    for issue in issues:
        if not issue:
            continue
        nums = extract_hard_numbers(str(issue))
        if not nums:
            kept.append(issue)
            continue
        bad = [f"{v}{u}" for v, u in nums if not _traceable(v, u, pool, bare, tol)]
        if bad:
            dropped.append(issue)
        else:
            kept.append(issue)
    return kept, dropped


def has_new_hard_numbers(
    original_text: str,
    new_text: str,
    tol: float = None,
) -> list[tuple[float, str]]:
    """Repair 输出校验：new_text 中引入的、在 original_text（原文+采集数据池）
    中不存在的新硬数字列表。空列表 = 通过。"""
    if tol is None:
        tol = _grounding_tol()
    pool, bare = _make_pools(original_text)
    return [(v, u) for v, u in extract_hard_numbers(new_text)
            if not _traceable(v, u, pool, bare, tol)]


_SENT_SPLIT = re.compile(r'(?<=[。！？；])|\n')


def strip_untraceable_sentences(
    text: str,
    source_text: str,
    tol: float = None,
    keep_patterns: tuple = ("不构成投资建议", "股市有风险"),
) -> tuple[str, list[str]]:
    """股评/长文幻觉拦截（句子级）：逐句检查，含不可溯源硬数字的句子整句剔除。

    数字可溯源判定与 has_new_hard_numbers 同池同容差。
    keep_patterns: 命中这些子串的句子永远保留（免责声明等）。
    返回 (清洗后文本, 被剔除句子列表)。全部剔除时返回空串。"""
    if not text:
        return text, []
    if tol is None:
        tol = _grounding_tol()
    pool, bare = _make_pools(source_text)
    kept: list[str] = []
    removed: list[str] = []
    for sent in _SENT_SPLIT.split(str(text)):
        s = sent.strip()
        if not s:
            continue
        if any(k in s for k in keep_patterns):
            kept.append(s)
            continue
        nums = extract_hard_numbers(s)
        bad = [(v, u) for v, u in nums if not _traceable(v, u, pool, bare, tol)]
        if bad:
            removed.append(s)
        else:
            kept.append(s)
    return "\n".join(kept), removed


# ---------------------------------------------------------------------------
# 指标生命周期注册表（FIX-07）
# ---------------------------------------------------------------------------

def disable_metric(item: dict, metric: str, reason: str) -> None:
    """将指标标记为禁用（数据失真/不可用），记录原因与禁用时间。"""
    item.setdefault("_metric_status", {})[metric] = {
        "status": "disabled",
        "reason": reason,
        "disabled_at": datetime.now().isoformat(timespec="seconds"),
    }


def enable_metric(item: dict, metric: str) -> None:
    """解除指标禁用状态。"""
    item.setdefault("_metric_status", {})[metric] = {
        "status": "active",
        "enabled_at": datetime.now().isoformat(timespec="seconds"),
    }


def is_active(item: dict, metric: str) -> bool:
    """指标是否可用：未注册或 status=active → True。"""
    st = item.get("_metric_status", {})
    if not isinstance(st, dict):
        return True
    info = st.get(metric)
    if not isinstance(info, dict):
        return True
    return info.get("status") == "active"


def disabled_metrics(item: dict) -> dict:
    """返回 {metric: reason} 的禁用指标清单。"""
    st = item.get("_metric_status", {})
    if not isinstance(st, dict):
        return {}
    return {m: info.get("reason", "") for m, info in st.items()
            if isinstance(info, dict) and info.get("status") == "disabled"}
