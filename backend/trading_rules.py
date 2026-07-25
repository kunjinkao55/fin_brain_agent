"""
FinBrain 涨跌停规则引擎（FIX-06）

背景：旧代码硬编码 9.9%/19.9% 阈值（backend/tools.py:1297-1303），不区分
板块、新股上市初期与 ST 股，导致误判——托伦斯（301583，创业板新股）上市
前 5 个交易日无涨跌幅限制，首日 +858.8% 被误标为"涨停"。

本模块按交易所规则判定：
    - 沪/深主板：上市首日上限 44%，常规 10%
    - 创业板/科创板：上市前 5 个交易日无涨跌幅限制，之后 20%
    - 北交所：上市首日无涨跌幅限制，之后 30%
    - ST 股：5%（简化口径，见 limit_pct docstring）

无交易日历可用，listing_days_since 用自然日近似换算交易日（误差 ±2 个
交易日），仅供涨跌停规则粗判，不用于精确交易日计算。

仅使用标准库。

用法：
    from backend.trading_rules import is_limit_up
    is_limit_up("301583", "托伦斯", 858.8, "2026-07-10", "2026-07-10")  # False
    is_limit_up("301583", "托伦斯", 20.0, "2026-07-10", "2026-07-24")   # True
"""

import math
from datetime import date
from typing import Optional


def board_of(symbol: str) -> str:
    """按代码前缀判断上市板块。

    "60" → 沪主板；"68" → 科创板；"00" → 深主板；"30" → 创业板；
    "8"/"4"/"92" 开头 → 北交所；其他 → "未知"。
    兼容带交易所后缀的写法（如 "301583.SZ"），只取点号前的数字前缀判断。
    """
    code = (symbol or "").strip().split(".")[0]
    if code.startswith("60"):
        return "沪主板"
    if code.startswith("68"):
        return "科创板"
    if code.startswith("00"):
        return "深主板"
    if code.startswith("30"):
        return "创业板"
    if code.startswith(("8", "4", "92")):
        return "北交所"
    return "未知"


def limit_pct(board: str, listing_days: Optional[int] = None,
              st: bool = False) -> Optional[float]:
    """返回涨跌幅限制百分比；None 表示无涨跌幅限制。

    规则：
        st=True → 5.0（简化口径：各板块一致按 5% 处理；实际上创业板 ST 股
            为 20%、科创板无 ST 制度，此处取最保守默认值，后续可按板块细化）
        创业板/科创板：listing_days <= 5（含未知 None 除外的情形）→ None，
            即上市前 5 个交易日无涨跌幅限制；否则 20.0
        北交所：listing_days == 1 → None（首日无限制）；否则 30.0
        沪/深主板：listing_days == 1 → 44.0（新股首日上限）；否则 10.0
        未知板块 → 10.0（保守默认）

    listing_days 为 None 表示上市天数未知，此时不按新股处理，直接返回
    常规限制（避免把老股误判为无限制）。
    """
    if st:
        return 5.0
    if board in ("创业板", "科创板"):
        if listing_days is not None and listing_days <= 5:
            return None  # 上市前 5 个交易日无涨跌幅限制
        return 20.0
    if board == "北交所":
        if listing_days == 1:
            return None  # 上市首日无涨跌幅限制
        return 30.0
    if board in ("沪主板", "深主板"):
        if listing_days == 1:
            return 44.0  # 主板新股首日涨幅上限 44%
        return 10.0
    return 10.0  # 未知板块保守默认


def listing_days_since(listing_date: str, ref_date: Optional[str] = None) -> Optional[int]:
    """自上市日起的第 N 个交易日（上市当日 = 1）。

    listing_date / ref_date 格式 "YYYY-MM-DD"；ref_date 缺省 = 今天。
    无交易日历可用，用自然日近似换算：交易日数 ≈ 1 + ceil(自然日差 × 5/7)。
    自然日近似，误差 ±2 个交易日（遇长假偏差更大），仅供规则粗判。

    日期解析失败、或 ref_date 早于 listing_date → None。
    """
    try:
        d0 = date.fromisoformat(listing_date)
        d1 = date.fromisoformat(ref_date) if ref_date else date.today()
    except (ValueError, TypeError):
        return None
    delta = (d1 - d0).days
    if delta < 0:
        return None
    return 1 + math.ceil(delta * 5 / 7)


def _is_limit(symbol: str, name: str, pct_chg: float,
              listing_date: Optional[str], ref_date: Optional[str],
              tol: float, direction: int) -> bool:
    """涨/跌停判定公共逻辑。direction=1 涨停，-1 跌停。"""
    st = "ST" in (name or "").upper()  # 覆盖 "ST"、"*ST"、"ST股" 等命名
    board = board_of(symbol)
    days = listing_days_since(listing_date, ref_date) if listing_date else None
    limit = limit_pct(board, days, st)
    if limit is None:
        return False  # 无涨跌幅限制，不存在涨停/跌停
    return abs(pct_chg - limit * direction) <= tol


def is_limit_up(symbol: str, name: str, pct_chg: float,
                listing_date: Optional[str] = None,
                ref_date: Optional[str] = None, tol: float = 0.2) -> bool:
    """是否涨停：|pct_chg - limit| <= tol。无涨跌幅限制 → 必然 False。

    name 含 "ST" → 按 ST 规则；listing_date 提供时换算上市天数判定新股。

    托伦斯验证：
        is_limit_up("301583", "托伦斯", 858.8, "2026-07-10", "2026-07-10") → False
        （创业板上市首日无涨跌幅限制，+858.8% 不是涨停）
        is_limit_up("301583", "托伦斯", 20.0, "2026-07-10", "2026-07-24") → True
        （上市第 11 个交易日，适用 20% 限制）
    """
    return _is_limit(symbol, name, pct_chg, listing_date, ref_date, tol, direction=1)


def is_limit_down(symbol: str, name: str, pct_chg: float,
                  listing_date: Optional[str] = None,
                  ref_date: Optional[str] = None, tol: float = 0.2) -> bool:
    """是否跌停：|pct_chg + limit| <= tol。无涨跌幅限制 → 必然 False。"""
    return _is_limit(symbol, name, pct_chg, listing_date, ref_date, tol, direction=-1)
