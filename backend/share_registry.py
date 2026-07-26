"""股本单一事实源（SSOT）与估值口径守卫。

对应修复任务：
- FIX-01：总股本目前被 5+ 处独立读取（agent.py / tools.py 各自解析
  ``get_valuation()`` 的 ``总股本`` 字段），口径易漂移。本模块提供
  :func:`resolve_share_context` 作为唯一取数入口，并带进程内日期缓存，
  保证一次审计（含重试）内数值不漂移。
- FIX-02：IPO 后 BPS 不做募集净额修正会导致 PB 严重失真（托伦斯案例：
  用 IPO 前 BPS 6.40 算 PB=27.2，正确应为发行后 BPS 9.87 → PB≈17.6）。
  :func:`adjust_bps_for_event` 提供发行后口径（含募集净额）的 BPS 调整，
  :attr:`ShareContext.effective_bps` 统一对外暴露"应使用的 BPS"。

约束：不得 import backend.agent（避免循环依赖）；仅依赖标准库。
``CorporateEvent`` 仅在 TYPE_CHECKING 下引用，运行时不产生模块依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # 仅类型标注用，避免运行时循环导入
    from backend.corporate_actions import CorporateEvent

# 总股本字段血缘（东财 F10 主要指标接口）
_SOURCE_TOTAL_SHARE = "eastmoney:RPT_F10_FINANCE_MAINFINADATA.TOTAL_SHARE"

# 进程内缓存：key = (symbol, 日期串)。保证同一交易日内审计重试间数值不漂移。
_CTX_CACHE: Dict[Tuple[str, str], "ShareContext"] = {}


def clear_cache() -> None:
    """清空 ShareContext 进程内缓存（跨交易日或强制刷新时调用）。"""
    _CTX_CACHE.clear()


@dataclass
class ShareContext:
    """股本上下文：总股本 / BPS / 股本变动事件的单一事实源。"""

    ticker: str
    total_shares: float = 0.0              # 单位：股
    float_shares: Optional[float] = None   # 流通股本，未知则 None
    as_of_date: str = ""                   # 取数期次，如 "2025-12-31 [年报]"
    source: str = _SOURCE_TOTAL_SHARE      # 血缘
    events: List[Any] = field(default_factory=list)  # List[CorporateEvent]
    bps: float = 0.0                       # 财报口径每股净资产（元）
    bps_adjusted: Optional[float] = None   # 股本变动事件后调整口径（元）
    bps_basis: str = "财报口径"            # BPS 口径说明

    @property
    def effective_bps(self) -> float:
        """对外统一使用的 BPS：调整后口径优先，缺失时回退财报口径。"""
        if self.bps_adjusted is not None and self.bps_adjusted > 0:
            return self.bps_adjusted
        return self.bps

    @property
    def has_recent_capital_event(self) -> bool:
        """近期是否存在股本变动事件（IPO/定增/配股等）。"""
        return bool(self.events)


def _to_float(value: Any) -> float:
    """宽松数值转换：兼容 None / 千分位逗号字符串，失败返回 0.0。"""
    if value is None:
        return 0.0
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if not value:
                return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _period_label(row: Dict[str, Any]) -> str:
    """从利润表行提取标准期间标签（年报/一季报/半年报/三季报）。

    兼容两种 ``报告期`` 格式：
    - 短标签（tools.py 利润表实际格式）："年报" / "一季报" / ...
    - 带日期描述（如快报/估值接口）："2026-03-31 [一季报]"
    """
    raw = str(row.get("报告期", "") or "")
    m = re.search(r"(半年报|三季报|一季报|一季度|年报)", raw)
    if not m:
        return raw
    label = m.group(1)
    return "一季报" if label == "一季度" else label


def resolve_share_context(
    symbol: str,
    val_data: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Any]] = None,
    bps_adjusted: Optional[float] = None,
    bps_basis: Optional[str] = None,
    float_shares: Optional[float] = None,
) -> ShareContext:
    """解析股本上下文（SSOT 唯一入口）。

    :param symbol: 股票代码，如 "301583"。
    :param val_data: ``get_valuation()`` 返回的列表（按报告期倒序），
        取第一行的 ``总股本`` / ``每股净资产`` / ``报告期``。
    :param events: CorporateEvent 列表（来自 corporate_actions.classify_events）。
    :param bps_adjusted: 股本变动事件后调整口径 BPS（通常由
        :func:`adjust_bps_for_event` 算出）。
    :param bps_basis: BPS 口径说明，缺省时按是否提供 bps_adjusted 自动选择。
    :param float_shares: 流通股本（股），未知传 None。

    注意：``total_shares <= 0`` 时也正常返回 ShareContext（total_shares=0），
    由调用方自行降级，本函数不抛异常。无显式输入且当日缓存命中时直接返回缓存，
    保证审计重试间数值不漂移。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    key = (symbol, today)
    # 无任何显式输入 → 优先命中当日缓存
    if val_data is None and events is None and bps_adjusted is None and float_shares is None:
        cached = _CTX_CACHE.get(key)
        if cached is not None:
            return cached

    total_shares = 0.0
    bps = 0.0
    as_of = ""
    if val_data:
        first = val_data[0]
        if isinstance(first, dict):
            total_shares = _to_float(first.get("总股本"))
            bps = _to_float(first.get("每股净资产"))
            as_of = str(first.get("报告期") or first.get("日期") or first.get("date") or "")
    if total_shares < 0:
        total_shares = 0.0

    if bps_adjusted is not None:
        basis = bps_basis or "发行后口径(含募集净额)"
    else:
        basis = bps_basis or "财报口径"

    ctx = ShareContext(
        ticker=symbol,
        total_shares=total_shares,
        float_shares=float_shares,
        as_of_date=as_of,
        source=_SOURCE_TOTAL_SHARE,
        events=list(events) if events else [],
        bps=bps,
        bps_adjusted=bps_adjusted,
        bps_basis=basis,
    )
    _CTX_CACHE[key] = ctx
    return ctx


def compute_ttm_eps(
    profit_data: Optional[List[Dict[str, Any]]],
    total_shares: float,
) -> Tuple[float, float]:
    """计算 TTM 每股收益，返回 (eps_ttm, ttm_net_profit)。

    语义与 backend/agent.py:2206-2228 完全一致：
    ``ttm_net = 最新年报归母 + 最新一期归母 - 去年同期归母``；
    首行为业绩快报（``_快报源=True``）时跳过它取下一行（真实季报）作为最新期；
    同期对比行同样排除快报源行。利润表行含 ``报告期`` 键
    （"一季报" 或 "2026-03-31 [一季报]" 均可）与 ``归母净利润``（单位：元）。

    无法计算时返回 (0.0, 0.0)；TTM 可算但股本 <= 0 时返回 (0.0, ttm_net)。
    """
    if not profit_data:
        return 0.0, 0.0

    def _net(row: Dict[str, Any]) -> float:
        return _to_float(row.get("归母净利润")) or _to_float(row.get("扣非净利润"))

    annuals = [p for p in profit_data if _period_label(p) == "年报"]
    ann_net = _net(annuals[0]) if annuals else 0.0
    ttm_net = ann_net

    latest_label = _period_label(profit_data[0])
    # 业绩快报行插入 position 0 时会破坏 TTM 对比：跳过它用下一行（真实季报）
    if profit_data[0].get("_快报源") and len(profit_data) >= 2:
        latest_label = _period_label(profit_data[1])
    if latest_label and latest_label != "年报":
        same = [p for p in profit_data
                if _period_label(p) == latest_label and not p.get("_快报源")]
        if len(same) >= 2:
            # same[0]=最新一期，same[1]=去年同期（列表按报告期倒序）
            ttm_net = ann_net + _net(same[0]) - _net(same[1])

    if ttm_net == 0:
        return 0.0, 0.0
    eps = ttm_net / total_shares if total_shares > 0 and ttm_net > 0 else 0.0
    return eps, ttm_net


def adjust_bps_for_event(
    latest_equity_yi: Optional[float],
    net_raised_yi: Optional[float],
    post_shares: Optional[float],
    disclosed_bps: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """股本变动事件后的 BPS 调整（FIX-02）。

    :param latest_equity_yi: 最近期归母权益（亿元）。
    :param net_raised_yi: 募集净额（亿元）。**公式路径必需**——未知时不得按 0 处理。
    :param post_shares: 发行后总股本（股）。
    :param disclosed_bps: 公告披露的发行后每股净资产（元），存在时优先采用。

    :return: (bps_adjusted, basis)。公告披露值优先
        （basis="公告披露发行后每股净资产"）；否则按
        ``(latest_equity_yi + net_raised_yi) * 1e8 / post_shares`` 计算
        （basis="发行后口径(含募集净额)"）；输入不足返回 (None, "")。

    注意：募集净额未知时 ``(现有权益+0)/变动后股本`` 会产生比财报口径更大的失真
    （托伦斯真实案例：9.1亿权益/1.8547亿股=4.91 元，掩盖 9.41 亿募资已到账的事实，
    且 basis 会谎称"含募集净额"）。此时宁可退回财报口径 + 跨源比对告警。
    """
    if disclosed_bps is not None and disclosed_bps > 0:
        return float(disclosed_bps), "公告披露发行后每股净资产"
    if latest_equity_yi is None or not post_shares or post_shares <= 0:
        return None, ""
    if not net_raised_yi or float(net_raised_yi) <= 0:
        return None, ""
    bps = (float(latest_equity_yi) + float(net_raised_yi)) * 1e8 / float(post_shares)
    return bps, "发行后口径(含募集净额)"


__all__ = [
    "ShareContext",
    "resolve_share_context",
    "compute_ttm_eps",
    "adjust_bps_for_event",
    "clear_cache",
]
