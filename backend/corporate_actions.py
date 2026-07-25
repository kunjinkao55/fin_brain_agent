"""公司行动事件分类器（FIX-03）。

背景问题：backend/agent.py:2468-2493 的现有逻辑把公告标题命中
"发行A股|非公开发行|定向增发|募集资金|发行股份" 一概当作定增，
导致 IPO 被误判为定增，且稀释系数硬编码 0.874。本模块提供：

- :func:`classify_events`：按优先级（IPO 优先于定增）对公告做标题级分类，
  可选地通过 ``fetch_content`` 回调抓取正文解析事件参数；
- :func:`parse_ipo_params`：从上市公告书正文提取发行价/募资/发行后股本等；
- :func:`parse_private_placement_params`：定增参数提取（迁移自
  agent.py:2476-2493 的正则逻辑，保留"新股数 > 3×总股本丢弃"的 sanity 校验）。

约束：不得 import backend.agent（避免循环依赖）；仅依赖标准库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

EVENT_TYPES = ("IPO", "定增", "配股", "送转", "回购注销", "可转债转股")


@dataclass
class CorporateEvent:
    """公司行动事件：类型 + 参数 + 血缘。"""

    type: str                     # EVENT_TYPES 之一
    params: Dict[str, Any] = field(default_factory=dict)
    as_of_date: str = ""          # 公告日期
    source: str = ""              # 血缘：公告标题/接口


# ---------------------------------------------------------------------------
#  标题分类（优先级从上到下，IPO 优先于定增）
# ---------------------------------------------------------------------------

_RE_PLACEMENT = re.compile(r"非公开发行|定向增发|向特定对象发行")
_RE_RATIONED = re.compile(r"配股")
_RE_BONUS = re.compile(r"权益分派|送转|10转\d+|10送\d+")
_RE_BUYBACK_CANCEL = re.compile(r"回购注销|注销.*回购|回购.*注销")
_RE_CB_CONVERT = re.compile(r"可转债.*转股|转股.*可转债|可转换公司债券")


def _classify_title(title: str) -> Optional[str]:
    """按标题分类公司行动事件；无法分类返回 None。

    注意："首次公开发行"（IPO）不含"非公开发行"字样，不会落入定增分支；
    IPO 判定要求含"上市"，单独的"首次公开发行"提示性公告不误判。
    """
    if "上市公告书" in title or ("首次公开发行" in title and "上市" in title):
        return "IPO"
    if _RE_PLACEMENT.search(title):
        return "定增"
    if _RE_RATIONED.search(title):
        return "配股"
    if _RE_BONUS.search(title):
        return "送转"
    if _RE_BUYBACK_CANCEL.search(title):
        return "回购注销"
    if _RE_CB_CONVERT.search(title):
        return "可转债转股"
    return None


def classify_events(
    announcements: Union[List[Dict[str, Any]], Dict[str, Any], None],
    fetch_content: Optional[Callable[[Dict[str, Any]], str]] = None,
    max_content_fetch: int = 3,
) -> List[CorporateEvent]:
    """对公告列表做公司行动事件分类。

    :param announcements: 公告 dict 列表（含 ``标题`` / ``日期`` 键），
        也兼容 ``get_recent_announcements`` 返回的 ``{"列表": [...]}`` 包装。
    :param fetch_content: 可选回调 ``func(ann_dict) -> str`` 抓取公告正文；
        未提供时仅做标题级分类（params 为空 dict）。
    :param max_content_fetch: 正文抓取次数上限（防止慢请求拖垮流程）。

    仅 IPO / 定增事件在提供 fetch_content 时解析正文参数；其余事件类型
    标题级分类即可生成事件。
    """
    if isinstance(announcements, dict):
        announcements = announcements.get("列表", []) or []
    if not announcements:
        return []

    events: List[CorporateEvent] = []
    fetched = 0
    for ann in announcements:
        if not isinstance(ann, dict):
            continue
        title = str(ann.get("标题") or ann.get("title") or "")
        date = str(ann.get("日期") or ann.get("date") or "")
        etype = _classify_title(title)
        if etype is None:
            continue

        params: Dict[str, Any] = {}
        content_used = False
        if etype in ("IPO", "定增") and fetch_content is not None and fetched < max_content_fetch:
            fetched += 1  # 计数在调用前：失败的抓取同样消耗额度
            try:
                content = fetch_content(ann) or ""
            except Exception:
                content = ""
            if content:
                content_used = True
                if etype == "IPO":
                    params = parse_ipo_params(content)
                else:
                    params = parse_private_placement_params(content)

        events.append(CorporateEvent(
            type=etype,
            params=params,
            as_of_date=date,
            source=f"公告标题《{title}》" + ("+正文解析" if content_used else ""),
        ))
    return events


# ---------------------------------------------------------------------------
#  数字解析辅助：兼容千分位逗号与"万/亿"单位
# ---------------------------------------------------------------------------

def _num(s: str) -> float:
    """去掉千分位逗号后转 float。"""
    return float(s.replace(",", ""))


def _amount_to_yi(value: str, unit: Optional[str]) -> float:
    """金额统一转亿元。"""
    v = _num(value)
    if unit == "亿":
        return v
    if unit == "万":
        return v / 10000
    return v / 1e8


def _shares_to_wan(value: str, unit: Optional[str]) -> float:
    """股数统一转万股。"""
    v = _num(value)
    if unit == "亿":
        return v * 10000
    if unit == "万":
        return v
    return v / 10000


def _shares_to_yi(value: str, unit: Optional[str]) -> float:
    """股数统一转亿股（与 agent.py:2476-2482 的单位换算一致）。"""
    v = _num(value)
    if unit == "亿":
        return v
    if unit == "万":
        return v / 10000
    return v / 1e8


# ---------------------------------------------------------------------------
#  正文参数解析
# ---------------------------------------------------------------------------

def parse_ipo_params(content: str) -> Dict[str, float]:
    """从上市公告书正文提取 IPO 关键参数。

    可提取键（全部允许缺失，缺失键不出现）：
    ``发行股数(万股)`` / ``发行价(元)`` / ``募资总额(亿元)`` / ``募资净额(亿元)``
    / ``发行后总股本(万股)`` / ``发行后每股净资产(元)`` / ``发行市盈率``。

    数字兼容千分位逗号与"万/亿"单位，统一返回 float
    （股数→万股，金额→亿元）。
    """
    params: Dict[str, float] = {}
    if not content:
        return params

    # 发行股数：如 "发行新股4,636.8423万股" / "首次公开发行股票数量为4,636.8423 万股"
    # 负向断言排除"公开发行后总股本为X万股"（总股本≠新发股数，托伦斯真实公告教训）
    m = re.search(r"发行新股\s*([\d,]+\.?\d*)\s*(亿|万)?股", content)
    if not m:
        m = re.search(r"公开发行(?:股票|A股)?(?:数量|股数)\s*(?:为)?\s*([\d,]+\.?\d*)\s*(亿|万)?股", content)
    if not m:
        m = re.search(r"(?:本次)?公开发行(?!后总股本)[^，。]{0,20}?([\d,]+\.?\d*)\s*(亿|万)?股", content)
    if m:
        params["发行股数(万股)"] = _shares_to_wan(m.group(1), m.group(2))

    # 发行价：如 "发行价格22.60元/股"
    m = re.search(r"发行价格?\s*[：:]?\s*([\d,]+\.?\d*)\s*元", content)
    if m:
        params["发行价(元)"] = _num(m.group(1))

    # 募资总额 / 募资净额：如 "募集资金总额104,791.82万元" / "募集资金净额94,100.00万元"
    m = re.search(r"(?:募集资金总额|募资总额)\s*[约：:]?\s*([\d,]+\.?\d*)\s*(亿|万)?元", content)
    if m:
        params["募资总额(亿元)"] = _amount_to_yi(m.group(1), m.group(2))
    m = re.search(r"(?:募集资金净额|募资净额)\s*[约：:]?\s*([\d,]+\.?\d*)\s*(亿|万)?元", content)
    if m:
        params["募资净额(亿元)"] = _amount_to_yi(m.group(1), m.group(2))

    # 发行后总股本：如 "发行后总股本18,547.3692万股"
    m = re.search(r"发行后总股本\s*[约：:]?\s*([\d,]+\.?\d*)\s*(亿|万)?股", content)
    if m:
        params["发行后总股本(万股)"] = _shares_to_wan(m.group(1), m.group(2))

    # 发行后每股净资产：如 "发行后每股净资产9.87元"
    m = re.search(r"发行后每股净资产\s*[约：:]?\s*([\d,]+\.?\d*)\s*元", content)
    if m:
        params["发行后每股净资产(元)"] = _num(m.group(1))

    # 发行市盈率：如 "发行市盈率：23.45倍"
    m = re.search(r"发行市盈率\s*[：:]?\s*([\d,]+\.?\d*)", content)
    if m:
        params["发行市盈率"] = _num(m.group(1))

    return params


def parse_private_placement_params(text: str, total_shares_yi: float = 0.0) -> Dict[str, float]:
    """从定增公告正文/标题拼接文本提取定增参数。

    迁移自 backend/agent.py:2476-2493 的现有正则逻辑，可提取键：
    ``发行股数(亿股)`` / ``募资金额(亿元)``（缺失键不出现）。

    保留 sanity 校验：解析出的新股数 > 3×总股本 时视为误匹配并丢弃
    （``total_shares_yi <= 0`` 时校验不生效，与原逻辑一致）。
    """
    params: Dict[str, float] = {}
    if not text:
        return params

    # 发行股数（统一转为亿股）
    m = re.search(r"([\d,]+\.?\d*)\s*(亿|万)?股", text)
    if m:
        new_shares = _shares_to_yi(m.group(1), m.group(2))
        # sanity：新股数不可能超过总股本 3 倍，超过说明匹配到了无关数字
        if not (total_shares_yi > 0 and new_shares > total_shares_yi * 3):
            params["发行股数(亿股)"] = new_shares

    # 募资金额（统一转为亿元）：如 "募集资金总额不超过50亿元"
    m = re.search(r"(?:募集资金|募资)(?:总额|净额)?(?:不超过)?\s*([\d,]+\.?\d*)\s*(亿|万)?元", text)
    if m:
        params["募资金额(亿元)"] = _amount_to_yi(m.group(1), m.group(2))

    return params


__all__ = [
    "EVENT_TYPES",
    "CorporateEvent",
    "classify_events",
    "parse_ipo_params",
    "parse_private_placement_params",
]
