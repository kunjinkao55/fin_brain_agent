"""次新股数据包（FIX-08）+ 业绩预告解析与情景联动（FIX-09）。

系统原本完全缺失上市日期、解禁日程、流通盘比例、客户集中度、IPO 发行参数等
次新股关键数据。本模块基于免费数据源（东财 datacenter F10 接口 + 公告正文）
尽力解析，所有字段允许 None，绝不抛异常，缺失项记入 missing 清单（优雅降级）。

FIX-09 联动：业绩预告增速可与三情景 EPS 隐含增速比对，输出情景概率标注文本。

本模块不 import backend.agent；网络请求仅在 get_listing_date 中进行（requests）。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Callable, Optional

import requests

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _to_float(s: str) -> Optional[float]:
    """去千分位逗号转 float，失败 None。"""
    try:
        return float(str(s).replace(',', ''))
    except (ValueError, TypeError):
        return None


def _parse_date(s: str) -> Optional[date]:
    """解析 "YYYY-MM-DD"（兼容带时间的 "YYYY-MM-DD HH:MM:SS"），失败 None。"""
    try:
        return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _add_months(d: date, months: int) -> date:
    """日期加 N 个月（月末溢出收敛到当月最后一天），不依赖 dateutil。"""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    last_day = [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return date(y, m, min(d.day, last_day))


def _market_suffix(symbol: str) -> str:
    """6 位代码 → 东财市场后缀。"""
    if symbol.startswith(('4', '8')):
        return "BJ"
    if symbol.startswith('6'):
        return "SH"
    return "SZ"


# ---------------------------------------------------------------------------
# 上市日期与次新股判定
# ---------------------------------------------------------------------------

def get_listing_date(symbol: str, fetch_announcements: Optional[Callable] = None) -> Optional[str]:
    """获取上市日期，返回 "YYYY-MM-DD" 或 None。

    主路径：东财 datacenter F10 公司概况接口（reportName=RPT_F10_BASIC_ORGINFO，
    字段 LISTING_DATE）；请求失败/无字段则回退到公告列表中标题含"上市公告书"
    的公告日期。网络请求 try/except 全包裹，超时 10s。
    """
    # 主路径：东财 F10 公司概况
    try:
        secucode = f"{symbol}.{_market_suffix(str(symbol))}"
        url = ("https://datacenter.eastmoney.com/securities/api/data/v1/get"
               f"?reportName=RPT_F10_BASIC_ORGINFO&columns=ALL"
               f"&filter=(SECUCODE%3D%22{secucode}%22)")
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=10)
        data = resp.json()
        rows = (data.get("result") or {}).get("data") or []
        if rows:
            d = _parse_date(str(rows[0].get("LISTING_DATE", "")))
            if d:
                return d.isoformat()
    except Exception:
        pass  # 网络/解析失败一律走回退

    # 回退：公告列表中"上市公告书"的公告日期
    if fetch_announcements is not None:
        try:
            anns = fetch_announcements(symbol)
            for it in (anns.get("列表") or []):
                if "上市公告书" in str(it.get("标题", "")):
                    d = _parse_date(str(it.get("日期", "")))
                    if d:
                        return d.isoformat()
        except Exception:
            pass
    return None


def is_new_listing(listing_date: str, months: int = 12, ref_date: Optional[date] = None) -> bool:
    """上市距今不超过 months 个月视为次新股。日期无法解析 → False。"""
    d = _parse_date(listing_date)
    if d is None:
        return False
    ref = ref_date or date.today()
    if d > ref:
        return True  # 未来上市（已申购未上市）按次新对待
    return (ref - d).days <= months * 30.44


# ---------------------------------------------------------------------------
# 上市公告书 / 招股说明书 正文解析
# ---------------------------------------------------------------------------

def parse_ipo_text(content: str) -> dict:
    """解析上市公告书正文中的发行/股本参数（独立实现，宽字段，尽力而为）。

    返回 dict，键取不到则为 None：
      issue_price      发行价格（元/股）
      issue_shares_wan 发行数量（万股）
      total_shares_wan 发行后总股本（万股）
      raise_total_yi   募集资金总额（亿元）
      pe_issue         发行市盈率（倍）
    """
    result: dict = {"issue_price": None, "issue_shares_wan": None,
                    "total_shares_wan": None, "raise_total_yi": None, "pe_issue": None}
    if not content:
        return result
    try:
        m = re.search(r'发行价格[为：:\s]*([\d.]+)\s*元', content)
        if m:
            result["issue_price"] = float(m.group(1))
        m = re.search(r'发行(?:股票|A股)?数量[为：:\s]*([\d,]+(?:\.\d+)?)\s*万股', content)
        if m:
            result["issue_shares_wan"] = _to_float(m.group(1))
        m = re.search(r'发行后总股本[为：:\s]*([\d,]+(?:\.\d+)?)\s*万股', content)
        if m:
            result["total_shares_wan"] = _to_float(m.group(1))
        m = re.search(r'募集资金总额[为约：:\s]*([\d,]+(?:\.\d+)?)\s*([亿万])元', content)
        if m:
            v = _to_float(m.group(1))
            if v is not None:
                result["raise_total_yi"] = round(v / 10000, 4) if m.group(2) == "万" else round(v, 4)
        m = re.search(r'发行市盈率[为：:\s]*([\d.]+)\s*倍', content)
        if m:
            result["pe_issue"] = float(m.group(1))
    except Exception:
        pass
    return result


def parse_lockups(content: str, listing_date: Optional[str] = None) -> list[dict]:
    """从上市公告书"限售安排/锁定期"段落提取解禁日程。

    返回 [{"date": "YYYY-MM-DD"|None, "shares_wan": float|None,
           "type": "网下配售|战略配售|首发原股东|其他", "note": str}]
    识别 "(N)个月锁定"/"锁定(N)个月"/"限售期(N)个月"；给出 listing_date 时推算解禁日。
    """
    lockups: list[dict] = []
    if not content:
        return lockups
    try:
        base = _parse_date(listing_date) if listing_date else None
        pat = r'(\d{1,3})\s*个月(?:内)?[^。；\n]{0,8}(?:锁定|限售)|(?:锁定|限售期|限售)[^。；\n]{0,8}?(\d{1,3})\s*个月'
        for m in re.finditer(pat, content):
            months = int(m.group(1) or m.group(2))
            ctx = content[max(0, m.start() - 80):m.end() + 40]
            # 结合上下文判断限售主体类型
            if "网下" in ctx:
                ltype = "网下配售"
            elif "战略" in ctx:
                ltype = "战略配售"
            elif any(w in ctx for w in ("控股股东", "实际控制人", "首发前", "发起人")):
                ltype = "首发原股东"
            else:
                ltype = "其他"
            shares = None
            sm = re.search(r'([\d,]+(?:\.\d+)?)\s*万股', ctx)
            if sm:
                shares = _to_float(sm.group(1))
            unlock = _add_months(base, months).isoformat() if base else None
            lockups.append({
                "date": unlock,
                "shares_wan": shares,
                "type": ltype,
                "note": re.sub(r'\s+', ' ', ctx).strip()[:80],
            })
    except Exception:
        pass
    return lockups


def parse_float_ratio(content: str) -> Optional[float]:
    """解析上市流通盘占总股本比例，返回小数（如 0.1662），失败 None。"""
    if not content:
        return None
    patterns = [
        # 兼容"无限售条件流通股"与"无限售条件的流通股"两种写法
        r'无限售条件?的?流通股[^%]{0,60}?占[^%]{0,20}?([\d.]+)\s*%',
        r'本次上市流通[^%]{0,60}?占[^%]{0,20}?([\d.]+)\s*%',
    ]
    for pat in patterns:
        try:
            m = re.search(pat, content)
            if m:
                return round(float(m.group(1)) / 100, 6)
        except Exception:
            continue
    return None


def parse_customer_concentration(content: str) -> Optional[dict]:
    """解析客户集中度：{"top5_pct": 小数, "detail": 原文片段}，
    能识别单一大客户时附 "top1_name"/"top1_pct"。失败 None。"""
    if not content:
        return None
    try:
        idx = content.find("前五大客户")
        if idx < 0:
            idx = content.find("前五名客户")
        if idx < 0:
            return None
        seg = content[idx:idx + 160]
        m = re.search(r'([\d.]+)\s*%', seg)
        if not m:
            return None
        result: dict = {"top5_pct": round(float(m.group(1)) / 100, 6),
                        "detail": re.sub(r'\s+', ' ', seg).strip()[:120]}
        # 尽力识别第一大客户名称与占比
        m1 = re.search(r'第一大客户(?:为|是)?[：:\s]*([一-龥A-Za-z0-9（）()]{2,20}?)[，,、\s]', seg)
        m1p = re.search(r'第一大客户[^%]{0,40}?([\d.]+)\s*%', seg)
        if m1:
            result["top1_name"] = m1.group(1)
        if m1p:
            result["top1_pct"] = round(float(m1p.group(1)) / 100, 6)
        # "其中A公司占12.34%"语序（招股书/年报常见）
        if "top1_pct" not in result:
            m2 = re.search(r'其中[，,]?\s*([一-龥A-Za-z0-9（）()]{2,20}?)占\s*([\d.]+)\s*%', seg)
            if m2:
                result.setdefault("top1_name", m2.group(1))
                result["top1_pct"] = round(float(m2.group(2)) / 100, 6)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 业绩预告解析（FIX-09）
# ---------------------------------------------------------------------------

def _range_pct(seg: str) -> tuple[Optional[float], Optional[float]]:
    """从片段解析增速区间："增长25%~30%"→(0.25,0.30)、"下降约30%"→(-0.30,-0.30)。
    兼容 "比上年同期增长：54% - 69%"（冒号+空格+连字符）与 en dash。"""
    m = re.search(r'(增长|上升|增加|下降|下滑|减少)\s*(?:约|近)?\s*[：:]?\s*([\d.]+)\s*%\s*[~～—–\-至]\s*([\d.]+)\s*%', seg)
    if m:
        sign = -1.0 if m.group(1) in ("下降", "下滑", "减少") else 1.0
        lo, hi = sorted([float(m.group(2)), float(m.group(3))])
        return round(sign * lo / 100, 6), round(sign * hi / 100, 6)
    m = re.search(r'(增长|上升|增加|下降|下滑|减少)\s*(?:约|近)?\s*[：:]?\s*([\d.]+)\s*%', seg)
    if m:
        sign = -1.0 if m.group(1) in ("下降", "下滑", "减少") else 1.0
        v = round(sign * float(m.group(2)) / 100, 6)
        return v, v
    return None, None


def _range_amount_yi(seg: str) -> tuple[Optional[float], Optional[float]]:
    """从片段解析金额区间（统一亿元）："38,300万元~39,800万元"→(3.83,3.98)。
    兼容 en dash（–）与"盈利：500,000 万元–550,000 万元"格式。"""
    m = re.search(r'([\d,]+(?:\.\d+)?)\s*([亿万])元?\s*[~～—–\-至]\s*([\d,]+(?:\.\d+)?)\s*([亿万])元?', seg)
    if not m:
        return None, None
    lo, hi = _to_float(m.group(1)), _to_float(m.group(3))
    if lo is None or hi is None:
        return None, None
    if m.group(2) == "万":
        lo /= 10000
    if m.group(4) == "万":
        hi /= 10000
    lo, hi = sorted([lo, hi])
    return round(lo, 4), round(hi, 4)


def _np_amount_yi(text: str) -> tuple[Optional[float], Optional[float]]:
    """归母净利润金额区间："盈利：500,000 万元–550,000 万元"→(50.0, 55.0)。
    兼容"净利润 X 亿元~Y 亿元"、"盈利：X亿元"单值。返回(下限,上限)或(None,None)。"""
    # 区间：盈利：X万/亿 ~ Y万/亿（"盈利："前缀可省略，但要求紧邻分隔符）
    m = re.search(r'([\d,]+(?:\.\d+)?)\s*([亿万])元?\s*[~～—–\-至]\s*([\d,]+(?:\.\d+)?)\s*([亿万])元?', text)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(3))
        if lo is not None and hi is not None:
            if m.group(2) == "万":
                lo /= 10000
            if m.group(4) == "万":
                hi /= 10000
            lo, hi = sorted([lo, hi])
            return round(lo, 4), round(hi, 4)
    # 单值：盈利：X 万/亿元（取第一个"盈利："后的数字）
    m = re.search(r'盈利[：:]\s*([\d,]+(?:\.\d+)?)\s*([亿万])元?', text)
    if m:
        v = _to_float(m.group(1))
        if v is not None:
            if m.group(2) == "万":
                v /= 10000
            return round(v, 4), round(v, 4)
    return None, None


def parse_performance_forecast(content: str, title: str = "") -> Optional[dict]:
    """解析业绩预告正文（FIX-09）。无任何有效数字时返回 None。

    返回 {"period": str, "rev_min_yi"/"rev_max_yi": 亿元区间,
          "rev_growth_min"/"rev_growth_max": 小数增速区间,
          "np_growth_min"/"np_growth_max": 小数增速区间,
          "forecast_type": "预增|预减|首亏|扭亏|续亏|不确定"}
    支持区间表述："38,300万元~39,800万元"、"增长25%~30%"、"下降约30%"。
    """
    if not content and not title:
        return None
    try:
        text = content or ""
        result: dict = {"period": None, "rev_min_yi": None, "rev_max_yi": None,
                        "rev_growth_min": None, "rev_growth_max": None,
                        "np_min_yi": None, "np_max_yi": None,
                        "np_growth_min": None, "np_growth_max": None,
                        "forecast_type": "不确定"}

        m = re.search(r'20\d{2}\s*年\s*(?:半年度|一季度|第一季度|前三季度|第三季度|年度)?', title or text)
        if m:
            result["period"] = m.group(0).replace(" ", "")

        # 营收金额区间（定位"营业收入/营业总收入"后的窗口）
        for kw in ("营业总收入", "营业收入"):
            idx = text.find(kw)
            if idx >= 0:
                lo, hi = _range_amount_yi(text[idx:idx + 120])
                if lo is not None:
                    result["rev_min_yi"], result["rev_max_yi"] = lo, hi
                    break

        # 营收增速区间：营收关键词窗口内的增长/下降表述
        rev_seg = ""
        for kw in ("营业总收入", "营业收入"):
            idx = text.find(kw)
            if idx >= 0:
                rev_seg = text[idx:idx + 150]
                break
        if rev_seg:
            glo, ghi = _range_pct(rev_seg)
            result["rev_growth_min"], result["rev_growth_max"] = glo, ghi

        # 归母净利金额+增速：定位"净利润"（排除"扣非"以免混入口径），
        # 窗口向前扩 100 字符（"盈利：500,000 万元–550,000 万元"可能位于标签之前）
        np_idx = -1
        for kw in ("归属于上市公司股东的净利润", "的净利润", "净利润"):
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx < 0:
                    break
                if "扣非" not in text[max(0, idx - 20):idx]:
                    np_idx = idx
                    break
                start = idx + len(kw)
            if np_idx >= 0:
                break
        if np_idx >= 0:
            np_win = text[max(0, np_idx - 100):np_idx + 150]
            lo, hi = _np_amount_yi(np_win)
            result["np_min_yi"], result["np_max_yi"] = lo, hi
            glo, ghi = _range_pct(np_win)
            result["np_growth_min"], result["np_growth_max"] = glo, ghi

        # 预告类型：标题/正文关键词优先，否则按净利增速方向推断
        merged = (title or "") + " " + text[:300]
        if "扭亏" in merged:
            result["forecast_type"] = "扭亏"
        elif "首亏" in merged:
            result["forecast_type"] = "首亏"
        elif "续亏" in merged:
            result["forecast_type"] = "续亏"
        elif "预亏" in merged:
            result["forecast_type"] = "首亏"
        elif "预增" in merged or "同向上升" in merged:
            result["forecast_type"] = "预增"
        elif "预减" in merged or "同向下降" in merged:
            result["forecast_type"] = "预减"
        elif result["np_growth_max"] is not None:
            result["forecast_type"] = "预减" if result["np_growth_max"] < 0 else "预增"

        # 一个有效数字都没有 → None
        numeric_keys = ("rev_min_yi", "rev_growth_min", "np_min_yi", "np_growth_min")
        if all(result[k] is None for k in numeric_keys):
            return None
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FIX-09 联动：业绩预告 × 三情景
# ---------------------------------------------------------------------------

def forecast_scenario_link(forecast: dict, scenario_growths: dict) -> Optional[str]:
    """业绩预告增速与三情景 EPS 隐含增速比对，返回情景概率标注文本。

    scenario_growths: {"悲观": -0.15, "基准": 0.04, "乐观": 0.23}（小数增速）。
    情景区间按相邻情景增速中点划分（悲观区间上界=(悲观+基准)/2，依此类推），
    且悲观区间下界/乐观区间上界外延一个区间宽度，超出外延视为"超出全部情景"。
    无法判定 → None。
    """
    try:
        if not isinstance(forecast, dict) or not isinstance(scenario_growths, dict):
            return None
        lo = forecast.get("np_growth_min")
        hi = forecast.get("np_growth_max")
        if lo is None and hi is None:
            return None
        if lo is None:
            lo = hi
        if hi is None:
            hi = lo
        mid = (float(lo) + float(hi)) / 2

        vals = [scenario_growths.get(n) for n in ("悲观", "基准", "乐观")]
        if any(v is None for v in vals):
            return None
        pess, base, opt = (float(v) for v in vals)
        if not (pess <= base <= opt):
            return None
        mid_pb = (pess + base) / 2   # 悲观/基准分界
        mid_bo = (base + opt) / 2    # 基准/乐观分界
        width_low = base - pess
        width_high = opt - base
        pct = f"{mid:+.0%}"

        if mid < pess - width_low or mid > opt + width_high:
            return f"业绩预告增速{pct}超出全部情景区间，建议重估概率权重"
        if mid < mid_pb:
            return f"业绩预告净利增速{pct}，落在悲观情景区间，悲观情形概率上升"
        if mid <= mid_bo:
            return f"业绩预告净利增速{pct}，落在基准情景区间，基准情形概率上升"
        return f"业绩预告净利增速{pct}，落在乐观情景区间，乐观情形概率上升"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 汇总入口
# ---------------------------------------------------------------------------

def fetch_new_listing_pack(symbol: str,
                           fetch_announcements: Optional[Callable] = None,
                           fetch_content: Optional[Callable] = None) -> dict:
    """汇总次新股数据包。所有步骤 try/except，单步失败不影响其他字段。

    fetch_announcements: func(symbol, count) -> {"列表": [{"标题","日期",...}]}
    fetch_content:       func(ann) -> 公告正文 str
    两者缺省时函数仍可用（如仅靠 datacenter 拿到 listing_date），缺失记入 missing。
    """
    pack: dict = {"listing_date": None, "is_new": False, "ipo": None,
                  "float_ratio": None, "lockups": [], "customer_concentration": None,
                  "forecast": None, "missing": []}

    # 1) 上市日期（内部已做网络异常兜底）
    try:
        pack["listing_date"] = get_listing_date(symbol, fetch_announcements)
    except Exception:
        pass

    # 2) 非次新股（上市超 12 个月）早退；listing_date 未知时继续尽力走公告路径
    if pack["listing_date"] and not is_new_listing(pack["listing_date"]):
        pack["missing"] = [k for k in ("ipo", "float_ratio", "lockups",
                                       "customer_concentration", "forecast")]
        return pack
    pack["is_new"] = bool(pack["listing_date"])

    # 3) 公告列表：找上市公告书/招股说明书/业绩预告（多候选，全文/提示性公告互补）
    ipo_candidates: list = []
    ann_fc = None
    if fetch_announcements is not None:
        try:
            try:
                anns = fetch_announcements(symbol, 30)
            except TypeError:
                anns = fetch_announcements(symbol)
            for it in (anns.get("列表") or []):
                title = str(it.get("标题", ""))
                if len(ipo_candidates) < 3 and any(k in title for k in ("上市公告书", "招股说明书", "招股意向书")):
                    ipo_candidates.append(it)
                if ann_fc is None and "业绩预告" in title:
                    ann_fc = it
        except Exception:
            pass

    # 4) 有 fetch_content 才抓正文解析；逐候选解析，字段级 first-non-None 合并
    # （np-cnotice 正文常为节选，提示性公告与全文各有缺失，互补合并覆盖率最高）
    def _first_none(current, new):
        return current if current is not None else new

    if fetch_content is not None:
        for ann_ipo in ipo_candidates:
            try:
                content = fetch_content(ann_ipo) or ""
                if not content:
                    continue
                ipo_parsed = parse_ipo_text(content)
                if pack["ipo"] is None:
                    pack["ipo"] = ipo_parsed
                elif isinstance(pack["ipo"], dict) and isinstance(ipo_parsed, dict):
                    pack["ipo"] = {k: _first_none(pack["ipo"].get(k), v)
                                   for k, v in ipo_parsed.items()}
                pack["float_ratio"] = _first_none(pack["float_ratio"], parse_float_ratio(content))
                if not pack["lockups"]:
                    pack["lockups"] = parse_lockups(content, pack["listing_date"])
                pack["customer_concentration"] = _first_none(
                    pack["customer_concentration"], parse_customer_concentration(content))
            except Exception:
                continue
    if fetch_content is not None and ann_fc is not None:
        try:
            fc_text = fetch_content(ann_fc) or ""
            pack["forecast"] = parse_performance_forecast(fc_text, str(ann_fc.get("标题", "")))
        except Exception:
            pass

    # 5) missing 清单
    for key in ("listing_date", "ipo", "float_ratio", "customer_concentration", "forecast"):
        if pack[key] is None:
            pack["missing"].append(key)
    if not pack["lockups"]:
        pack["missing"].append("lockups")
    return pack
