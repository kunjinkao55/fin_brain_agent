"""利润引擎解剖与 SOTP 拆分（分类引擎修正方法 支柱一/二）。

旧范式：按"行业标签"套行业平均 PE。
修正：按**利润贡献溯源**把公司拆成若干"利润引擎"，按经济属性分类；
单一引擎毛利贡献 >40% 时估值锚向该引擎倾斜（支柱一）；
板块增速差 >30pp 或毛利率差 >15pp 时强制 SOTP 拆分（支柱二）。

口径说明：板块净利润免费源不可得，**毛利额占比 ≈ 利润引擎占比**（最优近似）。
"其他/其中:子项"行不参与引擎判定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ENGINE_TYPES = ("复利型", "周期型", "成长型", "价值陷阱型", "常规")

# 支柱二：强制隔离阀阈值（默认值；实际取值以 configs/scoring.json [估值守卫.SOTP] 为准）
SOTP_GROWTH_GAP_PP = 30.0   # 板块增速差 > 30 个百分点
SOTP_GM_GAP_PP = 15.0       # 板块毛利率差 > 15 个百分点
# 支柱一：锚定倾斜阈值（单一引擎毛利贡献）
ANCHOR_GP_THRESHOLD = 0.40
# SOTP 总部未分配成本折价
HQ_DISCOUNT = 0.10


def _sotp_cfg() -> dict:
    """SOTP 参数（configs/scoring.json [估值守卫.SOTP]，禁止硬编码；缺失回退模块默认）。"""
    try:
        from backend.scoring_config import get_valuation_guards
        return get_valuation_guards()["sotp"]
    except Exception:
        return {"增速差pp": SOTP_GROWTH_GAP_PP, "毛利率差pp": SOTP_GM_GAP_PP,
                "锚定毛利占比": ANCHOR_GP_THRESHOLD, "总部折价": HQ_DISCOUNT}


@dataclass
class Engine:
    """一个利润引擎（业务板块）。"""
    name: str
    rev_pct: float                    # 收入占比（小数）
    gp_pct: float                     # 毛利占比（小数）= 利润引擎占比
    gross_margin: Optional[float]     # 毛利率（小数）
    rev_growth: Optional[float]       # 收入同比（小数），无前期数据则 None
    gm_trend_pp: Optional[float]      # 毛利率同比变化（百分点）
    engine_type: str = "常规"
    note: str = ""


def _main_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤 子项(其中:xx) 与 其他 行，只留主板块。"""
    return [s for s in segments
            if not s.get("_子项") and not s.get("_其他") and (s.get("收入") or 0) > 0]


def _growth_map(prev_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """前期板块名 → {收入, 毛利率}，用于同比计算。"""
    out: Dict[str, Dict[str, float]] = {}
    for s in _main_segments(prev_segments or []):
        out[str(s.get("名称", ""))] = {
            "rev": float(s.get("收入") or 0),
            "gm": float(s["毛利率"]) if s.get("毛利率") is not None else float("nan"),
        }
    return out


def classify_engines(
    segments: List[Dict[str, Any]],
    prev_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Engine]:
    """支柱一：利润引擎解剖——按经济属性给每个主板块打标签。

    规则（优先级 价值陷阱 > 成长 > 复利 > 周期 > 常规）：
      价值陷阱型: 收入增长但毛利率同比降 >3pp（降价冲规模）
      成长型:     收入同比 >20% 且毛利率同比不降（渗透率低、第二曲线）
      复利型:     毛利率 >30% 且同比不降（定价权）
      周期型:     毛利率同比波动 >5pp（供需敏感）
      常规:       其他（数据不足或特征不显著）
    """
    prev = _growth_map(prev_segments or [])
    engines: List[Engine] = []
    for s in _main_segments(segments):
        name = str(s.get("名称", ""))
        gm = s.get("毛利率")
        gm = float(gm) if gm is not None else None
        rev_g = None
        gm_pp = None
        p = prev.get(name)
        if p:
            if p["rev"] > 0:
                rev_g = float(s.get("收入") or 0) / p["rev"] - 1
            if gm is not None and p["gm"] == p["gm"]:  # NaN 检查
                gm_pp = (gm - p["gm"]) * 100

        etype = "常规"
        note = ""
        if rev_g is not None and rev_g > 0 and gm_pp is not None and gm_pp < -3:
            etype = "价值陷阱型"
            note = f"增收+毛利率{gm_pp:.1f}pp，降价冲规模特征"
        elif rev_g is not None and rev_g > 0.20 and (gm_pp is None or gm_pp >= 0):
            etype = "成长型"
            note = f"收入同比{rev_g*100:+.0f}%且毛利率不降"
        elif gm is not None and gm > 0.30 and (gm_pp is None or gm_pp >= 0):
            etype = "复利型"
            note = f"毛利率{gm*100:.1f}%且稳定/提升"
        elif gm_pp is not None and abs(gm_pp) > 5:
            etype = "周期型"
            note = f"毛利率同比{gm_pp:+.1f}pp，波动显著"
        engines.append(Engine(
            name=name,
            rev_pct=float(s.get("收入占比") or 0),
            gp_pct=float(s.get("毛利占比") or 0),
            gross_margin=gm,
            rev_growth=rev_g,
            gm_trend_pp=gm_pp,
            engine_type=etype,
            note=note,
        ))
    engines.sort(key=lambda e: e.gp_pct, reverse=True)
    return engines


def sotp_triggers(
    segments: List[Dict[str, Any]],
    prev_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """支柱二：强制隔离阀。命中任一条件 → 返回触发原因列表（空=不触发）。

    板块 ROIC 差免费源不可得，该子项标注跳过（口径见文档）。
    """
    reasons: List[str] = []
    mains = _main_segments(segments)
    _cfg = _sotp_cfg()
    _gm_gap = _cfg["毛利率差pp"]
    _g_gap = _cfg["增速差pp"]
    # 毛利率差 > 15pp（当期）
    gms = [float(s["毛利率"]) for s in mains if s.get("毛利率") is not None]
    if len(gms) >= 2 and (max(gms) - min(gms)) * 100 > _gm_gap:
        reasons.append(
            f"板块毛利率极差{(max(gms)-min(gms))*100:.1f}pp>{_gm_gap:.0f}pp，禁止混同估值")
    # 增速差 > 30pp（两期收入同比）
    growths: List[tuple[str, float]] = []
    prev = _growth_map(prev_segments or [])
    for s in mains:
        p = prev.get(str(s.get("名称", "")))
        if p and p["rev"] > 0:
            growths.append((str(s.get("名称", "")),
                            float(s.get("收入") or 0) / p["rev"] - 1))
    if len(growths) >= 2:
        gs = [g for _, g in growths]
        if (max(gs) - min(gs)) * 100 > _g_gap:
            hi = growths[gs.index(max(gs))]
            lo = growths[gs.index(min(gs))]
            reasons.append(
                f"板块增速极差{(max(gs)-min(gs))*100:.0f}pp>{_g_gap:.0f}pp"
                f"（{hi[0]}{hi[1]*100:+.0f}% vs {lo[0]}{lo[1]*100:+.0f}%），强制SOTP拆分")
    return reasons


def engine_anchor(
    engines: List[Engine],
    threshold: float = None,
) -> Optional[Engine]:
    """支柱一核心纪律：单一引擎毛利贡献超阈值 → 估值锚向该引擎倾斜。"""
    if threshold is None:
        threshold = _sotp_cfg()["锚定毛利占比"]
    for e in engines:
        if e.gp_pct > threshold:
            return e
    return None


def sotp_fair_value(
    engines: List[Engine],
    ttm_net: float,
    base_pe: float,
    hq_discount: float = None,
) -> tuple[float, List[str]]:
    """支柱二 SOTP 估值：Σ(各板块分配利润 × 板块独立PE) × (1-总部折价)。

    按毛利占比把 TTM 净利分配到板块；板块独立 PE = base_pe × 乘数档
    （乘数档复用支柱三矩阵 indicator_pe_matrix，按板块 增速/毛利率趋势 定档，
    CAPEX/CFO 为全公司指标不参与分板块）。
    返回 (每股合理价值之外的**总价值倍数说明**, 明细)——实际返回 (合理价值总额, 明细文本)，
    调用方自行除以总股本得到每股价值。

    :param engines: classify_engines 输出
    :param ttm_net: 公司 TTM 归母净利（元）
    :param base_pe: 行业 PE 中枢
    """
    from backend.scoring import indicator_pe_matrix

    if hq_discount is None:
        hq_discount = _sotp_cfg()["总部折价"]
    detail: List[str] = []
    if not engines or ttm_net <= 0 or base_pe <= 0:
        return 0.0, detail
    gp_sum = sum(e.gp_pct for e in engines)
    if gp_sum <= 0:
        return 0.0, detail
    total = 0.0
    for e in engines:
        share = e.gp_pct / gp_sum  # 归一化（子项/其他已剔除）
        seg_net = ttm_net * share
        growth_pct = (e.rev_growth * 100) if e.rev_growth is not None else 10.0
        gm_pp = e.gm_trend_pp if e.gm_trend_pp is not None else 0.0
        band, mult, _ = indicator_pe_matrix(growth_pct, gm_pp, None)
        seg_pe = base_pe * mult
        total += seg_net * seg_pe
        detail.append(
            f"{e.name}[{e.engine_type}]: 分配净利{seg_net/1e8:.2f}亿×PE{seg_pe:.0f}({band})"
            f"={seg_net*seg_pe/1e8:.1f}亿")
    total *= (1 - hq_discount)
    detail.append(f"总部未分配成本折价{hq_discount*100:.0f}%")
    return round(total, 2), detail
