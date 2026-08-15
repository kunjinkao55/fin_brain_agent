"""
黄金股票池 — 覆盖主要板块类别的评估基准数据。

设计：
    1. 每只股票内部自洽（TTM = 年报 + 最新季 - 上年同季；TTM EPS = TTM/股本；
       PE = 现价/TTM EPS；市值 = 现价×股本）。
    2. 数据来源标记 source:
         "real"      基于文档/审计报告中的真实披露值（301583 托伦斯、600131 国网信通）
         "synthetic" 内部自洽的合理构造值（覆盖其余板块，供 Layer B/C 测试，
                     不代表真实市场数据）
    3. Layer A（数据管道黄金比对）只对 real 股做精确数值比对；synthetic 股
       走宽松模式（仅验证结构完整性与内部自洽）。

板块覆盖（对应 accounting_rag 8 行业模板 + 精密制造）：
    半导体 / 电力能源 / 医药 / 消费品 / 通信光模块 / 制造新能源 / 金融 / 地产建筑 / 精密制造
"""

from dataclasses import dataclass, field


@dataclass
class GoldenStock:
    symbol: str
    name: str
    sector: str                 # 板块类别（对应行业模板）
    industry: str               # 行业名（注入 [INDUSTRY]）
    source: str                 # "real" | "synthetic"
    price: float
    total_shares: float         # 总股本（股）
    bps: float                  # 每股净资产（元）
    annual_net: float           # 最新年报归母净利润（元）
    q_last: float               # 最新一季报归母净利润（元）
    q_prev: float               # 上年同期归母净利润（元）
    pe: float                   # PE(TTM) 参考
    pb: float                   # PB 参考
    fair_value: float           # 合理价值（元）
    safe_price: float           # 安全买入价（元）
    roe: float = 15.0           # ROE(%)
    growth_note: str = ""       # 成长性依据
    risk_note: str = ""         # 主要风险
    extra_facts: dict = field(default_factory=dict)  # 额外黄金事实（如发行价）

    @property
    def ttm_net(self) -> float:
        return self.annual_net + self.q_last - self.q_prev

    @property
    def ttm_eps(self) -> float:
        return self.ttm_net / self.total_shares if self.total_shares else 0.0

    @property
    def market_cap(self) -> float:
        return self.price * self.total_shares / 1e8  # 亿元


# ============================================================
#  黄金股票池
# ============================================================

GOLDEN_STOCKS: dict[str, GoldenStock] = {
    # ---- 精密制造（real：上市公告书，见 tuolunsi golden）----
    "301583": GoldenStock(
        symbol="301583", name="托伦斯", sector="精密制造", industry="精密制造,机械设备",
        source="real", price=174.0, total_shares=1.85473692e8, bps=9.87,
        roe=11.78, annual_net=0.9818e8, q_last=0.1464e8, q_prev=0.1600e8,
        pe=337.0, pb=17.6, fair_value=31.9, safe_price=23.9,
        growth_note="营收增速 25%~30%（业绩快报）", risk_note="次新股高估值，解禁在即",
        extra_facts={"发行价(元)": 22.6},
    ),
    # ---- 电力/能源（real：快报披露，见审计报告 600131 冒烟）----
    "600131": GoldenStock(
        symbol="600131", name="国网信通", sector="电力能源", industry="电力,软件服务",
        source="real", price=18.5, total_shares=1.2e9, bps=4.2,
        annual_net=6.2e8, q_last=1.9e8, q_prev=1.6e8,
        pe=32.0, pb=4.4, fair_value=20.0, safe_price=16.0,
        growth_note="快报营收34.33亿(-2.6%)，扣非2.29亿(+11.2%)，拐点改善",
        risk_note="应收账款占比高，政企IT季节性",
    ),
    # ---- 半导体（synthetic 构造）----
    "600584": GoldenStock(
        symbol="600584", name="长电科技", sector="半导体", industry="半导体,封测",
        source="synthetic", price=35.0, total_shares=1.79e9, bps=12.5,
        annual_net=18.0e8, q_last=6.5e8, q_prev=4.5e8,
        pe=28.0, pb=2.8, fair_value=32.0, safe_price=27.0,
        growth_note="AI 先进封装需求驱动，扣非增速 30%+", risk_note="重资产周期，CAPEX 高",
    ),
    # ---- 医药（synthetic 构造）----
    "600276": GoldenStock(
        symbol="600276", name="恒瑞医药", sector="医药", industry="医药,创新药",
        source="synthetic", price=52.0, total_shares=6.4e9, bps=7.2,
        annual_net=48.0e8, q_last=15.0e8, q_prev=12.0e8,
        pe=45.0, pb=7.2, fair_value=48.0, safe_price=40.0,
        growth_note="创新药管线放量，研发费用率 20%+", risk_note="集采降价，研发失败风险",
    ),
    # ---- 消费品（synthetic 构造）----
    "600519": GoldenStock(
        symbol="600519", name="贵州茅台", sector="消费品", industry="白酒,食品饮料",
        source="synthetic", price=1500.0, total_shares=1.256e9, bps=210.0,
        annual_net=760.0e8, q_last=220.0e8, q_prev=200.0e8,
        pe=22.0, pb=7.1, fair_value=1600.0, safe_price=1350.0,
        growth_note="高端白酒量价稳增，ROE 30%+", risk_note="渠道库存周期，消费景气",
    ),
    # ---- 通信/光模块（synthetic 构造）----
    "300502": GoldenStock(
        symbol="300502", name="新易盛", sector="通信光模块", industry="光模块,通信",
        source="synthetic", price=120.0, total_shares=7.1e8, bps=15.0,
        annual_net=32.0e8, q_last=12.0e8, q_prev=6.0e8,
        pe=35.0, pb=8.0, fair_value=110.0, safe_price=90.0,
        growth_note="800G/1.6T 光模块放量，营收增速 80%+", risk_note="客户集中度高，技术路线迭代",
    ),
    # ---- 制造/新能源（synthetic 构造）----
    "300750": GoldenStock(
        symbol="300750", name="宁德时代", sector="制造新能源", industry="新能源,锂电",
        source="synthetic", price=250.0, total_shares=4.4e9, bps=85.0,
        annual_net=480.0e8, q_last=130.0e8, q_prev=100.0e8,
        pe=23.0, pb=2.9, fair_value=260.0, safe_price=220.0,
        growth_note="动力电池全球份额第一，储能高增", risk_note="产能过剩价格战，锂价波动",
    ),
    # ---- 金融（synthetic 构造）----
    "600036": GoldenStock(
        symbol="600036", name="招商银行", sector="金融", industry="银行,金融",
        source="synthetic", price=40.0, total_shares=2.52e10, bps=38.0,
        annual_net=1500.0e8, q_last=380.0e8, q_prev=360.0e8,
        pe=6.5, pb=1.05, fair_value=42.0, safe_price=35.0,
        growth_note="零售银行龙头，ROE 15%，拨备充足", risk_note="息差收窄，地产敞口",
    ),
    # ---- 地产/建筑（synthetic 构造）----
    "600048": GoldenStock(
        symbol="600048", name="保利发展", sector="地产建筑", industry="地产,建筑",
        source="synthetic", price=9.0, total_shares=1.2e10, bps=16.0,
        annual_net=120.0e8, q_last=25.0e8, q_prev=30.0e8,
        pe=8.0, pb=0.56, fair_value=10.0, safe_price=7.5,
        growth_note="央企地产龙头，销售回款改善", risk_note="行业下行，去化周期长，净负债率高",
    ),
}


# 按板块归类（用于前端选择器）
SECTORS: list[str] = list(dict.fromkeys(s.sector for s in GOLDEN_STOCKS.values()))


def get_stock(symbol: str) -> GoldenStock | None:
    return GOLDEN_STOCKS.get(symbol)


def get_symbols_by_source(source: str) -> list[str]:
    return [s for s, g in GOLDEN_STOCKS.items() if g.source == source]


def get_real_symbols() -> list[str]:
    return get_symbols_by_source("real")


def build_golden_collected(symbol: str) -> str:
    """构造某只股票的黄金 collected_data（Layer B 固定输入）。"""
    import json

    g = GOLDEN_STOCKS.get(symbol)
    if not g:
        raise KeyError(f"黄金股票池无 {symbol}")

    profit = [
        {"报告期": "一季报", "归母净利润": g.q_last, "扣非净利润": None},
        {"报告期": "年报", "归母净利润": g.annual_net, "扣非净利润": None},
        {"报告期": "一季报", "归母净利润": g.q_prev, "扣非净利润": None},
    ]
    valuation = [
        {"总股本": g.total_shares, "每股净资产": g.bps, "每股收益": round(g.ttm_eps, 3),
         "报告期": "2026-06-30 [半年报快报]", "ROE(%)": g.roe, "资产负债率(%)": 45.0},
    ]
    precomputed = {
        "PE(TTM)": g.pe, "PB": g.pb, "市值(亿)": round(g.market_cap, 1),
        "TTM EPS": round(g.ttm_eps, 3), "合理价值(元)": g.fair_value,
        "安全买入价(元)": g.safe_price,
    }
    facts_text = (
        f"{g.name}（{g.symbol}）当前价{g.price}元；TTM EPS {g.ttm_eps:.3f}元；"
        f"PE(TTM) {g.pe}倍；PB {g.pb}；市值{g.market_cap:.1f}亿；ROE {g.roe}%；"
        f"合理价值{g.fair_value}元，安全买入价{g.safe_price}元；"
        f"情景概率：悲观25%、基准55%、乐观20%；{g.growth_note}。"
    )
    if g.extra_facts:
        facts_text += " ".join(f"{k}{v}元" for k, v in g.extra_facts.items()) + "；"

    stock = {
        "代码": g.symbol,
        "名称": g.name,
        "行业": g.sector,
        "行情": {"当前价格": g.price, "name": g.name, "涨跌幅": -1.5},
        "公告": {"列表": [
            {"标题": f"{g.name}：2026年半年度业绩快报", "日期": "2026-07-17",
             "快报数据": {"报告期": "2026-06-30 [半年报快报]", "归母净利润": 0.5e8}},
        ]},
        "财报": {"利润表": profit, "现金流": []},
        "估值": {"data": valuation},
        "预计算分数": {"计算总分": {"得分": 60.0, "依据": "确定性计算"}},
        "市场情绪": {"label": "中性", "score": 50},
        "参考计算值": precomputed,
        "公告正文": facts_text,
    }
    collected = json.dumps([stock], ensure_ascii=False, indent=2)
    return f"[INDUSTRY] {g.industry}\n[TOOLS] 财报(✅) 估值(✅) 行情(✅) 行业(✅) 公告(✅) 评分(✅) 情绪(✅)\n" + collected


def build_golden_facts(symbol: str) -> list[tuple[str, float, float]]:
    """构造某只股票的黄金事实（Layer B 命中检查）：[(名称, 数值, 容差)]。"""
    g = GOLDEN_STOCKS.get(symbol)
    if not g:
        raise KeyError(f"黄金股票池无 {symbol}")
    facts = [
        ("TTM EPS", round(g.ttm_eps, 3), 0.05),
        ("PE(TTM)", g.pe, 0.05),
        ("PB", g.pb, 0.05),
        ("市值(亿)", round(g.market_cap, 1), 0.05),
        ("当前价", g.price, 0.01),
        ("合理价值", g.fair_value, 0.15),
    ]
    for k, v in g.extra_facts.items():
        facts.append((k, v, 0.05))
    return facts


def build_golden_data_facts(symbol: str) -> dict:
    """构造某只股票的 Layer A 黄金事实（数据管道比对）。"""
    g = GOLDEN_STOCKS.get(symbol)
    if not g:
        raise KeyError(f"黄金股票池无 {symbol}")
    return {
        "总股本(股)": g.total_shares,
        "每股净资产(元)": g.bps,
        "2025年报归母(元)": g.annual_net,
        "2026Q1归母(元)": g.q_last,
        "TTM归母(元)": g.ttm_net,
        "TTM EPS": round(g.ttm_eps, 3),
        "最新报告期": ["半年报", "一季报"],
    }
