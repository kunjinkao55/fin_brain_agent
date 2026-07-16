"""
数据源分层系统 — 支持从免费API到付费终端的渐进式升级。
每个高级数据源都是一个可插拔的"插槽"：配置了就用，没配置就优雅降级。

用法:
    from backend.datasource_tier import tier, require_tier, premium_slot

    # 检查当前等级
    if tier >= DataSourceTier.PREMIUM:
        data = fetch_from_wind()

    # 或用装饰器：等级不够时跳过并返回占位符
    @premium_slot(fallback="公开信息不足，需付费数据源支持")
    def get_management_background(symbol: str) -> dict:
        ...
"""

import os
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Optional


class DataSourceTier(IntEnum):
    """数据源等级。数值越大能力越强，高级别包含低级别所有能力。"""
    FREE = 0           # 免费API：新浪/东方财富/同花顺 — 行情+财报+公告
    PREMIUM = 1        # 付费终端：Wind/Choice/iFind — +管理层+机构持仓+行业深度
    INSTITUTIONAL = 2  # 机构级：Bloomberg/FactSet/CAPIQ — +实时Level2+另类数据+供应链


# ---- 全局等级标记 ----
_tier_name = os.getenv("FINBRAIN_DATA_TIER", "FREE").upper()
try:
    tier: DataSourceTier = DataSourceTier[_tier_name]
except KeyError:
    tier = DataSourceTier.FREE
    print(f"[datasource_tier] 未知等级 '{_tier_name}'，回退为 FREE")

# 数据源连接状态
_source_status: dict[str, bool] = {}


def set_tier(new_tier: DataSourceTier):
    """运行时切换数据源等级（用于前端设置面板）。"""
    global tier
    tier = new_tier


def register_source(name: str, connected: bool):
    """注册一个高级数据源的连接状态。"""
    _source_status[name] = connected


def is_source_available(name: str) -> bool:
    """检查指定高级数据源是否已连接。"""
    return _source_status.get(name, False)


# ---- 插槽装饰器 ----
def premium_slot(fallback: Any = None, required_tier: DataSourceTier = DataSourceTier.PREMIUM):
    """
    高级数据插槽装饰器。
    当前等级 >= required_tier 时正常执行函数；
    等级不足时跳过函数体，直接返回 fallback。
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if tier < required_tier:
                return fallback
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_tier(min_tier: DataSourceTier) -> bool:
    """检查当前等级是否满足最低要求。"""
    return tier >= min_tier


# ---- 预定义的高级插槽函数（当前为空实现，接入数据源后替换）----

@premium_slot(fallback=None)
def get_management_profile(symbol: str) -> Optional[dict]:
    """
    [PREMIUM] 管理层画像 — 实控人背景、核心团队履历、历史信披与分红记录。
    数据源: Wind 终端 / Choice  / 企查查API
    返回: {"实控人": str, "实控人类型": str, "核心团队": list, "信披评级": str, "分红连续性": str}
    """
    # TODO: 接入 Wind/Choice API
    raise NotImplementedError("管理层数据需要配置 Wind/Choice/企查查API")


@premium_slot(fallback=None)
def get_institutional_holdings(symbol: str) -> Optional[dict]:
    """
    [PREMIUM] 机构持仓详情 — 基金/社保/外资持仓变动、大宗交易明细。
    数据源: Wind 终端 / 东方财富Choice
    返回: {"机构持仓比例": float, "近季度变动": list, "大宗交易": list}
    """
    # TODO: 接入 Wind/Choice API
    raise NotImplementedError("机构持仓数据需要配置 Wind/Choice API")


@premium_slot(fallback=None)
def get_industry_supply_chain(symbol: str) -> Optional[dict]:
    """
    [PREMIUM] 产业链图谱 — 上游供应商集中度、下游客户集中度、替代品威胁评估。
    数据源: Wind 产业链 / Bloomberg SPLC
    返回: {"上游集中度": str, "下游客户": list, "替代品风险": str}
    """
    # TODO: 接入 Wind/Bloomberg API
    raise NotImplementedError("产业链数据需要配置 Wind/Bloomberg API")


@premium_slot(fallback=None)
def get_esg_and_governance(symbol: str) -> Optional[dict]:
    """
    [INSTITUTIONAL] ESG与治理深度数据 — MSCI ESG评级、董事会独立性、关联交易监控。
    数据源: MSCI ESG / Bloomberg / 商道融绿
    返回: {"ESG评级": str, "董事会独立性": str, "关联交易": list}
    """
    # TODO: 接入 Bloomberg/MSCI API
    raise NotImplementedError("ESG数据需要配置 Bloomberg/MSCI API")


@premium_slot(fallback=None)
def get_alternative_data(symbol: str) -> Optional[dict]:
    """
    [INSTITUTIONAL] 另类数据 — 卫星图像/供应链物流/电商爬虫/舆情情绪。
    数据源: Quandl / Thinknum / 自定义爬虫
    返回: {"数据来源": str, "信号": list, "时效": str}
    """
    # TODO: 接入另类数据提供商
    raise NotImplementedError("另类数据需要配置 Quandl/Thinknum/爬虫")


# ---- 插槽注册表 ----
PREMIUM_SLOTS = {
    "管理层画像": get_management_profile,
    "机构持仓": get_institutional_holdings,
    "产业链图谱": get_industry_supply_chain,
    "ESG与治理": get_esg_and_governance,
    "另类数据": get_alternative_data,
}


def query_premium_slot(slot_name: str, symbol: str) -> Optional[dict]:
    """统一的高级插槽查询入口。等级不足或未配置时返回 None。"""
    func = PREMIUM_SLOTS.get(slot_name)
    if func is None:
        return None
    return func(symbol)


# ---- 启动时报告 ----
def _report_tier():
    """启动时打印当前数据源等级和能力。"""
    capabilities = {
        DataSourceTier.FREE: "行情+财报+公告(K线/三大报表/估值/行业分类/资金流/涨停池/龙虎榜)",
        DataSourceTier.PREMIUM: "+管理层画像+机构持仓+产业链图谱+深度行业数据",
        DataSourceTier.INSTITUTIONAL: "+ESG治理+另类数据+供应链+实时Level2",
    }
    active_slots = [name for name, func in PREMIUM_SLOTS.items() if func() is not None]
    missing_slots = [name for name, func in PREMIUM_SLOTS.items() if func() is None]

    print(f"[FinBrain] 数据源等级: {tier.name} | 能力: {capabilities.get(tier, '?')}")
    if active_slots:
        print(f"[FinBrain] 已激活高级插槽: {', '.join(active_slots)}")
    if missing_slots and tier >= DataSourceTier.PREMIUM:
        print(f"[FinBrain] 未配置高级插槽: {', '.join(missing_slots)} (升级到对应等级后需手动配置)")


_report_tier()
