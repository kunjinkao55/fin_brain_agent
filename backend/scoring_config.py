"""
FinBrain 评分配置加载器 — 从 configs/scoring.json 读取，带默认值兜底。
所有阈值、权重、行业PE集中在此管理，tools.py 和 scoring.py 统一调用。
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "scoring.json")
_CONFIG = None


def _load_config() -> dict:
    """加载评分配置，解析失败返回空dict。自动校验必填section。"""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
        # Harness: 校验必填section
        required = ["盈利能力", "成长性", "财务健康", "估值合理", "动态权重", "安全边际"]
        missing = [s for s in required if s not in _CONFIG]
        if missing:
            logger.warning("scoring.json missing sections: %s. Using defaults for them.", missing)
        else:
            logger.info("Loaded scoring config from %s (%d sections)", _CONFIG_PATH, len(_CONFIG))
    except Exception as e:
        logger.warning("Failed to load scoring.json: %s. Using defaults.", e)
        _CONFIG = {}
    return _CONFIG


def reload_config():
    """强制重新加载（Settings 导入新配置后调用）"""
    global _CONFIG
    _CONFIG = None
    return _load_config()


# ---- 便捷访问函数（每处调用自带默认值，配置文件缺失不影响运行）----

def get_profitability() -> dict:
    cfg = _load_config().get("盈利能力", {})
    return {
        "roe_thresholds": cfg.get("ROE阈值", {"S": 50, "A": 30, "B": 20, "C": 10, "D": 5}),
        "roe_scores": cfg.get("ROE分值", {"S": 10, "A": 9, "B": 7, "C": 5, "D": 3, "E": 1}),
        "gm_bonus_threshold": cfg.get("毛利率加分阈值", 40),
        "gm_bonus": cfg.get("毛利率加分值", 1),
        "nm_bonus_threshold": cfg.get("净利率加分阈值", 20),
        "nm_bonus": cfg.get("净利率加分值", 1),
    }


def get_growth() -> dict:
    cfg = _load_config().get("成长性", {})
    return {
        "rev_thresholds": cfg.get("营收增速阈值", {"S": 100, "A": 50, "B": 30, "C": 20, "D": 10, "E": 0}),
        "rev_scores": cfg.get("营收增速分值", {"S": 10, "A": 9, "B": 8, "C": 6, "D": 5, "E": 3, "F": 0}),
        "rev_weight": cfg.get("营收权重", 0.4),
        "deduct_weight": cfg.get("扣非权重", 0.6),
        "trend_adj": cfg.get("趋势修正", {"加速": 2, "延续": 0, "放缓": -1, "拐点恶化": -3, "拐点改善": 3}),
    }


def get_financial_health() -> dict:
    cfg = _load_config().get("财务健康", {})
    return {
        "debt_thresholds": cfg.get("负债率阈值", {"S": 30, "A": 50, "B": 60}),
        "debt_scores": cfg.get("负债率分值", {"S": 10, "A": 7, "B": 5, "C": 3}),
        "cf_bonus_heavy": cfg.get("现金流加分", {}).get("重资产行业", {"阈值": 2.0, "分值": 2, "次阈值": 1.0, "次分值": 1}),
        "cf_bonus_light": cfg.get("现金流加分", {}).get("轻资产行业", {"阈值": 0.8, "分值": 2}),
        "dep_threshold": cfg.get("现金流加分", {}).get("折旧判定阈值", 0.5),
    }


def get_valuation() -> dict:
    cfg = _load_config().get("估值合理", {})
    return {
        "pe_ratio_thresholds": cfg.get("PE行业比值阈值", {"S": 0.6, "A": 0.9, "B": 1.2, "C": 1.6}),
        "pe_ratio_scores": cfg.get("PE行业比值分值", {"S": 10, "A": 8, "B": 6, "C": 4, "D": 2}),
        "default_ind_pe": cfg.get("默认行业PE", 18),
        "industry_pe": cfg.get("行业PE基准", {}),
    }


def get_weights(company_type: str = "默认") -> dict:
    cfg = _load_config().get("动态权重", {})
    return cfg.get(company_type, cfg.get("默认", {
        "商业模式": 0.15, "竞争优势": 0.15, "行业周期": 0.15,
        "财务质量": 0.20, "成长质量": 0.15, "估值": 0.20,
    }))


def get_safety_margin(company_type: str = "价值型") -> float:
    cfg = _load_config().get("安全边际", {})
    base = cfg.get("基准", {}).get(company_type, 0.25)
    return base


def get_safety_adjustments() -> dict:
    cfg = _load_config().get("安全边际", {})
    return {
        "roe_high": cfg.get("ROE调整", {}).get("高ROE阈值", 20),
        "roe_high_adj": cfg.get("ROE调整", {}).get("高ROE调整", -0.05),
        "roe_low": cfg.get("ROE调整", {}).get("低ROE阈值", 5),
        "roe_low_adj": cfg.get("ROE调整", {}).get("低ROE调整", 0.05),
        "debt_high": cfg.get("负债率调整", {}).get("高负债阈值", 70),
        "debt_high_adj": cfg.get("负债率调整", {}).get("高负债调整", 0.05),
        "debt_low": cfg.get("负债率调整", {}).get("低负债阈值", 20),
        "debt_low_adj": cfg.get("负债率调整", {}).get("低负债调整", -0.03),
        "limits": cfg.get("限制范围", [0.15, 0.55]),
    }


def get_roe_multipliers() -> dict:
    cfg = _load_config().get("ROE质量乘数", {})
    return {
        "thresholds": cfg.get("阈值", [35, 25, 18, 12, 8, 3]),
        "multipliers": cfg.get("乘数", [1.6, 1.3, 1.1, 0.85, 0.65, 0.45, 0.30]),
        "debt_high": cfg.get("负债率修正", {}).get("高负债阈值", 70),
        "debt_high_adj": cfg.get("负债率修正", {}).get("高负债调整", -0.10),
        "debt_mid": cfg.get("负债率修正", {}).get("中负债阈值", 50),
        "debt_mid_adj": cfg.get("负债率修正", {}).get("中负债调整", -0.05),
        "debt_low": cfg.get("负债率修正", {}).get("低负债阈值", 20),
        "debt_low_adj": cfg.get("负债率修正", {}).get("低负债调整", 0.05),
    }


def get_rating_thresholds() -> dict:
    cfg = _load_config().get("评级阈值", {})
    return {
        "sell_max": cfg.get("SELL分数上限", 40),
        "hold_min": cfg.get("HOLD分数下限", 40),
        "high_position": cfg.get("高仓位分数", 60),
        "high_conf": cfg.get("高置信度分数", 75),
        "mid_conf": cfg.get("中置信度分数", 55),
    }


def get_valuation_guards() -> dict:
    """估值守卫参数（估值层全部可调阈值，禁止硬编码在业务代码里）。

    默认值与 2026-07-26 外置时一致——配置缺失时行为不变。
    注意：交易所涨跌停规则（trading_rules.py）属监管事实常量，不在此管理。"""
    cfg = _load_config().get("估值守卫", {})
    return {
        "scenario_eps_band": cfg.get("情景EPS带",
            {"悲观": [0.3, 1.0], "基准": [0.5, 1.3], "乐观": [0.7, 2.0]}),
        "pe_band": cfg.get("情景PE带", {
            "基准下限": 0.8, "基准上限": 1.5,
            "悲观对基准": [0.6, 0.8], "悲观对中枢下限": 0.5,
            "乐观对基准下限": 1.1, "乐观对中枢上限": 1.5}),
        "scenario_fallback": cfg.get("情景兜底模板", {
            "悲观": {"EPS": 0.85, "PE": 0.6, "PE下限": 10, "概率": "20%"},
            "基准": {"EPS": 1.0, "PE": 1.0, "概率": "60%"},
            "乐观": {"EPS": 1.15, "PE": 1.2, "PE上限加": 15, "概率": "20%"}}),
        "growth_pe_mult": cfg.get("成长溢价", {"S": 1.8, "A": 1.3, "B": 1.1, "C": 0.7}),
        "cf_floor": cfg.get("现金流抬底", {"一档": 0.7, "二档": 0.5}),
        "matrix": cfg.get("乘数矩阵", {
            "增速高档": 30, "增速低档": 10, "毛利率档pp": 0.5,
            "资本开支高档": 0.5, "资本开支低档": 1.0,
            "高": 1.5, "中": 1.0, "低": 0.6}),
        "sotp": cfg.get("SOTP",
            {"增速差pp": 30, "毛利率差pp": 15, "锚定毛利占比": 0.40, "总部折价": 0.10}),
        "breakers": cfg.get("一致性断路器", {
            "INV1下限容差": 0.02, "INV1上限倍数": 1.1, "INV3增速偏差pp": 0.05,
            "INV6广度阈值": 30, "INV7增速阈值": 30,
            "INV9现金流倍数": 10, "INV9_ROIC阈值": 0.15, "INV10周期阈值": 0.4}),
        "verify": cfg.get("回溯验证",
            {"默认容差": 0.01, "锚点容差": 0.05, "跨源容差": 0.02, "grounding容差": 0.05}),
        "trend": cfg.get("趋势参考", {"支撑缓冲": 1.05, "现价折扣阈值": 0.9}),
        "pe_band_hist": cfg.get("历史PE区间", {"最小K线数": 240, "最小时点数": 3}),
        "new_listing_months": cfg.get("次新股月数", 12),
        "gap_conclusion_pp": cfg.get("预期差结论阈值pp", 30),
    }
