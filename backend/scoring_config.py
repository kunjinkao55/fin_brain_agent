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
    """加载评分配置，解析失败返回空dict（调用方用默认值兜底）"""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
        logger.info("Loaded scoring config from %s", _CONFIG_PATH)
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
    base = cfg.get("基准", {}).get(company_type, 0.30)
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
