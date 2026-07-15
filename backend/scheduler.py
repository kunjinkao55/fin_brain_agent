"""
FinBrain 数据调度器 — 定时预取数据到缓存，API 请求毫秒级返回。
独立进程运行: python backend/scheduler.py
"""

import os, sys, time, signal, logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finbrain.scheduler")

os.environ["FINBRAIN_DATA_MODE"] = "local"  # 调度器强制 local 模式

from backend.tools import (
    fetch_stock_price, get_financial_statements, get_valuation,
    get_industry_info, get_fund_flow, get_limit_up_pool,
    get_market_breadth, get_sector_fund_flow, get_concept_ranking,
    get_dragon_tiger_list, get_intraday,
)
from backend.cache import set as cache_set, stats as cache_stats

# 热门标的列表（可配置）
_WATCHLIST = [
    "000001", "000002", "000651", "000858", "002415", "002594",
    "300502", "300750", "600036", "600276", "600519", "600795",
    "601012", "601088", "601166", "601318", "601991",
]

_running = True


def _is_trading_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 5  # 9:15-15:05


def _fetch_with_cache(name, key, func, *args):
    try:
        result = func(*args)
        cache_set(name, key, result)
        return True
    except Exception as e:
        logger.warning("%s(%s) failed: %s", name, key, e)
        return False


def job_fast():
    """高频任务 — 每 30s"""
    if not _is_trading_hours():
        return
    for code in _WATCHLIST:
        _fetch_with_cache("stock_price", code, fetch_stock_price, code)


def job_mid():
    """中频任务 — 每 5min (交易时段)"""
    if not _is_trading_hours():
        return
    logger.info("Mid-frequency refresh: limit_up, fund_flow, sector_flow...")
    _fetch_with_cache("limit_up_pool", "50", get_limit_up_pool, 50)
    _fetch_with_cache("market_breadth", "all", get_market_breadth)
    _fetch_with_cache("sector_fund_flow", "100", get_sector_fund_flow, 100)
    _fetch_with_cache("concept_ranking", "30", get_concept_ranking, 30)
    _fetch_with_cache("dragon_tiger_list", "today", get_dragon_tiger_list, "")
    for code in _WATCHLIST[:5]:  # Top 5 only
        _fetch_with_cache("fund_flow", code, get_fund_flow, code)
        _fetch_with_cache("intraday", code, get_intraday, code)


def job_slow():
    """低频任务 — 每 30min"""
    logger.info("Slow refresh: financials, valuation, industry...")
    for code in _WATCHLIST:
        _fetch_with_cache("financial_statements", code, get_financial_statements, code)
        _fetch_with_cache("valuation", code, get_valuation, code)
        _fetch_with_cache("industry_info", code, get_industry_info, code)


def job_daily():
    """盘后任务 — 15:30"""
    logger.info("EOD snapshot")
    _fetch_with_cache("sector_fund_flow", "eod_100", get_sector_fund_flow, 100)
    _fetch_with_cache("market_breadth", "eod", get_market_breadth)
    logger.info("Cache stats: %s", cache_stats())


def run():
    logger.info("Scheduler started. Cache mode: %s", os.getenv("FINBRAIN_CACHE_MODE", "local"))
    last_fast = last_mid = last_slow = last_daily = 0

    while _running:
        now = time.time()
        try:
            if now - last_fast >= 30:
                job_fast()
                last_fast = now
            if now - last_mid >= 300:
                job_mid()
                last_mid = now
            if now - last_slow >= 1800:
                job_slow()
                last_slow = now
            # EOD
            dt = datetime.now()
            if dt.hour == 15 and dt.minute >= 30 and now - last_daily > 3600:
                job_daily()
                last_daily = now
            time.sleep(5)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Scheduler error: %s", e)
            time.sleep(10)

    logger.info("Scheduler stopped.")


def stop():
    global _running
    _running = False


signal.signal(signal.SIGINT, lambda *_: stop())
signal.signal(signal.SIGTERM, lambda *_: stop())

if __name__ == "__main__":
    run()
