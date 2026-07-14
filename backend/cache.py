"""
工具调用缓存 — 减少重复 API 请求，加速响应，降低 token 消耗。

缓存策略（TTL）：
- 实时行情: 30秒
- K线数据:   5分钟
- 财报/估值: 30分钟（日频更新）
- 行业信息:  1小时
- 资金流向:  5分钟
- 龙虎榜:    5分钟

线程安全：threading.Lock 保护共享 dict。
"""

import time, threading

_cache = {}
_lock = threading.Lock()

# TTL 配置（秒）
TTL = {
    "stock_price": 30,
    "stock_history": 300,
    "financial_statements": 1800,
    "valuation": 1800,
    "industry_info": 3600,
    "fund_flow": 300,
    "screen_stocks": 600,
    "limit_up_pool": 60,
    "concept_ranking": 600,
    "dragon_tiger_list": 300,
    "dragon_tiger_detail": 300,
    "search_youzi": 3600,
    "calculate_score": 300,
}


def get(tool_name: str, key: str) -> dict | None:
    """获取缓存结果。key 通常是股票代码或查询参数。返回 None 表示未命中。"""
    cache_key = f"{tool_name}:{key}"
    with _lock:
        entry = _cache.get(cache_key)
        if entry and time.time() - entry["ts"] < TTL.get(tool_name, 60):
            return entry["data"]
    return None


def set(tool_name: str, key: str, data: dict):
    """写入缓存"""
    cache_key = f"{tool_name}:{key}"
    with _lock:
        _cache[cache_key] = {"ts": time.time(), "data": data}
        # 简单淘汰：超过1000条清最旧的
        if len(_cache) > 1000:
            oldest = min(_cache, key=lambda k: _cache[k]["ts"])
            del _cache[oldest]


def clear():
    """清空全部缓存"""
    with _lock:
        _cache.clear()


def stats() -> dict:
    """缓存统计"""
    with _lock:
        return {"entries": len(_cache),
                "oldest_ts": min((e["ts"] for e in _cache.values()), default=0)}
