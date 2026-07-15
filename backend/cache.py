"""
工具调用缓存 — 减少重复 API 请求。支持 local(内存) / redis(共享) 双模式。

线程安全：local 用 threading.Lock，redis 自带原子操作。
"""

import os, time, threading, json, logging

logger = logging.getLogger(__name__)

_CACHE_MODE = os.getenv("FINBRAIN_CACHE_MODE", "local")
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# TTL 配置（秒）
TTL = {
    "stock_price": 30,       "stock_history": 300,
    "financial_statements": 1800, "valuation": 1800,
    "industry_info": 3600,   "fund_flow": 300,
    "screen_stocks": 600,    "limit_up_pool": 60,
    "concept_ranking": 600,  "dragon_tiger_list": 300,
    "dragon_tiger_detail": 300, "search_youzi": 3600,
    "calculate_score": 300,
}

# ---- Redis 后端 ----
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis
        _redis = redis.from_url(_REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Redis connected: %s", _REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable (%s), falling back to local cache", e)
        _redis = False
    return _redis


def get(tool_name: str, key: str):
    cache_key = f"finbrain:{tool_name}:{key}"
    ttl = TTL.get(tool_name, 60)

    if _CACHE_MODE == "redis":
        r = _get_redis()
        if r:
            try:
                raw = r.get(cache_key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return None

    # Local mode
    with _lock:
        entry = _local_cache.get(cache_key)
        if entry and time.time() - entry["ts"] < ttl:
            return entry["data"]
    return None


def set(tool_name: str, key: str, data):
    cache_key = f"finbrain:{tool_name}:{key}"
    ttl = TTL.get(tool_name, 60)

    if _CACHE_MODE == "redis":
        r = _get_redis()
        if r:
            try:
                r.setex(cache_key, ttl, json.dumps(data, ensure_ascii=False, default=str))
                return
            except Exception:
                pass

    # Local mode
    with _lock:
        _local_cache[cache_key] = {"ts": time.time(), "data": data}
        if len(_local_cache) > 1000:
            oldest = min(_local_cache, key=lambda k: _local_cache[k]["ts"])
            del _local_cache[oldest]


def clear():
    if _CACHE_MODE == "redis":
        r = _get_redis()
        if r:
            try:
                for k in r.keys("finbrain:*"):
                    r.delete(k)
            except Exception:
                pass
    with _lock:
        _local_cache.clear()


def stats():
    if _CACHE_MODE == "redis":
        r = _get_redis()
        if r:
            try:
                return {"mode": "redis", "keys": r.dbsize()}
            except Exception:
                pass
    with _lock:
        return {"mode": "local", "entries": len(_local_cache)}


# ---- Local 后端（保留） ----
_local_cache = {}
_lock = threading.Lock()
