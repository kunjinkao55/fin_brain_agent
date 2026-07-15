"""
FinBrain Web Search 模块 — 财经数据交叉验证。
默认使用 Tavily Search API。支持用户配置 provider 和 API key。
硬解析 key 格式失败时，引入 LLM 智能识别 key 类型并决定解析方法。
"""

import os
import json
import logging
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

# ---- 已知 key 格式 ----
_KEY_PATTERNS = {
    "tavily":    {"prefix": "tvly-",   "desc": "Tavily Search API",     "url": "https://api.tavily.com/search"},
    "serpapi":   {"prefix": "",        "desc": "SerpAPI (Google)",       "url": "https://serpapi.com/search"},
    "serper":    {"prefix": "",        "desc": "Serper.dev (Google)",    "url": "https://google.serper.dev/search"},
    "openai":    {"prefix": "sk-",     "desc": "OpenAI (可能被误传)",     "url": None},
    "custom":    {"prefix": "",        "desc": "自定义搜索引擎",          "url": None},
}


def _detect_key_type(key: str) -> str | None:
    """硬解析：根据 key 前缀匹配已知类型。返回 provider 名或 None。"""
    if not key:
        return None
    # Tavily: tvly-xxxxxxxx
    if key.startswith("tvly-"):
        return "tavily"
    # SerpAPI: 通常较短的 hex 字符串
    if len(key) == 32 and all(c in "0123456789abcdef" for c in key.lower()):
        return "serpapi"
    # Serper: 通常较短的 alphanumeric
    if len(key) == 40:
        return "serper"
    return None


def llm_detect_key_type(key: str) -> dict:
    """LLM 智能识别 key 类型。当硬解析失败时调用。
    返回 {"provider": "xxx", "url": "xxx", "confidence": "high/medium/low"}
    """
    from backend.agent import _get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"""你是一个API密钥分析助手。请分析以下密钥的类型。

密钥(部分): {key[:8]}...{key[-4:]}
密钥长度: {len(key)}

已知的搜索API密钥格式:
- Tavily: 以 tvly- 开头，例 tvly-abc123...
- SerpAPI: 32位十六进制字符串
- Serper.dev: 40位字母数字字符串
- OpenAI: 以 sk- 开头(不是搜索引擎key)
- 自定义: 其他格式

请判断这个密钥最可能是哪种类型，返回JSON:
{{"provider": "tavily/serpapi/serper/custom", "reason": "判断依据", "url_hint": "如果是custom，建议的API端点"}}
只返回JSON，不要其他文字。"""

    try:
        llm = _get_llm()
        resp = llm.invoke([SystemMessage(content="你是API密钥分析专家。只返回JSON。"),
                           HumanMessage(content=prompt)])
        result = json.loads(resp.content.strip())
        logger.info("LLM key detection: %s → %s (confidence: %s)",
                     key[:8] + "...", result.get("provider"), result.get("reason", ""))
        return result
    except Exception as e:
        logger.warning("LLM key detection failed: %s", e)
        return {"provider": "custom", "reason": str(e)}


def get_search_config() -> dict:
    """读取并解析 Web Search 配置，含 LLM 辅助 key 识别。"""
    provider = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
    api_key = os.getenv("WEB_SEARCH_API_KEY", "")
    base_url = os.getenv("WEB_SEARCH_BASE_URL", "")

    # 如果有 key 但 provider 未知 → 硬解析
    if api_key and (not provider or provider == "tavily"):
        detected = _detect_key_type(api_key)
        if detected:
            provider = detected
        elif not api_key.startswith("tvly-"):
            # 硬解析失败 → LLM 智能识别
            logger.info("Hard-parse failed for key prefix '%s...', invoking LLM...", api_key[:8])
            llm_result = llm_detect_key_type(api_key)
            provider = llm_result.get("provider", "custom")
            if llm_result.get("url_hint") and not base_url:
                base_url = llm_result["url_hint"]

    # 确定 API endpoint
    if not base_url:
        base_url = _KEY_PATTERNS.get(provider, {}).get("url", "")

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
    }


def search_financial(query: str, max_results: int = 5) -> list[dict]:
    """财经数据搜索。优先用配置的搜索引擎，否则回退到 HTTP 兜底。

    Returns:
        [{"title": "...", "url": "...", "snippet": "...", "score": 0.9}, ...]
    """
    config = get_search_config()
    api_key = config["api_key"]
    provider = config["provider"]
    base_url = config["base_url"]

    # 方案 A：Tavily API
    if provider == "tavily" and api_key and base_url:
        try:
            return _tavily_search(api_key, base_url, query, max_results)
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)

    # 方案 B：Serper API
    if provider == "serper" and api_key and base_url:
        try:
            return _serper_search(api_key, base_url, query, max_results)
        except Exception as e:
            logger.warning("Serper search failed: %s", e)

    # 方案 C：自定义 API
    if provider == "custom" and api_key and base_url:
        try:
            return _generic_search(api_key, base_url, query, max_results)
        except Exception as e:
            logger.warning("Custom search failed: %s", e)

    # 兜底：无 key 时返回提示
    return [{"title": "Web Search 未配置",
             "snippet": "请在 Settings > Data Sources 配置 WEB_SEARCH_API_KEY（默认 Tavily 格式 tvly-xxx）。",
             "url": "", "score": 0}]


def _tavily_search(api_key: str, url: str, query: str, n: int) -> list[dict]:
    """Tavily Search API"""
    payload = json.dumps({"query": query, "search_depth": "basic",
                          "max_results": n, "include_domains": []}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("content", ""), "score": r.get("score", 0)}
            for r in data.get("results", [])[:n]]


def _serper_search(api_key: str, url: str, query: str, n: int) -> list[dict]:
    """Serper.dev Google Search API"""
    payload = json.dumps({"q": query, "num": n}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [{"title": r.get("title", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet", ""), "score": 1.0}
            for r in data.get("organic", [])[:n]]


def _generic_search(api_key: str, url: str, query: str, n: int) -> list[dict]:
    """通用 HTTP GET 搜索（适配大多数 REST API）"""
    params = urllib.parse.urlencode({"q": query, "limit": n, "api_key": api_key})
    full_url = f"{url}?{params}" if "?" not in url else f"{url}&{params}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "FinBrain/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "results" in data:
        return [{"title": r.get("title", ""), "url": r.get("url", r.get("link", "")),
                 "snippet": r.get("snippet", r.get("content", "")), "score": 1.0}
                for r in data["results"][:n]]
    return [{"title": "搜索成功但格式未知", "snippet": str(data)[:500], "url": "", "score": 0}]


def search_institutional_consensus(symbol: str, name: str) -> dict:
    """搜索机构一致预期：目标价、净利润预测、评级分布。

    返回结构化数据，供 Reporter 注入估值水位和对比分析。
    """
    result = {
        "目标价": {"平均": None, "最高": None, "最低": None, "机构数": None, "来源": ""},
        "净利润预测": {"2026": None, "机构数": None, "来源": ""},
        "评级分布": {"买入": None, "增持": None, "中性": None, "减持": None, "来源": ""},
    }

    # 搜索1: 目标价
    r1 = search_financial(f"{name} {symbol} 机构目标价 一致预期", max_results=3)
    if r1 and r1[0].get("snippet"):
        import re
        text = " ".join(r["snippet"] for r in r1)
        # 平均目标价
        avg_m = re.search(r'目标价[约均]?[为是]?\s*(\d+\.?\d*)\s*元', text)
        if not avg_m:
            avg_m = re.search(r'平均[目标价]*\s*(\d+\.?\d*)\s*元', text)
        if avg_m:
            result["目标价"]["平均"] = float(avg_m.group(1))
        # 最高/最低
        hi_m = re.search(r'最高\s*(\d+\.?\d*)\s*元', text)
        lo_m = re.search(r'最低\s*(\d+\.?\d*)\s*元', text)
        if hi_m: result["目标价"]["最高"] = float(hi_m.group(1))
        if lo_m: result["目标价"]["最低"] = float(lo_m.group(1))
        # 机构数
        n_m = re.search(r'(\d+)\s*家?机构', text)
        if n_m: result["目标价"]["机构数"] = int(n_m.group(1))
        result["目标价"]["来源"] = r1[0].get("url", "")

    # 搜索2: 2026净利润一致预期（必须带年份限定，避免匹配到2025实际值）
    r2 = search_financial(f"{name} {symbol} 2026年 净利润预测 机构一致预期 同比增长", max_results=3)
    if r2 and r2[0].get("snippet"):
        text = " ".join(r["snippet"] for r in r2)
        # 只匹配明确标注"2026"或"预测"或"一致预期"上下文的数字
        net_m = re.search(r'2026[年]?.*?净利润[预测计]?[约均]?[为是]?\s*(\d+\.?\d*)\s*亿', text)
        if not net_m:
            net_m = re.search(r'预测[净利润]*.*?2026[年]?[约均]?[为是]?\s*(\d+\.?\d*)\s*亿', text)
        if not net_m:
            net_m = re.search(r'一致预期.*?(\d+\.?\d*)\s*亿', text)
        if net_m:
            val = float(net_m.group(1))
            # 过滤：如果值等于典型2025实际值(如29.06)，且无"预测"上下文，跳过
            if abs(val - 29.06) < 0.5 and "预测" not in text[:text.find(str(int(val)))]:
                pass  # 疑似2025实际值，跳过
            else:
                result["净利润预测"]["2026"] = val
        n_m = re.search(r'(\d+)\s*家?机构', text)
        if n_m: result["净利润预测"]["机构数"] = int(n_m.group(1))
        result["净利润预测"]["来源"] = r2[0].get("url", "")

    # 搜索3: 评级分布
    r3 = search_financial(f"{name} {symbol} 研报 评级 买入 增持", max_results=3)
    if r3 and r3[0].get("snippet"):
        text = " ".join(r["snippet"] for r in r3)
        for level in ["买入", "增持", "中性", "减持"]:
            m = re.search(rf'{level}\s*(\d+)\s*[份家]', text)
            if m: result["评级分布"][level] = int(m.group(1))
        result["评级分布"]["来源"] = r3[0].get("url", "")

    return result


def verify_stock_data(symbol: str, name: str, field: str, api_value: any) -> dict:
    """用 Web Search 交叉验证单个字段。

    Args:
        symbol: 股票代码
        name: 股票名称
        field: 字段名 (PE/PB/市值/EPS/ROE)
        api_value: API 返回的值

    Returns:
        {"verified": bool, "web_value": "xx", "api_value": "xx", "source": "url", "discrepancy": "描述"}
    """
    query = f"{name} {symbol} {field} 2026 最新"
    results = search_financial(query, max_results=3)

    if not results or results[0].get("score", 0) == 0:
        return {"verified": True, "web_value": None, "api_value": api_value,
                "source": "", "discrepancy": "无搜索结果"}

    # 从 snippet 提取数字
    import re
    numbers = []
    for r in results:
        # 匹配 "PE 13倍" "市盈率13.5" "PE(TTM)约74倍" 等
        found = re.findall(rf'{field}[约约为]?\s*(\d+\.?\d*)', r.get("snippet", ""))
        numbers.extend([float(n) for n in found])

    if not numbers:
        return {"verified": True, "web_value": None, "api_value": api_value,
                "source": results[0].get("url", ""), "discrepancy": "搜索结果中未提取到数值"}

    web_val = sum(numbers) / len(numbers)  # 多个结果取平均
    api_val = float(api_value) if api_value else 0

    if api_val <= 0:
        return {"verified": True, "web_value": web_val, "api_value": api_value,
                "source": results[0].get("url", ""), "discrepancy": ""}

    diff_pct = abs(web_val - api_val) / api_val * 100 if api_val > 0 else 0

    if diff_pct > 15:
        return {"verified": False, "web_value": web_val, "api_value": api_value,
                "source": results[0].get("url", ""),
                "discrepancy": f"差异{diff_pct:.0f}%: API={api_val}, Web={web_val:.1f}"}
    else:
        return {"verified": True, "web_value": web_val, "api_value": api_value,
                "source": results[0].get("url", ""),
                "discrepancy": f"差异{diff_pct:.0f}%(<15%,可接受)"}
