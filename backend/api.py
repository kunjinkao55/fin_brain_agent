"""
FinBrain Data API — FastAPI 服务
暴露所有数据端点 + Prompt 生成端点，供客户端远程调用。
启动: uvicorn backend.api:app --host 0.0.0.0 --port 8000
"""

import os, sys, json, logging
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finbrain.api")

app = FastAPI(title="FinBrain Data API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 工业标准三层缓存策略：
# L1: scheduler 定时预取热门数据 → 数据已在 Redis/内存中，命中率 >95%
# L2: worker 内 cache.get() → 毫秒级返回
# L3: 反向代理（Nginx proxy_cache）→ 处理并发冷请求去重
# 应用层只负责设置标准 HTTP 缓存头，去重和缓存由 Nginx/Redis 负责

from backend.cache import TTL as _CACHE_TTL

def _cache_header(response: Response, tool_name: str):
    """设置标准 Cache-Control 头，供 Nginx/CDN/浏览器缓存"""
    ttl = _CACHE_TTL.get(tool_name, 60)
    response.headers["Cache-Control"] = f"public, max-age={ttl}"
    response.headers["X-Cache-TTL"] = str(ttl)


# ---- 依赖注入：强制 local 模式获取数据 ----
def _ensure_local():
    os.environ["FINBRAIN_DATA_MODE"] = "local"


_ensure_local()
from backend.tools import (
    fetch_stock_price, fetch_stock_history, get_financial_statements,
    get_valuation, get_industry_info, get_fund_flow,
    get_limit_up_pool, get_market_breadth, get_sector_fund_flow,
    get_intraday, get_concept_ranking, get_dragon_tiger_list,
    get_dragon_tiger_detail, get_stock_streak, screen_stocks,
    calculate_scores,
)


# ============================================================
#  数据端点
# ============================================================

@app.get("/api/data/stock_history")
def api_stock_history(symbol: str, scale: int = 240, datalen: int = 30):
    return fetch_stock_history(symbol, scale, datalen)


@app.get("/api/data/financials")
def api_financials(symbol: str):
    return get_financial_statements(symbol)


@app.get("/api/data/stock_price")
def api_stock_price(symbol: str = Query(...)):
    return fetch_stock_price(symbol)


@app.get("/api/data/valuation")
def api_valuation(symbol: str):
    return get_valuation(symbol)


@app.get("/api/data/industry")
def api_industry(symbol: str):
    return get_industry_info(symbol)


@app.get("/api/data/fund_flow")
def api_fund_flow(symbol: str):
    return get_fund_flow(symbol)


@app.get("/api/data/intraday")
def api_intraday(symbol: str):
    return get_intraday(symbol)


@app.get("/api/data/stock_streak")
def api_stock_streak(symbol: str):
    return get_stock_streak(symbol)


# ============================================================
#  市场端点
# ============================================================

@app.get("/api/market/limit_up")
def api_limit_up(top_n: int = 30):
    return get_limit_up_pool(top_n)


@app.get("/api/market/breadth")
def api_breadth():
    return get_market_breadth()


@app.get("/api/market/sector_fund_flow")
def api_sector_fund_flow(top_n: int = 100):
    return get_sector_fund_flow(top_n)


@app.get("/api/market/concept_ranking")
def api_concept_ranking(top_n: int = 20):
    return get_concept_ranking(top_n)


@app.get("/api/market/dragon_tiger_list")
def api_dragon_tiger_list(date: str = ""):
    return get_dragon_tiger_list(date)


@app.get("/api/market/dragon_tiger_detail")
def api_dragon_tiger_detail(symbol: str):
    return get_dragon_tiger_detail(symbol)


@app.get("/api/market/screen_stocks")
def api_screen_stocks(max_pe: float = 30, max_pb: float = 5, min_mktcap: float = 20, top_n: int = 30):
    return screen_stocks(max_pe, max_pb, min_mktcap, top_n)


# ============================================================
#  Prompt 生成端点
# ============================================================

@app.post("/api/prompt/analysis")
async def api_prompt_analysis(body: dict):
    """生成分析 Prompt + 预计算评分，不调 LLM"""
    question = body.get("question", "")
    symbols = body.get("symbols", [])
    if not symbols:
        raise HTTPException(400, "请提供股票代码列表 symbols")

    # 并发拉取数据
    import concurrent.futures, re
    from backend.scoring_config import get_valuation as _get_val_cfg
    from backend.accounting_rag import search_kb, seed_industry_kb

    def _fetch_one(code):
        try:
            fin = get_financial_statements(code)
            val = get_valuation(code)
            price = fetch_stock_price(code)
            ind = get_industry_info(code)
            cs_data = {"profit": fin.get("profit",[]), "cashflow": fin.get("cashflow",[]),
                       "balance": fin.get("balance",[]), "valuation": val,
                       "price": dict(price) if isinstance(price, dict) else price,
                       "industry": ind.get("行业", ind.get("industry_name", "")) if isinstance(ind, dict) else ""}
            scores = calculate_scores(cs_data)
            name = price.get("name", code) if isinstance(price, dict) else code
            return {"代码": code, "名称": name, "行情": price, "行业": ind,
                    "财报": {"利润表": fin.get("profit",[])[:4], "现金流": fin.get("cashflow",[])[:2]},
                    "估值": val.get("data",[])[:2] if isinstance(val, dict) else [],
                    "预计算分数": scores}
        except Exception as e:
            return {"代码": code, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_fetch_one, symbols))

    collected = json.dumps(results, ensure_ascii=False, indent=2)

    # RAG 行业模板注入
    try: seed_industry_kb()
    except: pass
    industry_names = list(set(re.findall(r'"行业":\s*"([^"]+)"', collected)))
    industry_rag = ""
    for ind_name in industry_names[:3]:
        kb_results = search_kb(f"{ind_name} 分析 估值 护城河", "industry", top_k=2)
        if kb_results:
            snippets = [r["content"][:400] for r in kb_results if r.get("content")]
            if snippets:
                industry_rag += f"\n[RAG行业模板-{ind_name}]\n" + "\n---\n".join(snippets) + "\n"

    # 组装 Prompt（复用 agent.py 的 ANALYST_PROMPT）
    from backend.agent import ANALYST_PROMPT
    stock_count = len(symbols)
    multi_note = f"\n\n[!!!] 当前涉及{stock_count}只股票，输出{stock_count}个对象的JSON数组。" if stock_count >= 2 else ""

    user_prompt = (
        f"用户问题: {question}\n\n"
        f"=== 已搜集数据 ===\n{collected}\n{industry_rag}\n"
        f"[任务] 基于以上数据和行业模板，撰写完整分析JSON。输出纯JSON。{multi_note}"
    )

    return {
        "system_prompt": ANALYST_PROMPT,
        "user_prompt": user_prompt,
        "data": json.loads(collected),
        "industry_rag": industry_rag,
    }


@app.post("/api/prompt/phantom")
async def api_prompt_phantom(body: dict):
    """生成妖股分析 Prompt + 聚合数据快照"""
    question = body.get("question", "查找近期潜力妖股")
    # 聚合数据
    limit_up = get_limit_up_pool(50)
    dragon = get_dragon_tiger_list()
    concepts = get_concept_ranking(30)
    breadth = get_market_breadth()
    sector = get_sector_fund_flow(50)

    snapshot = {
        "涨停板": limit_up,
        "龙虎榜": dragon,
        "概念排行": concepts,
        "市场广度": breadth,
        "板块资金流": sector,
    }

    from backend.agent import _get_strategy
    phantom_prompt = _get_strategy().get("phantom", "")

    return {
        "system_prompt": phantom_prompt,
        "user_prompt": question,
        "snapshot": snapshot,
    }


# ============================================================
#  健康检查
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "mode": "server", "version": "2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
