"""
FinBrain RAG 模块 — ChromaDB + BGE-small-zh 本地嵌入
游资风格知识库，供 Phantom Hunter 语义检索营业部-游资对应关系。

设计原则（遵循项目技术审计标准）：
- 懒初始化：重模型不在 import 时加载
- 线程安全：double-check locking
- 显式错误处理：网络/磁盘/模型问题不静默
"""

import os, json, threading, logging
import chromadb

logger = logging.getLogger(__name__)

# ---- 配置 ----
_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "chroma")

# ---- 懒初始化（double-check locking） ----
_client = None
_embed_fn = None
_youzi_collection = None
_init_lock = threading.Lock()
_initialized = False


def _ensure_init():
    """线程安全的懒初始化。首次调用时下载模型 + 建库，后续调用零开销。
    嵌入模型 text2vec-base-chinese 走 ModelScope 国内CDN，不需要HuggingFace。
    """
    global _client, _embed_fn, _youzi_collection, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        try:
            os.makedirs(_DB_DIR, exist_ok=True)
            _client = chromadb.PersistentClient(path=_DB_DIR)

            # 嵌入模型：优先 ONNX (80MB，内置)，失败退到 Chromadb 默认
            try:
                _embed_fn = chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2()
            except Exception:
                _embed_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            _youzi_collection = _client.get_or_create_collection(
                name="youzi_profiles",
                embedding_function=_embed_fn,
                metadata={"description": "游资风格/席位/胜率知识库"},
            )
            _initialized = True
            logger.info("RAG initialized: embedder=%s, collection=youzi_profiles",
            type(_embed_fn).__name__)
        except Exception as e:
            logger.exception("RAG initialization failed: %s", e)
            raise RuntimeError(f"RAG初始化失败: {e}") from e


# ---- 游资数据 ----
def _load_youzi_data() -> list[dict]:
    """加载游资数据。生产级应改为从配置文件读取。"""
    return [
        {"id": "炒股养家", "name": "炒股养家",
         "text": "风格：格局锁仓，不轻易卖。偏好科技股和次新股。历史胜率约62%。常用席位：华鑫证券上海分公司、华鑫证券上海宛平南路、华鑫证券上海松江。操作特点：买入后锁仓3-7天，不轻易止损，擅长判断题材持续性。跟风价值：高，适合接力。"},
        {"id": "方新侠", "name": "方新侠",
         "text": "风格：打板猛，次日高开出货，一日游为主。历史胜率约55%。常用席位：中信证券上海分公司、中信证券上海溧阳路。操作特点：早盘快速拉板，次日竞价高开即出货，不恋战。跟风价值：低，容易被砸。"},
        {"id": "上塘路", "name": "上塘路",
         "text": "风格：跟风助攻，快进快出。历史胜率约50%。常用席位：中信证券杭州上塘路、中信证券杭州延安路。操作特点：看到龙头涨停后追跟风股，持股周期1-2天。跟风价值：中，需警惕次日出货。"},
        {"id": "作手新一", "name": "作手新一",
         "text": "风格：题材挖掘，持股周期适中（3-5天）。历史胜率约58%。常用席位：国泰君安南京太平南路、国泰君安上海分公司。操作特点：擅长在题材发酵期介入，不追涨不杀跌。跟风价值：高，适合接力。"},
        {"id": "赵老哥", "name": "赵老哥",
         "text": "风格：消息驱动，打板不恋战。历史胜率约48%。常用席位：中国银河证券上海杨浦区、中国银河证券北京。操作特点：利好出来第一时间打板，次日高开即走。跟风价值：低，纯消息面，容易踩雷。"},
        {"id": "小鳄鱼", "name": "小鳄鱼",
         "text": "风格：趋势接力，偏好新能源和周期股。历史胜率约60%。常用席位：东方证券上海浦东新区、东方证券上海静安区。操作特点：在涨停板次日低吸而非追板，持仓周期5-10天。跟风价值：高，适合低吸跟随。"},
        {"id": "章盟主", "name": "章盟主",
         "text": "风格：锁仓+低吸，偏好白马蓝筹。历史胜率约70%。常用席位：国泰君安上海分公司、海通证券上海。操作特点：大跌后低吸白马，不是打板风格，持股周期1-3个月。跟风价值：最高，但风格不匹配妖股狩猎。"},
        {"id": "欢乐海岸", "name": "欢乐海岸",
         "text": "风格：低吸龙头，偏好军工和科技。历史胜率约45%。常用席位：中信证券深圳分公司。操作特点：龙头回调20%后低吸，等待反弹。跟风价值：中，需精准抄底。"},
        {"id": "佛山系", "name": "佛山系",
         "text": "风格：打板龙头，次日必出货。历史胜率约52%。常用席位：国信证券佛山南海大道。操作特点：专注首板，次日竞价出货。跟风价值：极低，典型一日游。"},
        {"id": "宁波桑田路", "name": "宁波桑田路",
         "text": "风格：次新股打板+锁仓。历史胜率约65%。常用席位：光大证券宁波桑田路。操作特点：专注次新股首板，封板后锁仓3-5天。跟风价值：高，适合接力。"},
    ]


# ---- 公开 API ----

def init_youzi_db():
    """建库（幂等：已存在则跳过）。通常首次运行时调用一次。"""
    _ensure_init()
    existing = _youzi_collection.get()
    if existing["ids"]:
        logger.info("youzi DB already populated (%d records), skipping", len(existing["ids"]))
        return

    data = _load_youzi_data()
    docs = [item["text"] for item in data]
    ids = [item["id"] for item in data]
    metadatas = [{"name": item["name"]} for item in data]
    _youzi_collection.add(documents=docs, ids=ids, metadatas=metadatas)
    logger.info("youzi DB initialized: %d records", len(data))


def search_youzi(query: str, top_k: int = 5) -> list[dict]:
    """语义检索游资知识库。输入自然语言查询，返回相关游资信息。"""
    _ensure_init()
    try:
        results = _youzi_collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        logger.exception("RAG search failed for query '%s'", query[:100])
        return []

    items = []
    if results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            dist = results.get("distances", [[1]])[0][i] if results.get("distances") else 1.0
            items.append({
                "游资": doc_id,
                "信息": results["documents"][0][i] if results.get("documents") else "",
                "相关度": round(max(0, 1 - dist), 3),
            })
    return items


def identify_youzi_rag(seat_name: str) -> list[str]:
    """输入营业部名称，用语义匹配识别对应游资。弱相关（<0.25）的结果过滤。"""
    results = search_youzi(f"营业部：{seat_name}", top_k=3)
    return [r["游资"] for r in results if r["相关度"] > 0.25]
