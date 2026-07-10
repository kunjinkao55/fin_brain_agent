"""
股票代码 <-> 名称 双向映射。首次运行从新浪拉取全A股列表，缓存到 JSON。
"""

import json, os, ssl, urllib.request

MAP_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stock_map.json")


def _download_stock_list() -> dict:
    """从新浪全市场API拉取股票列表"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"User-Agent": "Mozilla/5.0"}

    code_to_name = {}
    name_to_code = {}

    for page in range(1, 60):
        url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/"
               f"json_v2.php/Market_Center.getHQNodeData?"
               f"page={page}&num=100&sort=code&asc=1&node=hs_a")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                data = json.loads(resp.read().decode("gbk"))
        except Exception:
            continue

        if not data:
            break

        for s in data:
            code = s.get("code", "")
            name = s.get("name", "")
            if not code or "ST" in name or "退" in name:
                continue
            if code in code_to_name:
                return {"code_to_name": code_to_name, "name_to_code": name_to_code}
            code_to_name[code] = name
            if name not in name_to_code or len(code) < len(name_to_code[name]):
                name_to_code[name] = code

    return {"code_to_name": code_to_name, "name_to_code": name_to_code}


def get_stock_map() -> dict:
    """获取映射表，首次调用自动下载并缓存"""
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    mapping = _download_stock_list()
    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping


def name_to_code(name: str) -> str | None:
    """股票名 → 代码 (精确匹配)"""
    m = get_stock_map()
    return m["name_to_code"].get(name)


def code_to_name(code: str) -> str | None:
    """代码 → 股票名"""
    m = get_stock_map()
    return m["code_to_name"].get(code)


def fuzzy_search(query: str, limit: int = 10) -> list:
    """模糊搜索股票名或代码，返回匹配列表"""
    m = get_stock_map()
    results = []

    # 代码包含
    for code, name in m["code_to_name"].items():
        if query in code:
            results.append((code, name, "code"))
        elif query in name:
            results.append((code, name, "name"))
        if len(results) >= limit * 2:
            break

    # 去重并按匹配质量排序（全名匹配 > 包含匹配 > 首字匹配）
    seen = set()
    final = []
    for code, name, typ in results:
        if code not in seen:
            seen.add(code)
            final.append({"代码": code, "名称": name})
        if len(final) >= limit:
            break

    return final
