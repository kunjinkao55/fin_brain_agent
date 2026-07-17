"""
FinBrain 模拟盘模块 — 多账户 JSON 持久化。支持市价/限价/百分比仓位/一键重置。
"""

import json, os, threading, glob, re
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DEFAULT_CASH = float(os.getenv("PORTFOLIO_CASH", "1000000"))
LOT_SIZE = 100
_portfolio_cache: dict = {}  # 账户名 → Portfolio 实例缓存


def _account_file(name: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_一-鿿-]', '_', name)
    return os.path.join(BASE_DIR, f"portfolio_{safe}.json")


def list_accounts() -> list:
    """列出所有模拟盘账户"""
    accounts = []
    for f in glob.glob(os.path.join(BASE_DIR, "portfolio_*.json")):
        base = os.path.basename(f)
        name = base[len("portfolio_"):-len(".json")]
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
            accounts.append({"name": name, "cash": d.get("cash", 0),
                             "initial_cash": d.get("initial_cash", 0),
                             "positions": len(d.get("positions", {})),
                             "total_value": d.get("cash", 0) + sum(
                                 p.get("market_value", 0) for p in d.get("positions", {}).values())})
        except Exception:
            accounts.append({"name": name, "cash": 0, "error": True})
    # 兼容旧版 portfolio.json → 迁移为 portfolio_default.json
    old_file = os.path.join(BASE_DIR, "portfolio.json")
    if os.path.exists(old_file) and not any(a["name"] == "default" for a in accounts):
        os.rename(old_file, _account_file("default"))
        accounts.insert(0, {"name": "default", "cash": 0, "initial_cash": DEFAULT_CASH, "positions": 0})
    return accounts


def delete_account(name: str) -> bool:
    f = _account_file(name)
    if os.path.exists(f):
        os.remove(f)
        _portfolio_cache.pop(name, None)  # 清缓存，否则同名重建时拿到旧对象
        return True
    return False


class Portfolio:
    def __init__(self, account_name: str = "default"):
        self.account = account_name
        self.file = _account_file(account_name)
        self._lock = threading.Lock()
        if os.path.exists(self.file):
            self._load()
        else:
            self._init_fresh(DEFAULT_CASH)

    def _init_fresh(self, cash: float):
        self.cash = cash
        self.initial_cash = cash
        self.positions = {}
        self.history = []
        self._save()

    def _load(self):
        with open(self.file, "r") as f:
            data = json.load(f)
        self.cash = data.get("cash", DEFAULT_CASH)
        self.initial_cash = data.get("initial_cash", DEFAULT_CASH)
        self.positions = data.get("positions", {})
        self.history = data.get("history", [])

    def _save(self):
        with self._lock:
            with open(self.file, "w") as f:
                json.dump({"cash": self.cash, "initial_cash": self.initial_cash,
                           "positions": self.positions, "history": self.history},
                          f, ensure_ascii=False, indent=2)

    # ------- 行情获取 -------
    def _get_price(self, symbol: str) -> dict:
        """获取实时价格（复用 tools 里的 fetch_stock_price）"""
        from backend.tools import fetch_stock_price
        return fetch_stock_price(symbol)

    # ------- 交易 -------
    def buy(self, symbol: str, shares: int, price: float = None) -> dict:
        """买入。price=None时自动取实时价。"""
        if price is None:
            quote = self._get_price(symbol)
            if "error" in quote:
                return {"error": f"获取价格失败: {quote['error']}"}
            price = float(quote["price"])
            name = quote["name"]
        else:
            name = ""

        cost = price * shares
        if cost > self.cash:
            return {"error": f"现金不足。需{cost:.0f}元, 可用{self.cash:.0f}元"}

        self.cash -= cost

        if symbol in self.positions:
            p = self.positions[symbol]
            total_shares = p["shares"] + shares
            p["avg_cost"] = (p["avg_cost"] * p["shares"] + price * shares) / total_shares
            p["shares"] = total_shares
        else:
            self.positions[symbol] = {"name": name, "shares": shares,
                                       "avg_cost": price, "date": str(datetime.now())[:10]}

        self.history.append({"symbol": symbol, "name": name, "action": "BUY",
                             "shares": shares, "price": price, "total": cost,
                             "date": str(datetime.now())[:19], "pnl": 0})

        self._save()
        return {"status": "ok", "action": "BUY", "symbol": symbol, "name": name,
                "shares": shares, "price": price, "cost": cost, "cash_remaining": self.cash}

    def sell(self, symbol: str, shares: int, price: float = None) -> dict:
        """卖出。shares=-1表示全仓卖出。"""
        if symbol not in self.positions:
            return {"error": f"未持有 {symbol}"}

        p = self.positions[symbol]
        if shares == -1 or shares >= p["shares"]:
            shares = p["shares"]
            del self.positions[symbol]
        else:
            p["shares"] -= shares

        if price is None:
            quote = self._get_price(symbol)
            if "error" in quote:
                return {"error": f"获取价格失败: {quote['error']}"}
            price = float(quote["price"])

        revenue = price * shares
        cost_basis = p["avg_cost"] * shares
        pnl = revenue - cost_basis
        self.cash += revenue

        self.history.append({"symbol": symbol, "name": p["name"], "action": "SELL",
                             "shares": shares, "price": price, "total": revenue,
                             "date": str(datetime.now())[:19],
                             "pnl": round(pnl, 2), "pnl_pct": f"{pnl/cost_basis*100:.1f}%"})

        self._save()
        return {"status": "ok", "action": "SELL", "symbol": symbol, "name": p["name"],
                "shares": shares, "price": price, "revenue": revenue, "pnl": round(pnl, 2),
                "pnl_pct": f"{pnl/cost_basis*100:.1f}%", "cash_remaining": self.cash}

    # ------- 百分比仓位 -------
    def buy_pct(self, symbol: str, pct: float) -> dict:
        """按总资产百分比买入（自动取整手）"""
        total_assets = self.cash + self._total_market_value()
        budget = total_assets * pct / 100
        quote = self._get_price(symbol)
        if "error" in quote:
            return {"error": f"获取价格失败: {quote['error']}"}
        price = float(quote["price"])
        shares = int(budget / price / LOT_SIZE) * LOT_SIZE
        if shares <= 0:
            return {"error": f"资金不足。{pct}%仓位需{budget:.0f}元, 股价{price}, 只够{int(budget/price)}股(不足1手)"}
        return self.buy(symbol, shares, price)

    def sell_pct(self, symbol: str, pct: float) -> dict:
        """按持仓百分比卖出。pct=100即清仓。"""
        if symbol not in self.positions:
            return {"error": f"未持有 {symbol}"}
        current_shares = self.positions[symbol]["shares"]
        shares = int(current_shares * pct / 100 / LOT_SIZE) * LOT_SIZE
        if shares <= 0:
            shares = min(current_shares, LOT_SIZE)
        if shares >= current_shares:
            shares = -1  # 全卖
        return self.sell(symbol, shares)

    # ------- 重置 -------
    def reset(self, cash: float = None) -> dict:
        """重置模拟盘。cash=None则沿用默认资金。"""
        init = cash or DEFAULT_CASH
        self._init_fresh(init)
        return {"status": "ok", "message": f"模拟盘已重置，初始资金: {init:,.0f}元"}

    # ------- 内部 -------
    def _total_market_value(self) -> float:
        total = 0
        for symbol, p in self.positions.items():
            quote = self._get_price(symbol)
            price = float(quote.get("price", p["avg_cost"])) if "error" not in quote else p["avg_cost"]
            total += price * p["shares"]
        return total

    # ------- 查询 -------
    def summary(self) -> dict:
        """获取持仓总览（含实时盈亏）"""
        positions_detail = []
        total_market_value = 0
        total_cost = 0

        for symbol, p in self.positions.items():
            quote = self._get_price(symbol)
            if "error" in quote:
                current_price = p["avg_cost"]
            else:
                current_price = float(quote["price"])
                if not p["name"]:
                    p["name"] = quote.get("name", "")

            market_value = current_price * p["shares"]
            cost = p["avg_cost"] * p["shares"]
            unrealized_pnl = market_value - cost

            total_market_value += market_value
            total_cost += cost

            positions_detail.append({
                "代码": symbol, "名称": p["name"],
                "持仓": p["shares"], "成本价": round(p["avg_cost"], 2),
                "现价": round(current_price, 2),
                "市值": round(market_value, 0),
                "盈亏": round(unrealized_pnl, 0),
                "盈亏%": f"{unrealized_pnl/cost*100:.1f}%" if cost else "0%",
                "建仓日": p.get("date", ""),
            })

        total_pnl = total_market_value - total_cost
        total_assets = self.cash + total_market_value
        total_return = total_assets - self.initial_cash
        return {
            "初始资金": round(self.initial_cash, 0),
            "现金": round(self.cash, 0),
            "持仓市值": round(total_market_value, 0),
            "总资产": round(total_assets, 0),
            "累计收益率": f"{total_return/self.initial_cash*100:.1f}%",
            "总盈亏": round(total_pnl, 0),
            "总盈亏%": f"{total_pnl/total_cost*100:.1f}%" if total_cost else "0%",
            "持仓明细": positions_detail,
            "持仓数": len(positions_detail),
        }

    def recent_trades(self, n: int = 20) -> list:
        """最近N笔交易记录"""
        return self.history[-n:]


# 全局实例
_portfolio = None


def get_portfolio(account: str = "default") -> Portfolio:
    """获取指定账户的模拟盘（按账户名缓存）。"""
    global _portfolio_cache
    if account not in _portfolio_cache:
        _portfolio_cache[account] = Portfolio(account)
    return _portfolio_cache[account]


