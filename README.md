# FinBrain — 多智能体财报分析与智能择时荐股平台

AI Agent 简历项目。独立运行，不依赖任何商业 Agent 平台。

## 项目概述

融合**价值成长选股**与**妖股狩猎**双策略的 AI 投研系统。输入自然语言，Agent 自动调用 10+ 金融数据工具，按策略评分，输出对齐表格报告。

## 架构

```
用户输入
    │
    ├─ 闲聊/查询 ──→ 轻量Agent (股价+K线)
    ├─ "分析/报告" ──→ 三节点流水线 (Data→Analyst→Reporter)
    └─ "妖股/涨停" ──→ Phantom Hunter (涨停池+龙虎榜+游资)
```

## 已完成功能

### 10 个数据工具 (tools.py, ~750行)

| # | 工具 | 数据源 | 用途 |
|---|------|--------|------|
| 1 | stock_price | 新浪 | 实时行情 |
| 2 | stock_history | 新浪 | K线数据 |
| 3 | financial_statements | 东方财富 datacenter | 三大报表(3年) |
| 4 | valuation | 东方财富 datacenter | ROE/毛利率/净利率/EPS |
| 5 | industry_info | 东方财富+同花顺 | 行业分类+指数 |
| 6 | screen_stocks | 新浪全市场 | PE/PB/市值扫描 |
| 7 | fund_flow | 同花顺(hexin-v鉴权) | 资金流向 |
| 8 | limit_up_pool | 新浪 | 涨停板池 |
| 9 | concept_ranking | 同花顺 | 概念板块 |
| 10 | dragon_tiger_list | 新浪(AkShare) | 龙虎榜+游资库 |
| — | format_report | (纯函数) | 评分卡格式化 |

### 3 条 Agent 路由 (agent.py, ~550行)

| 路由 | 触发词 | Agent | 工具数 |
|------|--------|-------|:--:|
| 闲聊 | 默认 | 轻量Agent | 2 (股价+K线) |
| 深度分析 | 分析/报告/评分/扫描/对比/选股 | Data→Analyst→Reporter | 7 |
| 妖股狩猎 | 妖股/猎妖/涨停/打板/短线爆发 | Phantom Hunter | 10 |

### Harness 工程基座

| 能力 | 实现 |
|------|------|
| 多 Agent 编排 | LangGraph StateGraph (Data→Analyst→Reporter) |
| 上下文压缩 | 超阈值自动摘要，保留最近N条 |
| 多 LLM 切换 | .env 一行配置 (DeepSeek/Anthropic/OpenAI) |
| 表格对齐 | 纯 Python 计算列宽，数字右对齐中文左对齐 |
| 编码兼容 | Windows UTF-8 surrogates 处理 |
| MCP 兼容 | MCP_server.py 可独立给 Claude Code 用 |

### 策略引擎

- **趋势投资法**：行业景气判断 → 龙头聚焦 → 回调买入 → 趋势持有
- **财报狙击**：跳空高开5%+缺口不回补 → 三重财务验证(营收增速/毛利率/合同负债)
- **错杀白马捡漏**：ROE>20% + 跌30%+ + 估值5年低位 → 分批建仓
- **妖股筛选**：小盘(20-80亿) + 涨停启动 + 热门概念 + 龙虎榜游资

## 快速开始

```bash
# 1. 安装依赖
pip install python-dotenv langgraph langchain langchain-openai langchain-anthropic
pip install akshare pandas requests beautifulsoup4 py-mini-racer

# 2. 配置 .env
echo "LLM_PROVIDER=deepseek" > .env
echo "DEEPSEEK_API_KEY=sk-your-key" >> .env

# 3. 运行
python agent.py
```

```
FinBrain Agent
Type 'quit' to exit, 'clear' to reset context

>$ 分析新易盛300502的财报和估值
================================================================
  FinBrain 分析报告: 新易盛 (300502)
================================================================
  [评分卡]  满分10分
  维度         得分  评级    依据
  ...
>$ 猎妖
# Phantom Hunter 输出涨停板+龙虎榜+题材分析
```

## 项目结构

```
agent/
├── agent.py               # LangGraph Agent 主程序
├── tools.py               # 10个纯数据函数 + 格式化工具
├── MCP_server.py          # MCP 兼容层(可给Claude Code用)
├── .env                   # LLM配置 + API Key
├── .gitignore
├── README.md              # 本文件
├── 开发文档.md             # 详细设计文档
├── 财报选股agent.md        # 原始需求文档
├── kline_601991.html      # K线图示例
└── kline_300502.html      # K线图示例
```

## 技术栈

| 层 | 选型 |
|----|------|
| Agent 框架 | LangGraph + LangChain |
| LLM | DeepSeek (默认) / Claude / GPT-4o 可切换 |
| 数据源 | 新浪财经 + 东方财富 datacenter + 同花顺 10jqka |
| 鉴权绕过 | SSL证书绕过 + py_mini_racer JS引擎 |
| 纯依赖 | urllib + json + re (核心工具零第三方依赖) |

## 待开发

| 优先级 | 模块 | 说明 |
|:--:|------|------|
| 1 | Streamlit 前端 | 可视化仪表盘、K线图、报告卡片 |
| 2 | RAG 知识库 | ChromaDB, 会计准则+游资风格+题材舆情向量检索 |
| 3 | 记忆持久化 | SQLite/MongoDB 存用户偏好+历史会话 |
| 4 | 席位级游资分析 | 需付费数据源 (东方财富push2被封) |
| 5 | Harness 熔断落地 | 妖股3日回撤>10%自动暂停模块 |
| 6 | 分时量价工具 | 日内分时数据 + 烂板出妖股识别 |
| 7 | 测试用例 | 端到端回归测试 + 离线评估集 |

## 简历要点

> FinBrain — 基于 LangGraph 的多智能体财报分析与智能择时荐股平台
>
> - 设计并实现 10 个金融数据工具，覆盖新浪/东方财富/同花顺三源，含 hexin-v JS 鉴权逆向
> - 基于 LangGraph StateGraph 构建 Data→Analyst→Reporter 三节点流水线，解耦数据搜集、策略评分、格式化输出
> - 独创 Phantom Hunter 妖股猎人 Agent，融合涨停板池+龙虎榜+概念热度+游资知识库的多维筛选
> - 实现上下文摘要压缩、多 LLM 热切换(DeepSeek/Claude/GPT)、三级路由分发等工程基座能力
> - 内置趋势投资法+财报狙击+错杀白马+妖股筛选四套策略，输出结构化评分卡

## 许可

MIT
