# FinBrain — 多 Agent 金融研究系统

面向自动化股票研究的 AI Agent 工作流，融合结构化数据检索、RAG 行业知识增强、多 Agent 推理与 Critic 审查、确定性守卫机制——以可靠性优先于生成能力。

> 项目定位：把 LLM 当成“会犯错的初级分析师”，用代码做财务计算、用审计做一致性检查、用多 Agent 做复核，最终输出可审计、可回测、可解释的投资报告。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **结构化数据抓取** | 财报、估值、行情、行业、公告、资金流量多源并发，TTM / 增速 / 现金流纯代码计算 |
| **RAG 行业知识** | ChromaDB + ONNX 本地向量库，按行业注入估值模板与会计规则 |
| **LangGraph 8 节点流水线** | Data → Classify → Analyst → Valuation → Critics(3路并行) → Repair → Reporter → Audit |
| **代码级审计** | 9 项确定性检查 + FCF 预警 + 框架分歧检测，评分/估值/单位/时效性不交给 LLM |
| **双通道业绩快报** | 财报快源 `RPT_FCI_PERFORMANCEE` + 公告快报正文解析，正式财报真空期仍能覆盖最新业绩 |
| **情景估值校验** | 结构化 EPS/PE + 代码算术校验 + 概率加权重算，杜绝“三错二” |
| **datasource tier 分层** | FREE / PREMIUM / INSTITUTIONAL 三级可插拔架构，预留付费数据源插槽 |
| **本地 / 远程双模式** | 数据服务与 LLM 推理可分离部署，用户 API Key 不上传 |

---

## 架构

```
用户输入（"分析国网信通600131"）
    │
    ▼
意图路由 → Chat / Deep Analysis / Phantom Hunter
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  研究流水线（LangGraph StateGraph）                  │
│                                                      │
│  Data → Classify → Analyst → Valuation               │
│  (并行)   (Code)      (LLM)     (LLM)                │
│                                                      │
│  → Critics(三路并行) → Repair → Reporter → Audit     │
│     Logic/Finance/Industry  (LLM)    (LLM)  (Code)   │
└─────────────────────────────────────────────────────┘
    │
    ▼
结构化报告 + 执行 Trace + 审计摘要
```

**核心设计原则**：确定性计算（财务指标、评分、估值、验证）由代码处理；概率性推理（投资逻辑、竞争分析、叙事生成）委托给 LLM。两者边界显式且强制。

---

## 数据说明

本项目默认使用国内免费公开数据源（新浪、东方财富、同花顺等）。由于免费接口与正式财报披露节奏不同，系统实现了**双通道业绩快报**机制：

```
正式财报(RPT_DMSK) > 业绩快报datacenter(RPT_FCI) > 公告快报正文(正则解析)
                         └ 结构化、无扣非 ─┘        └ 有扣非、文本解析 ─┘
                    两者经 merge_flash_into_profit 合并为完整最新期间行
```

- **RPT_DMSK**：三大报表，正式财报披露后更新，滞后约 4–6 周。
- **RPT_FCI_PERFORMANCEE**：主要指标快源，业绩快报发布次日入库，无扣非字段。
- **公告快报正文**：`np-cnotice` 接口读取业绩快报/预告全文，正则提取营收、归母、扣非及同比。

> 以 600131（国网信通）为例：2026-07-17 晚发布半年度业绩快报，原财报接口仍停在 2026Q1；快源 + 公告合并后，系统可在正式中报披露前生成基于 2026-06-30 半年报数据的报告。

---

## 运行方式

```bash
# 1. 进入项目目录
python run.py                    # CLI 交互模式
# 或
streamlit run frontend/app.py    # Web UI
```
注：首次使用时需在配置界面对所有默认配置项进行"apply&save",以写入本地配置文件中

环境配置参考 `configs/.env.example`（如项目提供）或 `configs/.env`，主要变量：

- `FINBRAIN_DATA_MODE=local|remote`：数据本地抓取或调用远程 API
- `FINBRAIN_LLM_MODE=local|remote_client`：LLM 本地调用或远程客户端调用
- `LLM_PROVIDER=deepseek|openai|anthropic`：模型提供商
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`：对应密钥

---

## 测试

```bash
python tests/test_e2e.py
```

当前测试覆盖：

- **54 项 e2e 测试**（含编译、数据工具、评分一致性、估值、Harness 守卫、配置、输出一致性、报告质量守卫）
- 关键真实数据冒烟：600131 业绩快报双通道接入、情景估值算术校验、成长性拐点改善

最新验证结果：54/54 全部通过 ✅

---

## 项目结构

```
├── backend/               # 核心后端：Agent、工具、评分、RAG、API、缓存、调度
│   ├── agent.py           # LangGraph 工作流编排
│   ├── tools.py           # 数据抓取与格式化工具
│   ├── scoring.py         # 评分与估值引擎
│   ├── scoring_config.py  # 评分配置与权重
│   ├── accounting_rag.py  # RAG 知识库
│   ├── api.py             # FastAPI 数据服务
│   ├── client.py          # 远程模式客户端
│   ├── cache.py           # TTL 缓存
│   ├── scheduler.py       # 定时数据预取
│   └── portfolio.py       # 模拟盘/多账户
├── frontend/              # Streamlit 前端
│   ├── app.py
│   └── kline_chart.py
├── configs/               # 配置与 .env
├── tests/                 # e2e 测试
├── docs/                  # 文档（技术教程、审计报告、README）
├── data/                  # 数据文件与 RAG 向量库
└── run.py                 # CLI 启动入口
```

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/README.md` / `README_CN.md` | 项目英文/中文介绍 |
| `docs/技术教程.md` | 按技术点从零到一讲解实现细节 |
| `docs/技术审计与改进报告.md` | 每轮修复的问题、根因、验证 |
| `docs/开发文档.md` | 开发规范与贡献指南 |

---

## 已修复的国网信通报告缺陷

| 缺陷 | 修复 |
|------|------|
| 业绩快报已出但结论“等待半年报” | 双通道快源接入 + 代码级时效性检测 |
| 情景估值三错二 | 结构化 EPS/PE + 代码算术校验 |
| 同一个乐观 PE 出现三个数 | 校验/审计统一读取结构化字段，禁止 LLM 重推导 |
| 前瞻 PE 986 倍（季节性失真） | Q1 利润占比低时禁用前瞻 PE，提示参考 TTM |
| 安全边际 3.10 元 ≈ 0.57 倍 PB | 质量乘数允许负向调整 + PB 地板（破净股豁免） |
| 风险清单被清空 | 修复 `isinstance(item, list)` 同构 bug |
| 现金流口径混用 | 年报口径优先，标注“年报/一季报” |
| 归母/扣非缺口 62pp 未解释 | 缺口 >30pp 自动注入非经常性损益风险 |
| 应收风险遗漏 | 应收账款 > 年净利润 ×3 自动注入回款/减值风险 |

---

## 后续扩展方向（已列入 Roadmap）

1. **回测机制完善**：加入波动率、最大回撤、夏普比率、仓位加权。
2. **Human-in-the-Loop**：LangGraph `interrupt` 在交易执行前人工审批。
3. **Streaming + 中间状态**：`stream()` / `astream_events` 实时展示节点进度。
4. **条件分支**：数据不足回退重取、Critic 通过跳过 Repair。
5. **模型 failover**：主模型失败自动切换备用模型。目前只对deepseek作为模型供应商跑通测试。
6. **Structured Output / Guardrails**：`with_structured_output` 或 `response_format` 替代 prompt + json.loads。
7. **Prompt Caching / Token 优化**：接入 Anthropic prompt caching / DeepSeek 上下文缓存。
8. **精细化单元测试**：portfolio 交易逻辑、评分边界、datasource_tier fallback 等。
9. **评估体系升级**：从“信号触发检测”扩展到“策略收益评估”与投资建议准确性。

---

*最后更新：2026-07-19*
