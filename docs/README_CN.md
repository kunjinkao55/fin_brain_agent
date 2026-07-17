# FinBrain

## 多 Agent 金融研究系统

面向自动化股票研究的 AI Agent 工作流，融合结构化数据检索、RAG 行业知识增强、多 Agent 推理与 Critic 审查、确定性守卫机制——以可靠性优先于生成能力。

---

## 设计动机

基于 LLM 的金融分析工具普遍存在四类失败模式：

1. **数据幻觉** — LLM 编造 PE、营收等财务数字
2. **领域知识缺失** — 通用模型缺乏行业特定的估值框架
3. **推理不一致** — 同一只股票分析两次可能得出矛盾结论
4. **缺乏验证** — 输出未经数学一致性或逻辑自洽性检查

FinBrain 将金融研究拆分为专业化组件来解决上述问题：确定性计算处理数字，RAG 注入领域知识，多 Agent 工作流完成结构化推理，双层审计（代码 + LLM）完成验证。

---

## 架构

```
用户输入（"分析长电科技600584"）
    │
    ▼
意图路由 → Chat / Deep Analysis / Phantom Hunter
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  研究流水线（LangGraph StateGraph）                    │
│                                                      │
│  Data → Classifier → Analyst → Valuation             │
│  (并行)   (Code)      (LLM)     (LLM)                │
│                                                      │
│  → Critics(三路并行) → Repair → Reporter              │
│     Logic/Finance/Industry  (LLM)    (LLM)           │
│                                                      │
│  → Audit Engine (Code)                                │
│     · 评分一致性 · 数据单位验证                         │
│     · 稀释修正 · 情景单调性 · FCF 预警                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
结构化报告 + 执行 Trace + 审计摘要
```

**核心设计原则**：确定性计算（财务指标、评分、估值、验证）由代码处理。概率性推理（投资逻辑、竞争分析、叙事生成）委托给 LLM。两者之间的边界显式且强制。

---

## Agent 工作流

### Data Agent（并行，无 LLM）
通过 ThreadPoolExecutor 并发抓取财报、估值、行情、行业分类、公告。确定性计算 TTM EPS、PE/PB、增速趋势和现金流质量。注入 `[INDUSTRY]` 和 `[TOOLS]` 标记供下游追踪。典型延迟：6 个数据源约 5 秒。

### Company Classifier（Code，无 LLM）
从财务数据中提取 ROE 水平与波动性、营收增速趋势、CAPEX/CFO 强度。输出混合属性权重（如 `{"cyclical": 0.5, "growth": 0.3, "theme": 0.2}`），供 Valuation Agent 选择估值框架。

### Analyst Agent（LLM）
接收结构化财务数据 + RAG 检索的行业模板。按 10 步分析框架执行：公司分类→商业模式→护城河评估→行业周期→财务验证→增长逻辑→市场预期→情景估值→催化剂→投资决策。输出结构化 JSON。

### Valuation Agent（LLM）
接收 Classifier 的混合权重。输出 PE 折价链（行业 PE 40x→周期折价-30%→ROE 不足-20%→CAPEX 压力-15%→成长溢价+20%=最终 22x），以及多框架估值区间（保守 PE 正常化、混合、乐观 PEG）。

### Critics — 三路并行审查（LLM × 3 + Code）
- **Logic Critic**：检查因果链自洽性、过度自信措辞、评级-操作一致性
- **Financial Critic**：代码层预检（FCF=CFO-CAPEX、PE/ROE 匹配度、CFO/折旧失真）→将发现注入 LLM prompt 进行语义层审查
- **Industry Critic**：验证竞争格局描述、技术路线定位、产业链分析

通过代码层 Aggregator 合并去重，输出结构化修复清单。

### Repair Agent（LLM）
接收 Critics 的结构化修复清单。自动修正分析 JSON：降低绝对化表述、修正数据误读、补充遗漏风险。仅修改被标记的字段——不重新生成整份报告。

### Reporter Agent（LLM）
生成最终叙述文本，受 Critics 发现和 Repair 修正双重约束。

### Audit Engine（Code，无 LLM）
8 项确定性检查 + FCF 预警注入。当评分引擎的保守 PE 锚点与 LLM 的增长叙事冲突时，生成 `[框架分歧]` 段落同时呈现两套逻辑，而非强行压平。

---

## 执行 Trace

每次流水线运行生成可观测的 Trace：

```
📡 Data — SUCCESS | 预取1只股票
  Latency: 4800ms | Actions: ✅财报 ✅估值 ✅行情 ✅行业 ✅评分 ✅公告

🏷️ Classify — SUCCESS | 50%周期制造 + 30%成长 + 20%主题
  Metrics: ROE 0.056, volatility 0.8, CAPEX/CFO 1.4

🧠 Analysis — SUCCESS | 投资分析生成完成
  Output: 8500 chars | RAG: 行业模板(半导体): 2条

📊 Valuation — SUCCESS | PE折价链: 40→周期-30%→ROE-20%→CAPEX-15%→成长+20%=22x
  Frameworks: PE(正常化利润), EV/EBITDA, PEG

🔍 Critics — WARNING | 三路审查: Logic:2 Financial:3 Industry:1
  Code findings: FCF=-17亿, PE/ROE=17x mismatch

🔧 Repair — SUCCESS | 已修正5项问题

📝 Report — SUCCESS | 报告生成完成
  Output: 14200 chars | Audit: 0/3 retries | Precheck: 通过
```

---

## 工程亮点

### 确定性与概率性边界
财务计算（EPS、PE、ROE、评分、合理价值、稀释比例）是纯 Python 函数。LLM 永不对数字负责。这保证了：同一股票 → 每次评分一致（验证：100 次运行一致性 >96%）。

### 双层验证
- **Code 层（Audit Engine）**：8 项预检（数学一致性、单位有效性、字段完整性）
- **LLM 层（Critics 三路并行）**：语义审查（逻辑、财务解读、行业事实）
- **Code 预检**：所有确定性检查通过则 LLM 审计降级为"仅警告"模式

### RAG 知识系统
8 个行业模板覆盖半导体、电力/能源、医药、消费品、光模块/通信、制造/新能源、金融、地产/建筑。每个模板包含：行业周期、估值规律（含历史 PE 区间）、竞争格局、产业链位置、关键指标。ONNX MiniLM 本地嵌入（无需 API 调用）。检索到的知识注入 Analyst 和 Valuation Agent 上下文。

### 评分引擎 + 成长溢价
合理 PE = 行业 PE 锚点 × 财务质量乘数 × 成长溢价。S 级成长（≥9/10）获 ×1.8 PE 乘数。现金流质量为重资产公司抬底——ROE 低但经营现金流强劲的企业不会被过度折价。

### 框架分歧处理
当保守评分引擎（PE 锚定）与成长叙事（PEG 框架）冲突时，系统不强求一致。而是并排呈现两套框架，各自给出具体买入价、仓位比例和决策框架——让使用者选择自己的投资哲学。

### 数据源分层
三级可配置（FREE/PREMIUM/INSTITUTIONAL），高级数据插槽可插拔（管理层画像、机构持仓、产业链、ESG）。高级源不可用时优雅降级，显示"数据不可用"标记而非幻觉。

---

## 功能清单

**数据层**
- 财务报表（8 期，归母+扣非）
- 估值指标（ROE、毛利率/净利率、EPS、BPS、总股本）
- 实时行情与 K 线图（分时/五日/日/周/月，午休裁剪）
- 公告扫描（20 条，三级优先级，定增检测+自动摊薄修正）
- 板块资金流向与市场宽度
- 板块动量评分（资金流+相对强弱+集中度，四档温度计）

**分析层**
- 10 步投资框架
- 6 维度评分卡
- 情景估值（悲观/基准/乐观+概率加权）
- 催化剂跟踪+证伪条件
- 长期结构性审视（行业终局、管理层、终极风险）

**Agent 层**
- 意图路由（Chat / Deep Analysis / Phantom Hunter）
- 三路并行 Critic（Logic/Financial/Industry）+ Financial 代码层预检
- Repair Agent（自动修正闭环）
- Valuation Agent（公司阶段分类+PE 折价链）
- Company Classifier（纯代码混合权重）
- Audit Engine（8 项检查+代码预检+4 级递进）
- 处理标记系统（防止重试时重复处理）

**输出层**
- 结构化 Markdown 报告（评分卡+估值链+审计摘要）
- 框架分歧 A/B 对比（价值 vs 趋势，价格止损 vs 逻辑止损）
- 执行 Trace（每阶段状态、延迟、发现数）
- 工具调用 + RAG 查询证据页脚

**评估**
- 内置评估页面：N 只股票 × M 次运行 → 评分一致性/字段完整率/工具成功率

---

## Tech Stack

| 层 | 技术 |
|-------|-----------|
| Agent 框架 | LangGraph StateGraph + LangChain tools |
| LLM | DeepSeek（默认）/ OpenAI / Anthropic（可切换） |
| 状态持久化 | LangGraph SqliteSaver |
| 向量数据库 | ChromaDB + ONNX MiniLM-L6-V2（本地嵌入） |
| 数据源 | 新浪财经 + 东方财富 datacenter + 同花顺 10jqka |
| 前端 | Streamlit + Plotly |
| 缓存 | 本地内存（TTL）/ Redis（可切换） |
| 数据分层 | FREE / PREMIUM / INSTITUTIONAL |
| 测试 | 33 项 e2e 测试 |

---

## 快速开始

```bash
git clone <repo-url>
cd finbrain
pip install python-dotenv langgraph langchain langchain-openai chromadb
pip install pdfplumber python-docx plotly streamlit py-mini-racer
```

配置 `configs/.env`：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=<your-api-key>
FINBRAIN_DATA_TIER=FREE
```

运行：

```bash
python run.py                    # CLI
streamlit run frontend/app.py    # Web UI
```

---

## License

MIT
