"""
故障目录 — 审查链 fault-injection。

每条 Fault 描述一种"已知错误类型"（映射 README/审计报告中已修复的真实缺陷），
包含：
    golden       干净数据（检测器必须零误报通过）
    faulted      注入错误后的数据（检测器必须非空检出）
    detect       运行对应守卫，返回违规描述列表（空列表 = 未检出）

全部纯代码、不调 LLM、不访问网络 —— 可离线进 CI。

守卫命名约定（与代码对应）：
    consistency.filter_untraceable_issues    — Critic 幻觉拦截（发现级）
    consistency.has_new_hard_numbers          — Repair 新数字校验
    consistency.strip_untraceable_sentences  — 股评句子级幻觉拦截
    agent._financial_code_check               — Financial Critic 代码层预检
    calc_engine.verify_report_text            — 报告数字回溯验证
    consistency.check_invariants              — 一致性不变量 INV-1~11
    agent._validate_scenarios                 — 情景估值代码校验
"""

import json
from dataclasses import dataclass, field


@dataclass
class Fault:
    id: str
    category: str          # critics | repair | report | audit
    name: str              # 中文名
    description: str       # 对应 README/审计报告中的真实缺陷
    expected_guard: str    # 期望拦截它的守卫（人类可读）
    golden: object         # 干净 payload
    faulted: object        # 注入错误后的 payload
    detect: object         # callable(payload) -> list[str]
    # 覆盖哪些文档缺陷（README 已修复缺陷 / FIX 项），供定位
    maps_to: list = field(default_factory=list)


# ============================================================
#  黄金数据（托伦斯 301583 口径，与 tests/test_tuolunsi_regression.py 一致）
# ============================================================

GOLDEN_ANALYSIS = {
    "代码": "301583",
    "名称": "托伦斯",
    "估值水位": {"PE": "337倍", "PB": "17.6", "市值": "322.7亿", "前瞻PE": "季节性失真"},
    "投资评级": {"评级": "HOLD", "合理价值": 31.9, "买入区间": "≤23.9元", "当前价格": 174.0},
    "情景估值": {
        "悲观": {"价格": 13.5, "EPS": 0.45, "PE": 30, "概率": "25%"},
        "基准": {"价格": 33.0, "EPS": 0.55, "PE": 60, "概率": "55%"},
        "乐观": {"价格": 52.0, "EPS": 0.65, "PE": 80, "概率": "20%"},
        "概率加权价值": 31.9,
    },
    "操作建议": "当前174元估值偏贵，建议等待股价回归合理区间（约24-32元）后再评估",
    "结论": {"总评": "基本面尚可但估值偏高", "买入策略": "≤23.9元再建仓"},
    "_eps_ttm": 0.516,
    "_rev_growth": 25.0,
    "_cyclical": 0.2,
}

# 干净报告文本：每个标签值必须与黄金 CalcTable 登记值容差内一致
GOLDEN_REPORT_TEXT = (
    "市场隐含未来3年利润年复合增速需达102%以上才能消化当前估值。\n"
    "当前市值: 322.7亿，PE: 337倍，PB: 17.6。\n"
    "合理价值: 31.9元，安全买入价≤23.9元，概率加权价值: 31.9元。"
)


def _golden_calc_table():
    from backend.calc_engine import CalcTable
    ct = CalcTable()
    ct.register("市值", 322.7, formula="174.00 × 1.8547亿股 = 322.7亿",
                inputs={"price": 174.0, "total_shares": 1.85473692e8}, source="quote")
    ct.register("PE", 337, formula="322.7亿 / 0.9818亿 = 328.6（取整337）",
                inputs={"mktcap": 322.7e8, "net": 0.9818e8})
    ct.register("PB", 17.6, formula="174.00 / 9.87 = 17.63",
                inputs={"price": 174.0, "bps": 9.87})
    ct.register("合理价值", 31.9, formula="概率加权价值")
    ct.register("安全买入价", 23.9, formula="合理价值 × 0.75")
    ct.register("隐含增速", 102.0, formula="(322.7亿÷40÷0.98亿)^(1/3)-1 = 102%",
                inputs={"mktcap": 322.7e8, "target_pe": 40.0, "net": 0.98e8, "years": 3})
    ct.register("概率加权价值", 31.9, formula="Σ 情景价格×概率")
    return ct


def _mk_analysis_text(overrides: dict) -> str:
    """构造分析 JSON 字符串，overrides 覆盖黄金字段（注入用）。"""
    item = dict(GOLDEN_ANALYSIS)
    item.update(overrides)
    return json.dumps(item, ensure_ascii=False)


# ============================================================
#  Fault 构造
# ============================================================

def build_faults() -> list[Fault]:
    faults: list[Fault] = []

    # ---- critics 层：Critic 幻觉拦截 ----
    def _c1_detect(payload: dict):
        """payload = {"source": 原文+采集数据, "issues": Critic 发现列表}。
        返回被丢弃（含无出处数字）的发现。"""
        from backend.consistency import filter_untraceable_issues
        _, dropped = filter_untraceable_issues(payload["issues"], payload["source"])
        return dropped

    # C1 故障 = Critic 发现里含编造数字（986倍无出处）
    faults.append(Fault(
        id="C1",
        category="critics",
        name="Critic 发现含编造硬数字",
        description="发现引用原文不存在的数字（986倍），grounding 校验应丢弃（对应 README: 幻觉拦截）",
        expected_guard="consistency.filter_untraceable_issues",
        golden={"source": _mk_analysis_text({}),
                "issues": ["操作建议与评级方向矛盾"]},
        faulted={"source": _mk_analysis_text({}),
                 "issues": ["PE 986倍明显泡沫，比行业均值高20倍",
                            "操作建议与评级方向矛盾"]},
        detect=_c1_detect,
        maps_to=["幻觉拦截：数字必须可溯源至原文", "前瞻 PE 986 倍季节性失真"],
    ))

    def _c2_detect(text: str):
        from backend.agent import _financial_code_check
        return [i for i in _financial_code_check(text) if "FCF" in i]

    # FCF 为负（CAPEX>CFO）却宣称"利润含金量极高"
    fcf_golden = _mk_analysis_text({
        "亮点": ["利润含金量高：经营现金流1.89亿覆盖资本开支0.59亿"],
    })
    fcf_faulted = _mk_analysis_text({
        "亮点": ["利润含金量极高：经营现金流0.59亿被资本开支1.89亿吞噬"],
    })
    faults.append(Fault(
        id="C2",
        category="critics",
        name="FCF 为负却宣称利润含金量极高",
        description="资本开支>经营现金流 时仍用『利润含金量极高』定性，代码层预检应报 FCF 预警",
        expected_guard="agent._financial_code_check (FCF)",
        golden=fcf_golden,
        faulted=fcf_faulted,
        detect=_c2_detect,
        maps_to=["FCF 预警", "现金流口径混用"],
    ))

    def _c3_detect(text: str):
        from backend.agent import _financial_code_check
        return [i for i in _financial_code_check(text) if "PE" in i and "ROE" in i]

    pe_roe_golden = _mk_analysis_text({
        "操作建议": "当前PE: 20倍估值合理",
        "评分": {"盈利能力": {"得分": 7, "依据": "ROE 15%，盈利能力良好"}},
    })
    pe_roe_faulted = _mk_analysis_text({
        "操作建议": "当前PE: 337倍严重高估",
        "评分": {"盈利能力": {"得分": 3, "依据": "ROE 5.5%，盈利能力偏弱"}},
    })
    faults.append(Fault(
        id="C3",
        category="critics",
        name="PE/ROE 严重不匹配",
        description="高 PE 配低 ROE 未加注『需极高增速支撑』，代码层预检应检出",
        expected_guard="agent._financial_code_check (PE/ROE)",
        golden=pe_roe_golden,
        faulted=pe_roe_faulted,
        detect=_c3_detect,
        maps_to=["PE/ROE 匹配度", "同一乐观 PE 出现三个数"],
    ))

    # ---- repair 层：Repair 引入新数字 ----
    def _r_detect(payload: tuple[str, str]):
        from backend.consistency import has_new_hard_numbers
        return [f"{v}{u}" for v, u in has_new_hard_numbers(payload[0], payload[1])]

    orig_raw = _mk_analysis_text({})
    # R1: Repair 新增原文不存在的数字（目标价 45 元）
    r1_clean = (orig_raw, orig_raw)
    r1_faulted = (orig_raw, orig_raw.replace('"结论": {"总评": "基本面尚可但估值偏高"',
                                            '"结论": {"总评": "估值偏高，目标价45元"}'))
    faults.append(Fault(
        id="R1",
        category="repair",
        name="Repair 引入原文不存在的新数字",
        description="Repair 不得新增任何原始数据中不存在的硬数字，违反则整体作废回退原文",
        expected_guard="consistency.has_new_hard_numbers",
        golden=r1_clean,
        faulted=r1_faulted,
        detect=_r_detect,
        maps_to=["Repair 新数字作废"],
    ))

    # R2: Repair 篡改既有硬数字（337倍 → 986倍，编造）
    r2_clean = (orig_raw, orig_raw)
    r2_faulted = (orig_raw, orig_raw.replace('"PE": "337倍"', '"PE": "986倍"'))
    faults.append(Fault(
        id="R2",
        category="repair",
        name="Repair 篡改既有硬数字",
        description="把原文 PE 337倍 改成 986倍（无出处），应被新数字校验作废",
        expected_guard="consistency.has_new_hard_numbers",
        golden=r2_clean,
        faulted=r2_faulted,
        detect=_r_detect,
        maps_to=["Repair 引入新数字作废", "前瞻 PE 986 倍季节性失真"],
    ))

    # ---- report 层：报告数字回溯验证 ----
    def _p_detect(payload: tuple[str, object]):
        from backend.calc_engine import verify_report_text
        return verify_report_text(payload[0], payload[1])

    _ct = _golden_calc_table()
    # P1: 报告隐含增速写 40%，登记值为 102%
    faults.append(Fault(
        id="P1",
        category="report",
        name="报告隐含增速与登记值不符",
        description="LLM 写『年复合增速40%』而登记值102%（托伦斯 B1 真实缺陷），回溯验证须报违规",
        expected_guard="calc_engine.verify_report_text",
        golden=(GOLDEN_REPORT_TEXT, _ct),
        faulted=(GOLDEN_REPORT_TEXT.replace("102%", "40%"), _ct),
        detect=_p_detect,
        maps_to=["计算可回溯：报告数字回溯验证", "B1 隐含增速 40% vs 102%"],
    ))

    # P2: 报告合理价值写 60 元，登记值 31.9
    faults.append(Fault(
        id="P2",
        category="report",
        name="报告合理价值与登记值不符",
        description="LLM 自行改写合理价值（60元 vs 登记 31.9），回溯验证须报违规",
        expected_guard="calc_engine.verify_report_text",
        golden=(GOLDEN_REPORT_TEXT, _ct),
        faulted=(GOLDEN_REPORT_TEXT.replace("合理价值: 31.9元", "合理价值: 60元"), _ct),
        detect=_p_detect,
        maps_to=["计算可回溯", "同一个乐观 PE 出现三个数"],
    ))

    # P3: 同一指标全文出现两个不一致值
    faults.append(Fault(
        id="P3",
        category="report",
        name="同一指标全文多处不一致",
        description="合理价值 31.9 与 60 同时出现在全文，回溯验证须检出全文不一致",
        expected_guard="calc_engine.verify_report_text",
        golden=(GOLDEN_REPORT_TEXT, _ct),
        faulted=(GOLDEN_REPORT_TEXT + " 合理价值: 60元", _ct),
        detect=_p_detect,
        maps_to=["同一个乐观 PE 出现三个数", "全文唯一值约束"],
    ))

    # ---- audit 层：一致性不变量 ----
    def _inv_detect(item: dict):
        from backend.consistency import check_invariants
        return [f"{i.rule}:{i.detail}" for i in check_invariants(item)]

    # A1: INV-1 合理区间上限高于合理价值×1.1
    a1_clean = dict(GOLDEN_ANALYSIS)
    a1_faulted = dict(GOLDEN_ANALYSIS)
    a1_faulted["操作建议"] = "建议等待股价回归合理区间（约30-60元）后再评估"
    faults.append(Fault(
        id="A1",
        category="audit",
        name="合理区间与合理价值口径冲突 (INV-1)",
        description="区间上限60元 > 合理价值31.9×1.1，决策区间与自身估值矛盾",
        expected_guard="consistency.check_invariants (INV-1)",
        golden=a1_clean,
        faulted=a1_faulted,
        detect=_inv_detect,
        maps_to=["风险清单被清空", "评级-操作一致性", "B4 合理区间(30-60元)高于自身合理价值"],
    ))

    # A2: INV-7 评级 SELL 但营收高增长
    a2_clean = dict(GOLDEN_ANALYSIS)
    a2_faulted = dict(GOLDEN_ANALYSIS)
    a2_faulted["投资评级"] = dict(GOLDEN_ANALYSIS["投资评级"], 评级="SELL")
    a2_faulted["_rev_growth"] = 40.0
    faults.append(Fault(
        id="A2",
        category="audit",
        name="评级 SELL 但营收高增长 (INV-7)",
        description="高增长转型期用静态 PE 卖出可能犯错，断路器应告警",
        expected_guard="consistency.check_invariants (INV-7)",
        golden=a2_clean,
        faulted=a2_faulted,
        detect=_inv_detect,
        maps_to=["评级-操作一致性", "框架分歧检测"],
    ))

    # A3: INV-3 情景价格倒挂
    a3_clean = dict(GOLDEN_ANALYSIS)
    a3_faulted = dict(GOLDEN_ANALYSIS)
    a3_faulted["情景估值"] = dict(GOLDEN_ANALYSIS["情景估值"])
    a3_faulted["情景估值"] = dict(a3_faulted["情景估值"],
                                  **{"悲观": dict(GOLDEN_ANALYSIS["情景估值"]["悲观"], 价格=52.0)})
    faults.append(Fault(
        id="A3",
        category="audit",
        name="情景价格倒挂 (INV-3)",
        description="悲观 52 > 基准 33，情景单调性被破坏",
        expected_guard="consistency.check_invariants (INV-3)",
        golden=a3_clean,
        faulted=a3_faulted,
        detect=_inv_detect,
        maps_to=["情景估值三错二", "情景单调性"],
    ))

    return faults
