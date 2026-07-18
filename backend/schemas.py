"""
FinBrain 结构化输出 Schema

用于 LangChain / LangGraph 节点的 with_structured_output，
替代原来脆弱的 json.loads + 正则兜底。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ScoreItem(BaseModel):
    """评分维度条目"""
    得分: Optional[float] = Field(None, description="0-10 分，缺失可填 null")
    依据: Optional[str] = Field(None, description="该维度得分的依据说明")
    现金流标签: Optional[str] = Field(None, description="财务健康维度的现金流标签，如 🟢优秀")
    现金流严重度: Optional[int] = Field(None, description="现金流问题严重度 0-5")
    FCF预警: Optional[str] = Field(None, description="自由现金流预警说明")


class CompanyProfile(BaseModel):
    """公司画像"""
    主营业务: Optional[str] = Field(None, description="公司主营业务描述")
    公司类型: Optional[str] = Field(None, description="价值型/成长型/周期型/困境反转型/事件驱动型")
    生命周期: Optional[str] = Field(None, description="初创/成长/成熟/衰退/周期底部回升等")


class Moat(BaseModel):
    """竞争优势 / 护城河"""
    核心资产: Optional[str] = None
    护城河来源: Optional[List[str]] = Field(default_factory=list, alias="护城河来源")
    护城河类型: Optional[List[str]] = Field(default_factory=list, alias="护城河类型")
    复制难度: Optional[str] = None
    持续时间: Optional[str] = None
    可持续性: Optional[str] = None
    竞争格局: Optional[str] = None
    毛利率归因: Optional[str] = None


class Scenario(BaseModel):
    """情景估值条目"""
    价格: Optional[float] = Field(None, description="该情景下的目标价格（元）")
    概率: Optional[str] = Field(None, description="概率，如 20%")
    假设: Optional[str] = Field(None, description="该情景的核心假设")
    EPS: Optional[float] = None
    PE: Optional[float] = None


class ValuationLevel(BaseModel):
    """估值水位"""
    PE: Optional[str] = None
    PB: Optional[str] = None
    市值: Optional[str] = None
    前瞻PE: Optional[str] = None


class Conclusion(BaseModel):
    """综合结论"""
    总评: Optional[str] = None
    买入策略: Optional[str] = None
    持有策略: Optional[str] = None
    卖出条件: Optional[str] = None
    预期收益: Optional[str] = None
    持仓周期: Optional[str] = None


class InvestmentRating(BaseModel):
    """投资决策评级（代码层会覆盖部分字段）"""
    评级: Optional[str] = Field(None, description="BUY / HOLD / SELL")
    合理价值: Optional[Any] = None
    实际安全边际: Optional[str] = None
    安全边际要求: Optional[str] = None
    买入区间: Optional[str] = None
    估值差距: Optional[str] = None
    加权总分: Optional[float] = None
    置信度: Optional[str] = None
    当前价格: Optional[float] = None
    估值明细: Optional[Dict[str, Any]] = Field(default_factory=dict)


class KeySignal(BaseModel):
    """关键信号"""
    信号: Optional[str] = None
    数据: Optional[str] = None
    解读: Optional[str] = None


class Catalyst(BaseModel):
    """催化剂"""
    正面: Optional[List[str]] = Field(default_factory=list)
    负面: Optional[List[str]] = Field(default_factory=list)
    强度: Optional[str] = None


class MarketExpectation(BaseModel):
    """市场预期拆解"""
    当前估值隐含的增长率: Optional[str] = None
    市场主要担忧: Optional[List[str]] = Field(default_factory=list)
    可能的预期差: Optional[str] = None


class AnalystOutput(BaseModel):
    """
    Analyst Agent 的完整输出。

    所有字段均为 Optional，允许 LLM 在部分数据缺失时仍输出合法对象。
    代码层会在下游覆盖评分、投资评级、估值水位等字段。
    """
    代码: Optional[str] = None
    名称: Optional[str] = None
    当前股价: Optional[float] = None
    公司画像: Optional[CompanyProfile] = None
    竞争优势: Optional[Moat] = None
    投资逻辑链: Optional[str] = None
    估值方法: Optional[str] = None
    估值框架: Optional[str] = None
    评分: Optional[Dict[str, ScoreItem]] = Field(default_factory=dict)
    亮点: Optional[List[str]] = Field(default_factory=list)
    风险: Optional[List[str]] = Field(default_factory=list)
    业绩驱动力: Optional[str] = None
    关键信号: Optional[List[KeySignal]] = Field(default_factory=list)
    估值水位: Optional[ValuationLevel] = None
    情景估值: Optional[Dict[str, Scenario]] = Field(default_factory=dict)
    概率加权价值: Optional[Any] = None
    观察指标: Optional[List[str]] = Field(default_factory=list)
    催化剂: Optional[Catalyst] = None
    市场预期拆解: Optional[MarketExpectation] = None
    市场已定价: Optional[str] = None
    证伪条件: Optional[List[str]] = Field(default_factory=list)
    操作建议: Optional[str] = None
    止损: Optional[str] = None
    执行状态: Optional[str] = None
    框架分歧: Optional[str] = None
    结论: Optional[Any] = None  # 可能是字符串或 Conclusion dict
    综合结论: Optional[Conclusion] = None
    投资评级: Optional[InvestmentRating] = None
    定增信息: Optional[Dict[str, Any]] = None
    公告: Optional[Dict[str, Any]] = None
    机构共识: Optional[Dict[str, Any]] = None
    校验: Optional[List[str]] = Field(default_factory=list)
    偏见修正: Optional[str] = None

    # 内部元数据，LLM 通常不输出，但 schema 保留兼容性
    _flash_data: Optional[Dict[str, Any]] = None
    _data_tier: Optional[str] = None
    _premium_data_notes: Optional[List[str]] = None
    _unavailable_premium: Optional[List[str]] = None
    _fwd_pe_note: Optional[str] = None
    _dilution_coefficient: Optional[float] = None
    _dilution_shares: Optional[float] = None
    _dilution_fund_amount: Optional[float] = None
    _scenario_check: Optional[Dict[str, Any]] = None


class ValuationOutput(BaseModel):
    """估值框架 Agent 输出"""
    公司阶段: Optional[str] = Field(None, description="如成长早期/成长期/成熟期/周期底部/困境反转")
    适用框架: Optional[List[str]] = Field(default_factory=list, description="推荐估值方法列表")
    估值参考: Optional[Dict[str, str]] = Field(default_factory=dict, description="各估值方法参考区间")
    阶段说明: Optional[str] = None


class CriticOutput(BaseModel):
    """单个 Critic Agent 输出"""
    通过: Optional[bool] = Field(True, description="是否通过审查")
    逻辑漏洞: Optional[List[str]] = Field(default_factory=list)
    财务误读: Optional[List[str]] = Field(default_factory=list)
    行业误述: Optional[List[str]] = Field(default_factory=list)
    建议: Optional[str] = None


class AuditIssue(BaseModel):
    """审计发现的问题"""
    级别: Optional[str] = Field(None, description="❌ 严重 / ⚠️ 警告")
    描述: Optional[str] = None
    修正建议: Optional[str] = None


class AuditOutput(BaseModel):
    """审计 Agent 输出"""
    通过: Optional[bool] = Field(True, description="是否通过审计")
    问题: Optional[List[AuditIssue]] = Field(default_factory=list)


class RepairOutput(BaseModel):
    """Repair Agent 输出，结构与 AnalystOutput 一致"""
    修正后的JSON: Optional[AnalystOutput] = Field(None, alias="修正后的JSON")
    修正说明: Optional[str] = None

    class Config:
        populate_by_name = True
