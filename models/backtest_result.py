"""
BacktestResult数据模型
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class BacktestMetrics:
    """回测指标"""
    sharpe: float  # 夏普比率
    return_pct: float  # 收益率
    max_drawdown: float  # 最大回撤
    volatility: float = 0.0  # 波动率
    turnover: float = 0.0  # 换手率
    fitness: float = 0.0  # 适应度分数
    
    def is_good(self, threshold: dict) -> bool:
        """
        判断是否优质
        
        Args:
            threshold: 阈值配置
            
        Returns:
            是否优质
        """
        return (
            self.sharpe >= threshold.get("min_sharpe", 1.0) and
            self.return_pct >= threshold.get("min_return", 0.05) and
            self.max_drawdown <= threshold.get("max_drawdown", 0.15)
        )
    
    def is_bad(self, threshold: dict) -> bool:
        """
        判断是否劣质
        
        Args:
            threshold: 阈值配置
            
        Returns:
            是否劣质
        """
        return self.sharpe <= threshold.get("max_sharpe", 0.3)
    
    def need_improve(self, threshold: dict) -> bool:
        """
        判断是否需要改进
        
        Args:
            threshold: 阈值配置
            
        Returns:
            是否需要改进
        """
        return (
            not self.is_good(threshold) and
            not self.is_bad(threshold)
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "sharpe": self.sharpe,
            "return_pct": self.return_pct,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "turnover": self.turnover,
            "fitness": self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BacktestMetrics":
        """从字典创建"""
        return cls(
            sharpe=data["sharpe"],
            return_pct=data["return_pct"],
            max_drawdown=data["max_drawdown"],
            volatility=data.get("volatility", 0.0),
            turnover=data.get("turnover", 0.0),
            fitness=data.get("fitness", 0.0)
        )


@dataclass
class BacktestResult:
    """回测结果"""
    expr_id: str  # 表达式ID
    metrics: BacktestMetrics  # 回测指标
    status: str = "unknown"  # 状态: good, bad, need_improve
    defect_reason: Optional[str] = None  # 缺陷原因
    tested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def classify(self, threshold: dict):
        """
        根据阈值分类
        
        Args:
            threshold: 阈值配置
        """
        if self.metrics.is_good(threshold):
            self.status = "good"
        elif self.metrics.is_bad(threshold):
            self.status = "bad"
        else:
            self.status = "need_improve"
            # 自动判断缺陷原因
            self._infer_defect_reason(threshold)
    
    def _infer_defect_reason(self, threshold: dict):
        """推断缺陷原因"""
        reasons = []
        if self.metrics.sharpe < threshold.get("min_sharpe", 1.0):
            reasons.append("收益不足")
        if self.metrics.max_drawdown > threshold.get("max_drawdown", 0.15):
            reasons.append("回撤过大")
        if self.metrics.volatility > 0.3:
            reasons.append("波动过高")
        
        self.defect_reason = ";".join(reasons) if reasons else "综合表现一般"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "expr_id": self.expr_id,
            "metrics": self.metrics.to_dict(),
            "status": self.status,
            "defect_reason": self.defect_reason,
            "tested_at": self.tested_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BacktestResult":
        """从字典创建"""
        return cls(
            expr_id=data["expr_id"],
            metrics=BacktestMetrics.from_dict(data["metrics"]),
            status=data.get("status", "unknown"),
            defect_reason=data.get("defect_reason"),
            tested_at=data.get("tested_at")
        )
