"""
Expression数据模型
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class Expression:
    """因子表达式数据类"""
    idea_id: str  # 关联的想法ID
    code: str  # 表达式代码
    id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:8]}")
    valid: bool = True  # 是否通过语法校验
    error_msg: Optional[str] = None  # 错误信息
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "idea_id": self.idea_id,
            "code": self.code,
            "valid": self.valid,
            "error_msg": self.error_msg,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Expression":
        """从字典创建"""
        return cls(
            id=data.get("id"),
            idea_id=data["idea_id"],
            code=data["code"],
            valid=data.get("valid", True),
            error_msg=data.get("error_msg"),
            created_at=data.get("created_at")
        )
    
    def __hash__(self):
        """用于去重"""
        return hash(self.code)
    
    def __eq__(self, other):
        """相等比较"""
        if isinstance(other, Expression):
            return self.code == other.code
        return False
