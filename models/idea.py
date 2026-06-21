"""
Idea数据模型
"""
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import uuid


@dataclass
class Idea:
    """因子想法数据类"""
    content: str  # 想法描述
    use_class: List[str]  # 使用的数据集分类
    use_op: List[str]  # 使用的操作符
    id: str = field(default_factory=lambda: f"i_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "use_class": self.use_class,
            "use_op": self.use_op,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Idea":
        """从字典创建"""
        return cls(
            id=data.get("id"),
            content=data["content"],
            use_class=data["use_class"],
            use_op=data["use_op"],
            created_at=data.get("created_at")
        )
    
    def __hash__(self):
        """用于去重"""
        return hash(self.content)
    
    def __eq__(self, other):
        """相等比较"""
        if isinstance(other, Idea):
            return self.content == other.content
        return False
