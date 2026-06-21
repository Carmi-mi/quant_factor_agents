"""
Agent基类定义
所有Agent必须继承此类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class Agent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent任务
        
        Args:
            input_data: 输入数据
            
        Returns:
            输出数据字典，必须包含 'status' 字段
            status: 'success' | 'error'
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            required_fields: 必需字段列表
            
        Returns:
            是否验证通过
        """
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"缺少必需字段: {field}")
        return True
