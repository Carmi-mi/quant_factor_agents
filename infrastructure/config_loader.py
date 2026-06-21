"""
配置加载器
支持YAML配置文件和环境变量覆盖
"""
import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load(path: str = "config/settings.yaml") -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从环境变量覆盖配置
        ConfigLoader._override_from_env(config)
        
        return config
    
    @staticmethod
    def _override_from_env(config: Dict[str, Any], prefix: str = ""):
        """
        从环境变量覆盖配置
        
        Args:
            config: 配置字典
            prefix: 前缀
        """
        for key, value in config.items():
            env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
            
            if isinstance(value, dict):
                ConfigLoader._override_from_env(value, f"{env_key}_")
            else:
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # 尝试类型转换
                    config[key] = ConfigLoader._convert_type(env_value, value)
    
    @staticmethod
    def _convert_type(value: str, original: Any) -> Any:
        """
        类型转换
        
        Args:
            value: 字符串值
            original: 原始值（用于推断类型）
            
        Returns:
            转换后的值
        """
        if isinstance(original, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(original, int):
            return int(value)
        elif isinstance(original, float):
            return float(value)
        elif original is None:
            # 尝试解析null
            if value.lower() in ('null', 'none', ''):
                return None
            return value
        else:
            return value
    
    @staticmethod
    def save(config: Dict[str, Any], path: str):
        """
        保存配置到YAML文件
        
        Args:
            config: 配置字典
            path: 文件路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
