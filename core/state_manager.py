"""
状态管理器
管理全局状态和缓存，支持持久化
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class GlobalState:
    """全局状态数据类"""
    iteration_count: int = 0
    total_good: int = 0
    total_bad: int = 0
    total_improved: int = 0
    cache: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GlobalState":
        """从字典创建"""
        return cls(**data)


class StateManager:
    """状态管理器"""
    
    def __init__(self, persist_path: str = "state.json"):
        """
        初始化状态管理器
        
        Args:
            persist_path: 状态持久化文件路径
        """
        self.persist_path = persist_path
        self._state = GlobalState()
        self._load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取状态值
        
        Args:
            key: 键名
            default: 默认值
            
        Returns:
            状态值
        """
        return getattr(self._state, key, default)
    
    def set(self, key: str, value: Any, auto_save: bool = True):
        """
        设置状态值
        
        Args:
            key: 键名
            value: 值
            auto_save: 是否自动保存
        """
        setattr(self._state, key, value)
        if auto_save:
            self._save()
    
    def increment(self, key: str, delta: int = 1, auto_save: bool = True):
        """
        计数器增加
        
        Args:
            key: 键名
            delta: 增量
            auto_save: 是否自动保存
        """
        current = self.get(key, 0)
        self.set(key, current + delta, auto_save)
    
    def update_cache(self, key: str, value: Any, auto_save: bool = True):
        """
        更新缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            auto_save: 是否自动保存
        """
        self._state.cache[key] = value
        if auto_save:
            self._save()
    
    def get_cache(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值
        """
        return self._state.cache.get(key, default)
    
    def get_all_cache(self) -> Dict[str, Any]:
        """获取所有缓存"""
        return self._state.cache.copy()
    
    def _save(self):
        """保存状态到文件"""
        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self._state.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[StateManager] 保存状态失败: {e}")
    
    def _load(self):
        """从文件加载状态"""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._state = GlobalState.from_dict(data)
                print(f"[StateManager] 已加载状态: {self.persist_path}")
            except Exception as e:
                print(f"[StateManager] 加载状态失败: {e}，使用默认状态")
                self._state = GlobalState()
        else:
            print(f"[StateManager] 状态文件不存在，创建新状态")
            self._state = GlobalState()
    
    def reset(self):
        """重置状态"""
        self._state = GlobalState()
        self._save()
        print("[StateManager] 状态已重置")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "iteration_count": self._state.iteration_count,
            "total_good": self._state.total_good,
            "total_bad": self._state.total_bad,
            "total_improved": self._state.total_improved,
            "cache_keys": list(self._state.cache.keys())
        }
