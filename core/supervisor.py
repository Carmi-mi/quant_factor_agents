"""
Supervisor 调度器
负责任务调度、重试、执行历史记录
"""
import time
from typing import Dict, Any, Callable, Optional
from enum import Enum
from collections import defaultdict


class AgentStatus(Enum):
    """Agent执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class Supervisor:
    """Agent调度器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        初始化调度器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.agents: Dict[str, Callable] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.execution_history: list = []
        self.current_agent: Optional[str] = None
        self._agent_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0
        })
    
    def register(self, name: str, agent: Callable):
        """
        注册Agent
        
        Args:
            name: Agent名称
            agent: Agent实例（必须实现run方法）
        """
        self.agents[name] = agent
        print(f"[Supervisor] 注册Agent: {name}")
    
    def run_agent(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent，带重试机制
        
        Args:
            name: Agent名称
            input_data: 输入数据
            
        Returns:
            Agent执行结果
            
        Raises:
            ValueError: Agent未注册
            Exception: 达到最大重试次数后仍失败
        """
        if name not in self.agents:
            raise ValueError(f"未注册的Agent: {name}")
        
        agent = self.agents[name]
        last_error = None
        
        for attempt in range(self.max_retries):
            self.current_agent = name
            start_time = time.time()
            
            try:
                print(f"[Supervisor] 执行: {name} (尝试 {attempt + 1}/{self.max_retries})")
                
                # 执行Agent
                result = agent.run(input_data)
                elapsed = time.time() - start_time
                
                # 记录执行历史
                self._record_execution(name, AgentStatus.SUCCESS, attempt + 1, elapsed)
                
                # 更新统计
                self._agent_stats[name]["count"] += 1
                self._agent_stats[name]["success"] += 1
                self._agent_stats[name]["total_time"] += elapsed
                
                print(f"[Supervisor] 完成: {name} (耗时 {elapsed:.2f}s)")
                
                return result
                
            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                
                # 记录执行历史
                self._record_execution(name, AgentStatus.FAILED, attempt + 1, elapsed, str(e))
                
                # 更新统计
                self._agent_stats[name]["count"] += 1
                self._agent_stats[name]["failed"] += 1
                
                print(f"[Supervisor] 错误: {name} - {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"[Supervisor] {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[Supervisor] {name} 已达到最大重试次数")
                    raise last_error
            
            finally:
                self.current_agent = None
    
    def _record_execution(self, agent_name: str, status: AgentStatus, 
                         attempt: int, elapsed: float, error: str = None):
        """记录执行历史"""
        record = {
            "agent": agent_name,
            "status": status.value,
            "attempt": attempt,
            "elapsed_time": elapsed,
            "timestamp": time.time()
        }
        if error:
            record["error"] = error
        
        self.execution_history.append(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取执行统计
        
        Returns:
            统计信息字典
        """
        total = len(self.execution_history)
        success = sum(1 for h in self.execution_history if h["status"] == "success")
        failed = total - success
        
        avg_time = 0.0
        if total > 0:
            avg_time = sum(h.get("elapsed_time", 0) for h in self.execution_history) / total
        
        return {
            "total_executions": total,
            "success_count": success,
            "failed_count": failed,
            "success_rate": success / total if total > 0 else 0.0,
            "avg_execution_time": avg_time,
            "agent_stats": dict(self._agent_stats)
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n=== Supervisor 执行统计 ===")
        print(f"总执行次数: {stats['total_executions']}")
        print(f"成功: {stats['success_count']} | 失败: {stats['failed_count']}")
        print(f"成功率: {stats['success_rate']*100:.1f}%")
        print(f"平均执行时间: {stats['avg_execution_time']:.2f}s")
        
        if stats['agent_stats']:
            print("\n各Agent统计:")
            for name, agent_stat in stats['agent_stats'].items():
                success_rate = agent_stat['success'] / agent_stat['count'] * 100 if agent_stat['count'] > 0 else 0
                avg_time = agent_stat['total_time'] / agent_stat['count'] if agent_stat['count'] > 0 else 0
                print(f"  {name}: 执行{agent_stat['count']}次, 成功率{success_rate:.1f}%, 平均{avg_time:.2f}s")
    
    def get_current_agent(self) -> Optional[str]:
        """获取当前正在执行的Agent"""
        return self.current_agent
    
    def is_running(self) -> bool:
        """是否有Agent正在执行"""
        return self.current_agent is not None
