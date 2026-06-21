"""
指标统计服务
提供执行指标统计和报告功能
"""
import time
from typing import Dict, Any, List
from collections import defaultdict


class MetricsService:
    """指标统计服务"""
    
    def __init__(self):
        """初始化指标服务"""
        self.round_stats: List[Dict[str, Any]] = []
        self.agent_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0
        })
        self.start_time = time.time()
    
    def record_round(self, round_num: int, data: Dict[str, Any]):
        """
        记录每轮统计
        
        Args:
            round_num: 轮次编号
            data: 统计数据
        """
        record = {
            "round": round_num,
            "timestamp": time.time(),
            **data
        }
        self.round_stats.append(record)
    
    def record_agent_execution(self, agent_name: str, elapsed: float, success: bool):
        """
        记录Agent执行
        
        Args:
            agent_name: Agent名称
            elapsed: 执行耗时
            success: 是否成功
        """
        self.agent_stats[agent_name]["count"] += 1
        self.agent_stats[agent_name]["total_time"] += elapsed
        if success:
            self.agent_stats[agent_name]["success"] += 1
        else:
            self.agent_stats[agent_name]["failed"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取汇总统计
        
        Returns:
            统计信息字典
        """
        total_rounds = len(self.round_stats)
        
        # 计算累计值
        total_ideas = sum(r.get("ideas", 0) for r in self.round_stats)
        total_exprs = sum(r.get("expressions", 0) for r in self.round_stats)
        total_good = sum(r.get("good", 0) for r in self.round_stats)
        total_bad = sum(r.get("bad", 0) for r in self.round_stats)
        total_improved = sum(r.get("improved", 0) for r in self.round_stats)
        
        # 计算比率
        total_exprs_all = total_good + total_bad + total_improved
        good_rate = total_good / total_exprs_all if total_exprs_all > 0 else 0
        bad_rate = total_bad / total_exprs_all if total_exprs_all > 0 else 0
        improve_rate = total_improved / total_exprs_all if total_exprs_all > 0 else 0
        
        return {
            "total_rounds": total_rounds,
            "uptime_seconds": time.time() - self.start_time,
            "cumulative": {
                "ideas": total_ideas,
                "expressions": total_exprs,
                "good": total_good,
                "bad": total_bad,
                "improved": total_improved
            },
            "rates": {
                "good_rate": good_rate,
                "bad_rate": bad_rate,
                "improve_rate": improve_rate
            },
            "agent_stats": dict(self.agent_stats),
            "recent_rounds": self.round_stats[-10:]  # 最近10轮
        }
    
    def print_summary(self):
        """打印汇总报告"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("执行统计报告")
        print("="*60)
        
        print(f"\n总轮数: {summary['total_rounds']}")
        print(f"运行时间: {summary['uptime_seconds']/3600:.2f} 小时")
        
        print(f"\n累计生成:")
        cum = summary['cumulative']
        print(f"  想法: {cum['ideas']}")
        print(f"  表达式: {cum['expressions']}")
        print(f"  优质因子: {cum['good']}")
        print(f"  劣质因子: {cum['bad']}")
        print(f"  待改进: {cum['improved']}")
        
        print(f"\n比率:")
        rates = summary['rates']
        print(f"  优质率: {rates['good_rate']*100:.1f}%")
        print(f"  劣质率: {rates['bad_rate']*100:.1f}%")
        print(f"  改进率: {rates['improve_rate']*100:.1f}%")
        
        if summary['agent_stats']:
            print(f"\nAgent统计:")
            for name, stats in summary['agent_stats'].items():
                success_rate = stats['success'] / stats['count'] * 100 if stats['count'] > 0 else 0
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {name:20s}: 执行{stats['count']:4d}次, 成功率{success_rate:5.1f}%, 平均{avg_time:.2f}s")
        
        print("="*60)
    
    def get_recent_rounds(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近N轮统计
        
        Args:
            n: 轮数
            
        Returns:
            最近N轮统计
        """
        return self.round_stats[-n:]
    
    def export_to_file(self, filepath: str):
        """
        导出统计到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        
        summary = self.get_summary()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[MetricsService] 统计已导出到: {filepath}")
