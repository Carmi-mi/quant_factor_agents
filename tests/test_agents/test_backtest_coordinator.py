"""
测试 BacktestCoordinator 功能
使用 expressions_ASI_MINVOL1M_fundamental17_MATRIX.json 作为输入
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
# test_backtest_coordinator.py -> test_agents -> tests -> quant_factor_agents
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.backtest_coordinator import BacktestCoordinator
from infrastructure.config_loader import ConfigLoader


def load_test_ideas(file_path: str) -> list:
    """加载测试用的 ideas"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("ideas", [])


def downstream_callback(result: dict):
    """下游 Agent 回调函数 - 打印结果"""
    idea_id = result.get("idea_id", "unknown")
    print(f"\n[Callback] 收到 idea '{idea_id}' 的结果")
    print(f"[Callback] 表达式数量: {result.get('expressions_count', 0)}")
    print("-" * 60)


def test_backtest_coordinator():
    """测试 BacktestCoordinator"""
    print("=" * 80)
    print("测试 BacktestCoordinator 功能")
    print("=" * 80)
    
    # 1. 加载配置
    print("\n[1] 加载配置...")
    config_path = project_root / "config" / "settings.yaml"
    config = ConfigLoader.load(str(config_path))
    print(f"   配置加载完成: {config_path}")
    
    # 2. 加载测试数据
    print("\n[2] 加载测试数据...")
    input_file = project_root / "output" / "expressions_ASI_MINVOL1M_other455_GROUP.json"
    ideas = load_test_ideas(str(input_file))
    print(f"   加载了 {len(ideas)} 个 ideas")
    
    # 只取前2个 idea 进行测试（避免测试时间过长）
    test_ideas = ideas[:1]
    print(f"   使用其中 {len(test_ideas)} 个 ideas 进行测试")
    
    for idea in test_ideas:
        print(f"   - {idea.get('idea_id')}: {len(idea.get('expressions', []))} 个表达式")
    
    # 3. 创建 BacktestCoordinator
    print("\n[3] 创建 BacktestCoordinator...")
    max_workers = config.get("backtest", {}).get("max_workers", 2)
    coordinator = BacktestCoordinator(config, llm_service=None, max_workers=max_workers)
    coordinator.set_downstream_callback(downstream_callback)
    print(f"   Coordinator 创建完成，max_workers={max_workers}")
    
    # 5. 运行并行回测
    print("\n[5] 开始并行回测...")
    print("-" * 80)
    
    try:
        results = coordinator.run_parallel_backtest(
            ideas=test_ideas,
            file_info={
                "test": True,
                "input_file": str(input_file),
                "region": "ASI",
                "universe": "MINVOL1M",
                "dataset": "fundamental17",
                "type": "MATRIX"
            }
        )
        
        # 6. 打印结果汇总
        print("\n" + "=" * 80)
        print("回测完成 - 结果汇总")
        print("=" * 80)
        
        summary = results.get("summary", {})
        print(f"\n总 ideas 数: {summary.get('total_ideas', 0)}")
        print(f"总表达式数: {summary.get('total_expressions', 0)}")
        print(f"耗时: {summary.get('duration_seconds', 0):.2f} 秒")
        print(f"并行 workers: {summary.get('parallel_workers', 0)}")
        
    except Exception as e:
        print(f"\n[错误] 回测过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_backtest_coordinator()
    sys.exit(0 if success else 1)
