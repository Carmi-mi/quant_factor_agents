"""
Test for ImprovementAgent
验证 alpha 改进功能，使用 backtest_results 作为输入
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory
from agents.improvement_agent import ImprovementAgent


def load_backtest_results(filepath):
    """Load backtest results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("ImprovementAgent Test - Alpha Improvement")
    print("=" * 70)

    # ==================== 配置参数 ====================
    backtest_results_file = "backtest_results_ASI_MINVOL1M_fundamental17_MATRIX.json"
    max_rounds = 2  # 测试时每轮改进次数
    target_sharpe = 1.25
    # =================================================

    # 构建文件路径
    output_dir = os.path.join(project_root, "output")
    backtest_results_path = os.path.join(output_dir, backtest_results_file)

    print(f"\n[Config]")
    print(f"  Backtest results: {backtest_results_path}")
    print(f"  Max rounds per alpha: {max_rounds}")
    print(f"  Target Sharpe: {target_sharpe}")

    # 检查文件是否存在
    if not os.path.exists(backtest_results_path):
        print(f"\n[Error] Backtest results file not found: {backtest_results_path}")
        print(f"  Please run backtest first or check the file path.")
        return

    # 加载配置
    print("\n[Loading Config]")
    config_path = os.path.join(project_root, "config", "settings.yaml")
    config = ConfigLoader.load(config_path)
    print(f"  Config loaded from: {config_path}")

    # 初始化 LLM Service
    print("\n[Initializing LLM Service]")
    llm_service = LLMServiceFactory.create_from_config(config)
    print(f"  LLM Service: {config.get('llm', {}).get('provider', 'default')}")

    # 加载回测结果
    print(f"\n[Loading Backtest Results]")
    backtest_results = load_backtest_results(backtest_results_path)
    print(f"  Status: {backtest_results.get('status', 'unknown')}")
    print(f"  File info: {backtest_results.get('file_info', {})}")

    final_alphas = backtest_results.get("final_alphas", [])
    print(f"  Total ideas: {len(final_alphas)}")

    if not final_alphas:
        print("\n[Error] No alphas found in backtest results")
        return

    # 显示第一个 idea 的信息
    first_idea = final_alphas[0]
    print(f"\n[First Idea]")
    print(f"  Idea ID: {first_idea.get('idea_id')}")
    print(f"  Content: {first_idea.get('idea_content', '')[:100]}...")
    print(f"  Alphas count: {len(first_idea.get('alphas', []))}")

    # 初始化 ImprovementAgent
    print("\n[Initializing ImprovementAgent]")
    try:
        agent = ImprovementAgent(config, llm_service)
        print(f"  Agent initialized successfully")
        print(f"  Max rounds: {agent.max_rounds}")
        print(f"  Target Sharpe: {agent.target_sharpe}")
    except Exception as e:
        print(f"  [Error] Failed to initialize agent: {e}")
        return

    # 准备输入数据
    input_data = {
        "backtest_results": backtest_results,
        "max_rounds": max_rounds,
        "target_sharpe": target_sharpe,
        "verbose": True
    }

    # 运行改进
    print("\n" + "=" * 70)
    print("Running Improvement")
    print("=" * 70)

    try:
        result = agent.run(input_data)

        # 显示结果
        print("\n" + "=" * 70)
        print("Improvement Results")
        print("=" * 70)

        print(f"\n[Status]")
        print(f"  Overall status: {result.get('status', 'unknown')}")

        improved_alphas = result.get("improved_alphas", [])
        failed_alphas = result.get("failed_alphas", [])

        print(f"\n[Summary]")
        print(f"  Improved: {len(improved_alphas)}")
        print(f"  Failed: {len(failed_alphas)}")

        # 显示改进详情
        if improved_alphas:
            print(f"\n[Improved Alphas Details]")
            for idx, alpha_result in enumerate(improved_alphas, 1):
                print(f"\n  [{idx}] Alpha ID: {alpha_result.get('alpha_id')}")
                print(f"      Status: {alpha_result.get('status')}")
                print(f"      Initial Sharpe: {alpha_result.get('initial_sharpe', 0):.4f}")
                print(f"      Final Sharpe: {alpha_result.get('final_sharpe', 0):.4f}")
                print(f"      Improvement: {alpha_result.get('sharpe_improvement', 0):.4f}")
                print(f"      Rounds completed: {alpha_result.get('rounds_completed', 0)}")

                # 显示改进历史
                history = alpha_result.get("improvement_history", [])
                if history:
                    print(f"      History:")
                    for h in history:
                        print(f"        Round {h.get('round')}: {len(h.get('alphas', []))} alphas saved")

                # 显示最佳表达式
                best = alpha_result.get("best_expression", {})
                if best:
                    print(f"      Best expression: {best.get('code', 'N/A')[:80]}...")
                    print(f"      Best Sharpe: {best.get('metrics', {}).get('sharpe', 0):.4f}")

        # 显示失败的 alpha
        if failed_alphas:
            print(f"\n[Failed Alphas]")
            for idx, failed in enumerate(failed_alphas, 1):
                print(f"  [{idx}] Alpha ID: {failed.get('alpha_id')}")
                print(f"      Status: {failed.get('status')}")
                print(f"      Reason: {failed.get('reason', 'Unknown')}")

        # 保存结果
        output_file = os.path.join(output_dir, "improvement_test_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[Results Saved]")
        print(f"  Output file: {output_file}")

    except Exception as e:
        print(f"\n[Error] Improvement failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
