"""
阶段性测试脚本 - Pipeline Stage Test
合并三个Agent的循环测试：IdeaGeneratorAgent -> ExpressionGeneratorAgent -> BacktestCoordinator
（DataClassifierAgent 只在第一次运行时执行）

执行流程：
    Phase 1: DataClassifierAgent - 对数据字段进行分类（仅首次执行）
    Phase 2: IdeaGeneratorAgent - 基于分类结果生成想法（循环）
    Phase 3: ExpressionGeneratorAgent - 基于想法生成表达式（循环）
    Phase 4: BacktestCoordinator - 并行回测表达式（循环）

输出文件命名规范：
    - data_classifier_output.json                    (固定名称)
    - ideas_{region}_{universe}_{dataset}_{type}.json
    - expressions_{region}_{universe}_{dataset}_{type}.json
    - backtest_{region}_{universe}_{dataset}_{type}.json
    例如: ideas_ASI_MINVOL1M_other455_MATRIX.json
         expressions_ASI_MINVOL1M_other455_MATRIX.json
         backtest_ASI_MINVOL1M_other455_MATRIX.json

使用方法：
    python tests/test_agents/test_pipeline_stage.py

可选参数（通过修改CONFIG配置）：
    - skip_classifier: 跳过Phase 1（从已有文件加载）
    - skip_idea_generator: 跳过Phase 2（从已有文件加载）
    - skip_expression_generator: 跳过Phase 3（从已有文件加载）
    - skip_backtest: 跳过Phase 4（从已有文件加载）
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory
from services.rag_dedup_service import RAGDedupService
from services.backtest_coordinator import BacktestCoordinator
from agents.data_classifier import DataClassifierAgent
from agents.idea_generator import IdeaGeneratorAgent
from agents.expr_generator import ExpressionGeneratorAgent


# ==================== 配置参数 ====================
CONFIG = {
    "skip_classifier": False,       # 如果已有分类结果，可设为True跳过
    "skip_idea_generator": False,   # 如果已有ideas，可设为True跳过
    "skip_expression_generator": False,  # 如果已有expressions，可设为True跳过
    "skip_backtest": False,         # 如果已有回测结果，可设为True跳过
    "max_iterations": 0,            # 最大迭代次数，0表示无限循环
    "loop_delay_seconds": 0,        # 迭代间隔秒数
}


def print_phase_header(phase_num, title):
    """打印阶段标题"""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {title}")
    print("=" * 80)


def print_section(title):
    """打印小节标题"""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def load_json_file(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json_file(filepath, data):
    """Save JSON file"""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    # 转换数据为可序列化格式
    serializable_data = convert_to_serializable(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def get_context_from_file_info(file_info):
    """从file_info生成context字符串"""
    region = file_info.get('region', 'UNKNOWN')
    universe = file_info.get('universe', 'UNKNOWN')
    dataset = file_info.get('dataset', 'UNKNOWN')
    data_type = file_info.get('type', 'UNKNOWN')
    return f"{region}_{universe}_{dataset}_{data_type}"


# =============================================================================
# PHASE 1: DataClassifierAgent
# =============================================================================
def run_phase_1_classifier(config, llm_service):
    """
    Phase 1: 运行 DataClassifierAgent
    输入: 配置文件
    输出: field_mapping, class_list, file_info
    """
    print_phase_header(1, "DataClassifierAgent - 数据字段分类")

    # 检查是否跳过
    if CONFIG["skip_classifier"]:
        print("[SKIP] Phase 1 skipped (skip_classifier=True)")
        output_path = os.path.join(project_root, 'output', 'data_classifier_output.json')
        if os.path.exists(output_path):
            data = load_json_file(output_path)
            print(f"[OK] Loaded existing classification from: {output_path}")
            return data
        else:
            print("[ERROR] No existing classification file found!")
            return None

    # 创建并运行agent
    agent = DataClassifierAgent(config, llm_service)
    result = agent.run(config)

    # 保存结果 - 命名规范: data_classifier_output.json
    output_path = os.path.join(project_root, 'output', 'data_classifier_output.json')
    save_json_file(output_path, result)
    print(f"\n[OK] Saved to: {output_path}")

    return result


# =============================================================================
# PHASE 2: IdeaGeneratorAgent
# =============================================================================
def run_phase_2_idea_generator(config, llm_service, classifier_result):
    """
    Phase 2: 运行 IdeaGeneratorAgent
    输入: classifier_result (包含 class_list, field_mapping, file_info)
    输出: ideas list
    """
    print_phase_header(2, "IdeaGeneratorAgent - 生成想法")

    file_info = classifier_result.get('file_info', {})
    context = get_context_from_file_info(file_info)

    # 检查是否跳过
    if CONFIG["skip_idea_generator"]:
        print("[SKIP] Phase 2 skipped (skip_idea_generator=True)")
        ideas_file = os.path.join(project_root, 'output', f'ideas_{context}.json')
        if os.path.exists(ideas_file):
            data = load_json_file(ideas_file)
            print(f"[OK] Loaded existing ideas from: {ideas_file}")
            return data
        else:
            print("[ERROR] No existing ideas file found!")
            return None

    # 准备输入数据
    input_data = {
        "class_list": classifier_result.get('class_list', []),
        "file_info": file_info
    }

    print_section("Input Data")
    print(f"    Classes: {len(input_data['class_list'])}")
    print(f"    Context: {context}")

    # 创建RAG服务（配置从config读取）
    rag_service = RAGDedupService(config=config.get("rag_dedup", {}))

    # 创建并运行agent
    agent = IdeaGeneratorAgent(
        config=config,
        rag_service=rag_service,
        llm_service=llm_service
    )
    result = agent.run(input_data)

    # 保存结果 - 命名规范: ideas_{region}_{universe}_{dataset}_{type}.json
    output_path = os.path.join(project_root, 'output', f'ideas_{context}.json')
    save_json_file(output_path, result)
    print(f"\n[OK] Saved to: {output_path}")

    return result


# =============================================================================
# PHASE 3: ExpressionGeneratorAgent
# =============================================================================
def run_phase_3_expr_generator(config, llm_service, classifier_result, ideas_result):
    """
    Phase 3: 运行 ExpressionGeneratorAgent
    输入: ideas_result (包含 ideas list), classifier_result (包含 field_mapping)
    输出: expressions list
    """
    print_phase_header(3, "ExpressionGeneratorAgent - 生成表达式")

    file_info = classifier_result.get('file_info', {})
    context = get_context_from_file_info(file_info)

    # 检查是否跳过
    if CONFIG["skip_expression_generator"]:
        print("[SKIP] Phase 3 skipped (skip_expression_generator=True)")
        expressions_file = os.path.join(project_root, 'output', f'expressions_{context}.json')
        if os.path.exists(expressions_file):
            data = load_json_file(expressions_file)
            print(f"[OK] Loaded existing expressions from: {expressions_file}")
            return data
        else:
            print("[ERROR] No existing expressions file found!")
            return None

    # 准备输入数据
    input_data = {
        "ideas": ideas_result.get('ideas', []),
        "field_mapping": classifier_result.get('field_mapping', {}),
        "file_info": file_info
    }

    print_section("Input Data")
    print(f"    Ideas: {len(input_data['ideas'])}")
    print(f"    Field Categories: {len(input_data['field_mapping'])}")

    # 创建并运行agent
    agent = ExpressionGeneratorAgent(
        config=config,
        llm_service=llm_service
    )
    result = agent.run(input_data)

    # 保存结果 - 命名规范: expressions_{region}_{universe}_{dataset}_{type}.json
    output_path = os.path.join(project_root, 'output', f'expressions_{context}.json')
    save_json_file(output_path, result)
    print(f"\n[OK] Saved to: {output_path}")

    return result


# =============================================================================
# PHASE 4: BacktestCoordinator
# =============================================================================
def backtest_callback(result: dict):
    """回测回调函数 - 打印每个idea的回测结果"""
    idea_id = result.get("idea_id", "unknown")
    print(f"\n[Callback] 收到 idea '{idea_id}' 的回测结果")
    print(f"[Callback] 表达式数量: {result.get('expressions_count', 0)}")
    print("-" * 60)


def run_phase_4_backtest(config, llm_service, classifier_result, expressions_result):
    """
    Phase 4: 运行 BacktestCoordinator
    输入: expressions_result (包含 ideas list), classifier_result (包含 file_info)
    输出: backtest results
    """
    print_phase_header(4, "BacktestCoordinator - 并行回测")

    file_info = classifier_result.get('file_info', {})
    context = get_context_from_file_info(file_info)

    # 检查是否跳过
    if CONFIG["skip_backtest"]:
        print("[SKIP] Phase 4 skipped (skip_backtest=True)")
        backtest_file = os.path.join(project_root, 'output', f'backtest_{context}.json')
        if os.path.exists(backtest_file):
            data = load_json_file(backtest_file)
            print(f"[OK] Loaded existing backtest results from: {backtest_file}")
            return data
        else:
            print("[ERROR] No existing backtest file found!")
            return None

    # 准备输入数据
    ideas = expressions_result.get('ideas', [])
    
    print_section("Input Data")
    print(f"    Ideas: {len(ideas)}")
    for idea in ideas:
        print(f"    - {idea.get('idea_id')}: {len(idea.get('expressions', []))} 个表达式")

    # 创建 BacktestCoordinator
    max_workers = config.get("backtest", {}).get("max_workers", 2)
    coordinator = BacktestCoordinator(config, llm_service=llm_service, max_workers=max_workers)
    coordinator.set_downstream_callback(backtest_callback)
    print(f"\n[OK] BacktestCoordinator 创建完成，max_workers={max_workers}")

    # 运行并行回测
    print_section("开始并行回测")
    try:
        result = coordinator.run_parallel_backtest(
            ideas=ideas,
            file_info=file_info
        )
        
        # 保存结果 - 命名规范: backtest_{region}_{universe}_{dataset}_{type}.json
        output_path = os.path.join(project_root, 'output', f'backtest_{context}.json')
        save_json_file(output_path, result)
        print(f"\n[OK] Saved to: {output_path}")
        
        # 提取 alpha_id 并追加到 need_improve.txt
        need_improve_path = os.path.join(project_root, 'output', 'need_improve.txt')
        final_alphas = result.get('final_alphas', [])
        with open(need_improve_path, 'a', encoding='utf-8') as f:
            for idea in final_alphas:
                for alpha in idea.get('alphas', []):
                    alpha_id = alpha.get('alpha_id', '')
                    if alpha_id:
                        f.write(f"{alpha_id}\n")
        print(f"[OK] Alpha IDs appended to: {need_improve_path}")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] 回测过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN
# =============================================================================
def run_three_agent_loop(config, llm_service, classifier_result, iteration):
    """运行三个Agent的循环（IdeaGeneratorAgent -> ExpressionGeneratorAgent -> BacktestCoordinator）"""
    print("\n" + "=" * 80)
    print(f"THREE AGENT LOOP ITERATION #{iteration}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Skip Idea Generator: {CONFIG['skip_idea_generator']}")
    print(f"  Skip Expression Generator: {CONFIG['skip_expression_generator']}")
    print(f"  Skip Backtest: {CONFIG['skip_backtest']}")

    # 执行三个阶段（循环部分）
    results = {}

    # Phase 2: IdeaGeneratorAgent
    ideas_result = run_phase_2_idea_generator(config, llm_service, classifier_result)
    if ideas_result is None:
        print("\n[FAIL] Pipeline stopped at Phase 2")
        return None
    results['ideas'] = ideas_result

    # Phase 3: ExpressionGeneratorAgent
    expressions_result = run_phase_3_expr_generator(config, llm_service, classifier_result, ideas_result)
    if expressions_result is None:
        print("\n[FAIL] Pipeline stopped at Phase 3")
        return None
    results['expressions'] = expressions_result

    # Phase 4: BacktestCoordinator
    backtest_result = run_phase_4_backtest(config, llm_service, classifier_result, expressions_result)
    if backtest_result is None:
        print("\n[FAIL] Pipeline stopped at Phase 4")
        return None
    results['backtest'] = backtest_result

    # 迭代总结
    print("\n" + "=" * 80)
    print(f"ITERATION #{iteration} COMPLETED SUCCESSFULLY")
    print("=" * 80)

    context = get_context_from_file_info(classifier_result.get('file_info', {}))
    summary = backtest_result.get("summary", {})

    print("\nSummary:")
    print(f"  Phase 2 (Ideas): {ideas_result.get('count', 0)} ideas generated")
    print(f"  Phase 3 (Expressions): {expressions_result.get('count', 0)} expressions generated")
    print(f"  Phase 4 (Backtest): {summary.get('total_ideas', 0)} ideas backtested")
    print(f"         Total Expressions: {summary.get('total_expressions', 0)}")
    print(f"         Duration: {summary.get('duration_seconds', 0):.2f} seconds")
    print(f"         Parallel Workers: {summary.get('parallel_workers', 0)}")

    print("\nOutput Files (命名规范):")
    print(f"  - output/data_classifier_output.json")
    print(f"  - output/ideas_{context}.json")
    print(f"  - output/expressions_{context}.json")
    print(f"  - output/backtest_{context}.json")

    return results


def main():
    print("\n" + "=" * 80)
    print("PIPELINE STAGE TEST - THREE AGENT LOOP MODE")
    print("量化因子挖掘 Agents Team - 三Agent循环集成测试")
    print("IdeaGeneratorAgent -> ExpressionGeneratorAgent -> BacktestCoordinator")
    print("=" * 80)

    # 加载配置
    print("\n" + "-" * 70)
    print("Loading Configuration...")
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    config = ConfigLoader.load(config_path)
    print("[OK] Config loaded")

    # 创建LLM服务（共享）
    print("-" * 70)
    print("Creating LLM Service...")
    llm_service = LLMServiceFactory.create_from_config(config)
    print("[OK] LLM Service created")

    # Phase 1: DataClassifierAgent（仅执行一次）
    print("\n" + "-" * 70)
    print("Running Phase 1: DataClassifierAgent (One-time execution)")
    print("-" * 70)
    classifier_result = run_phase_1_classifier(config, llm_service)
    if classifier_result is None:
        print("\n[FAIL] Pipeline stopped at Phase 1")
        return 1

    # 循环执行三个Agent（Phase 2-4）
    iteration = 0
    max_iterations = CONFIG.get('max_iterations', 0)  # 0 表示无限循环
    
    while True:
        iteration += 1
        
        # 运行三个Agent的循环
        results = run_three_agent_loop(config, llm_service, classifier_result, iteration)
        
        if results is None:
            print(f"\n[ERROR] Iteration #{iteration} failed, stopping loop...")
            break
        
        # 检查是否达到最大迭代次数
        if max_iterations > 0 and iteration >= max_iterations:
            print(f"\n[INFO] Reached max iterations ({max_iterations}), stopping loop...")
            break
        
        # 迭代间隔（可选）
        delay = CONFIG.get('loop_delay_seconds', 0)
        if delay > 0:
            print(f"\n[INFO] Waiting {delay} seconds before next iteration...")
            import time
            time.sleep(delay)
    
    # 最终总结
    print("\n" + "=" * 80)
    print("ALL ITERATIONS COMPLETED!")
    print("=" * 80)
    print(f"\nTotal iterations: {iteration}")
    print(f"Results saved to output/ directory")

    return 0


if __name__ == "__main__":
    sys.exit(main())
