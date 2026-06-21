"""
Test for ExpressionGeneratorAgent
验证表达式生成功能，使用真实的 ideas 和 field_mapping 数据
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory
from agents.expr_generator import ExpressionGeneratorAgent


def load_json_file(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("ExpressionGeneratorAgent Test")
    print("=" * 70)

    # ==================== 文件路径 ====================
    ideas_file = os.path.join(project_root, 'output', 'ideas_ASI_MINVOL1M_fundamental17_MATRIX.json')
    classifier_file = os.path.join(project_root, 'output', 'data_classifier_output.json')

    # ==================== 加载数据 ====================
    print("\n[1] Loading input data...")

    # Load ideas
    ideas_data = load_json_file(ideas_file)
    ideas = ideas_data.get("ideas", [])
    file_info = ideas_data.get("file_info", {})
    print(f"    Loaded {len(ideas)} ideas from {os.path.basename(ideas_file)}")

    # Load field_mapping from classifier output
    classifier_data = load_json_file(classifier_file)
    field_mapping = classifier_data.get("field_mapping", {})
    print(f"    Loaded {len(field_mapping)} field categories from {os.path.basename(classifier_file)}")

    # ==================== 创建 Agent ====================
    print("\n[2] Creating ExpressionGeneratorAgent...")

    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    config = ConfigLoader.load(config_path)
    config["config_dir"] = os.path.join(project_root, 'config')

    llm_service = LLMServiceFactory.create_from_config(config)

    agent = ExpressionGeneratorAgent(
        config=config,
        llm_service=llm_service
    )
    print(f"    Agent created (LLM enabled: {agent.use_llm})")

    # ==================== 调用 Agent ====================
    print("\n[3] Calling agent.run()...")
    print("=" * 70)

    input_data = {
        "ideas": ideas,
        "field_mapping": field_mapping,
        "file_info": file_info
    }

    try:
        result = agent.run(input_data)
        idea_results = result.get('ideas', [])

        print(f"\n[OK] Agent completed")
        print(f"    Status: {result.get('status')}")
        print(f"    Generated: {result.get('count')} expressions for {len(idea_results)} ideas")

        # 只显示每个 idea 生成的表达式数量，不打印具体表达式
        print(f"\nGenerated Expressions Summary:")
        for idea_result in idea_results:
            idea_id = idea_result.get('idea_id')
            expr_count = idea_result.get('count', 0)
            print(f"  Idea: {idea_id} -> {expr_count} expressions")

    except Exception as e:
        print(f"\n[ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ==================== 保存结果 ====================
    print("\n[4] Saving results...")
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Build output filename from file_info
    region = file_info.get('region', 'UNKNOWN')
    universe = file_info.get('universe', 'UNKNOWN')
    dataset = file_info.get('dataset', 'UNKNOWN')
    data_type = file_info.get('type', 'UNKNOWN')
    output_filename = f"expressions_{region}_{universe}_{dataset}_{data_type}.json"

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
