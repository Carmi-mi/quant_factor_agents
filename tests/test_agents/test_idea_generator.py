"""
Test for IdeaGeneratorAgent with RAG deduplication
验证 idea 生成功能及其 RAG 语义去重效果（单次调用版本）
"""
import sys
import os
import json
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory
from services.rag_dedup_service import RAGDedupService
from agents.idea_generator import IdeaGeneratorAgent


def parse_classification_filename(filename):
    """Parse classification filename to extract region, universe, dataset, type"""
    match = re.match(r'classification_(\w+)_(\w+)_(\w+)_(\w+)\.json', filename)
    if match:
        return {
            "region": match.group(1),
            "universe": match.group(2),
            "dataset": match.group(3),
            "type": match.group(4)
        }
    return None


def load_classification_data(filepath):
    """Load classification data and extract class_list"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    field_mapping = data.get("field_mapping", {})
    class_list = list(field_mapping.keys())
    return class_list, field_mapping


def main():
    print("=" * 70)
    print("IdeaGeneratorAgent Test - Single LLM Call")
    print("=" * 70)

    # ==================== 配置参数 ====================
    classification_file = "classification_ASI_MINVOL1M_other455_MATRIX.json"
    generate_num = 4
    similarity_threshold = 0.85

    # ==================== 加载数据 ====================
    classification_path = os.path.join(project_root, 'cache', 'classification', classification_file)
    cache_dir = os.path.join(project_root, 'cache', 'ideas')
    os.makedirs(cache_dir, exist_ok=True)

    file_info = parse_classification_filename(classification_file)
    if not file_info:
        print(f"[ERROR] Failed to parse filename: {classification_file}")
        return 1

    class_list, _ = load_classification_data(classification_path)
    print(f"[OK] Loaded {len(class_list)} classes from {classification_file}")

    # ==================== 创建 Agent ====================
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    config = ConfigLoader.load(config_path)
    config["config_dir"] = os.path.join(project_root, 'config')
    config["cache_dir"] = cache_dir

    llm_service = LLMServiceFactory.create_from_config(config)
    rag_service = RAGDedupService(config={
        "similarity_threshold": similarity_threshold
        # 使用默认模型 all-MiniLM-L6-v2，自动检测本地模型
    })

    agent = IdeaGeneratorAgent(
        config=config,
        rag_service=rag_service,
        llm_service=llm_service
    )
    print("[OK] Agent created with RAG dedup")

    # ==================== 调用 Agent ====================
    print("\n" + "=" * 70)
    print("CALLING AGENT")
    print("=" * 70)

    input_data = {
        "class_list": class_list,
        "file_info": file_info,
        "generate_num": generate_num
    }

    try:
        result = agent.run(input_data)
        ideas = result.get('ideas', [])

        print(f"\n[OK] Agent completed")
        print(f"    Status: {result.get('status')}")
        print(f"    Generated: {result.get('count')} ideas")
        print(f"    Retry count: {result.get('retry_count', 0)}")

        print(f"\nGenerated Ideas:")
        for i, idea in enumerate(ideas, 1):
            content = idea.get('content', '')
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"\n  [{i}] {idea.get('id')}")
            print(f"      Content: {preview}")
            print(f"      Classes: {idea.get('use_class', [])}")
            print(f"      Operators: {idea.get('use_op', [])}")

    except Exception as e:
        print(f"\n[ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ==================== 保存结果 ====================
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"ideas_{result.get('context')}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
