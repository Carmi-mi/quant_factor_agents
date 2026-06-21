"""
Simple test for DataClassifierAgent
Shows LLM output and Agent final output for agent communication testing
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory
from agents.data_classifier import DataClassifierAgent


def main():
    print("=" * 70)
    print("DataClassifierAgent Test")
    print("=" * 70)

    # Load config
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    config = ConfigLoader.load(config_path)
    print("[OK] Config loaded")

    # Create agent
    llm_service = LLMServiceFactory.create_from_config(config)
    agent = DataClassifierAgent(config, llm_service)
    print("[OK] Agent created")

    # Run classification
    print("\n" + "-" * 70)
    print("Starting classification...")
    print("-" * 70)

    try:
        result = agent.run(config)

        # Print agent output using agent's built-in method
        print("\n" + "=" * 70)
        print("AGENT FINAL OUTPUT")
        print("=" * 70)
        agent.print_classification(result)

        # Show example: how to use output for next agent
        print("\n" + "=" * 70)
        print("EXAMPLE: IdeaGeneratorAgent Input")
        print("=" * 70)
        idea_generator_input = {
            "class_list": result['class_list'],
            "region": result['file_info'].get('region', 'USA'),
            "universe": result['file_info'].get('universe', 'TOP3000'),
            "dataset": result['file_info'].get('dataset', 'unknown'),
            "type": result['file_info'].get('type', 'MATRIX'),
            "generate_num": 4
        }
        print(json.dumps(idea_generator_input, indent=2))

        # Save outputs
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Save agent output (using result directly)
        output_file = os.path.join(output_dir, 'data_classifier_output.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Output saved to: {output_file}")

        print("\n" + "=" * 70)
        print("[OK] Test passed!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
