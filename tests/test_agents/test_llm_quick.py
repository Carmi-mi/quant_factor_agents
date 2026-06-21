#!/usr/bin/env python3
"""
Quick test to verify LLM is working
Usage: python test_llm_quick.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from infrastructure.config_loader import ConfigLoader
from services.llm_service import LLMServiceFactory


def main():
    print("Testing LLM connection...")
    
    # Get absolute path to config (project root)
    config_path = os.path.join(project_root, "config", "settings.yaml")
    
    try:
        # Load config with absolute path
        config = ConfigLoader.load(config_path)
        
        if not config.get("llm"):
            print("[ERROR] No LLM configuration found!")
            return 1
        
        provider = config["llm"].get("provider", "unknown")
        model = config["llm"].get("model", "unknown")
        print(f"Provider: {provider}")
        print(f"Model: {model}")
        
        # Initialize LLM
        llm_service = LLMServiceFactory.create_from_config(config)
        print("[OK] LLM service initialized")
        
        # Test completion
        print("\nSending test prompt...")
        response = llm_service.client.complete(
            "What is 2+2? Answer with just the number."
        )
        
        print(f"[OK] Response: {response.strip()}")
        print("\n[OK] LLM is working correctly!")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
