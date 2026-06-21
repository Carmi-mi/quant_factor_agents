"""
Idea Generator Agent
Generates factor ideas based on data classes and operator descriptions using LLM
"""
import os
import json
import re
import numpy as np
from typing import Dict, Any, List
from difflib import SequenceMatcher
from core.agent_base import Agent
from services.rag_dedup_service import RAGDedupService
from services.llm_service import LLMService
from infrastructure.prompt_templates import PromptTemplates


class IdeaGeneratorAgent(Agent):
    """Idea Generator Agent - generates creative factor ideas using LLM"""

    def __init__(self, config: Dict[str, Any],
                 rag_service: RAGDedupService = None,
                 llm_service: LLMService = None):
        """
        Initialize

        Args:
            config: Configuration
            rag_service: RAG deduplication service for semantic similarity
            llm_service: LLM service (required)
        """
        super().__init__("IdeaGenerator", config)
        self.rag = rag_service
        self.llm = llm_service
        self.generate_num = config.get("idea_generation", {}).get("generate_num", 4)
        self.cache_dir = config.get("data", {}).get("cache_dir", "cache")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load operator descriptions mapping (code -> desc)
        self.operators_desc = self._load_operators_desc(config)
        # Build reverse mapping (desc -> code)
        self.desc_to_code = {v: k for k, v in self.operators_desc.items()}
        # Get all operator descriptions for LLM
        self.all_op_descriptions = list(self.operators_desc.values())

        if self.llm is None:
            raise ValueError("LLM service is required for IdeaGeneratorAgent")

    def _get_history_filepath(self, context_key: str) -> str:
        """Get file path for idea history"""
        safe_key = context_key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, 'ideas', f"history_{safe_key}.json")

    def _get_embeddings_filepath(self, context_key: str) -> str:
        """Get file path for embeddings"""
        safe_key = context_key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, 'ideas', f"embeddings_{safe_key}.npy")

    def _load_history(self, context_key: str) -> List[str]:
        """Load idea history for a context"""
        filepath = self._get_history_filepath(context_key)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("contents", [])
            except Exception as e:
                print(f"[IdeaGenerator] Failed to load history: {e}")
        return []

    def _load_embeddings(self, context_key: str):
        """Load embeddings for a context"""
        filepath = self._get_embeddings_filepath(context_key)
        if os.path.exists(filepath):
            try:
                return np.load(filepath)
            except Exception as e:
                print(f"[IdeaGenerator] Failed to load embeddings: {e}")
        return None

    def _save_history(self, context_key: str, contents: List[str]):
        """Save idea history for a context"""
        filepath = self._get_history_filepath(context_key)
        data = {
            "context_key": context_key,
            "contents": contents,
            "count": len(contents)
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[IdeaGenerator] Failed to save history: {e}")

    def _save_embeddings(self, context_key: str, embeddings):
        """Save embeddings for a context"""
        filepath = self._get_embeddings_filepath(context_key)
        try:
            np.save(filepath, embeddings)
        except Exception as e:
            print(f"[IdeaGenerator] Failed to save embeddings: {e}")

    def _load_operators_desc(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Load operator descriptions from config file"""
        # 尝试从配置文件中获取config目录路径
        config_dir = config.get("config_dir")
        if not config_dir:
            # 如果没有设置，使用相对于当前文件的路径
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        
        op_file = os.path.join(config_dir, "operators_desc.json")

        # 从文件加载
        if os.path.exists(op_file):
            try:
                with open(op_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[IdeaGenerator] Failed to load operators_desc.json: {e}")
                raise  # 文件存在但读取失败时抛出异常
        
        # 文件不存在时抛出异常，不返回默认值
        raise FileNotFoundError(f"operators_desc.json not found at: {op_file}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ideas

        Args:
            input_data: Contains class_list, file_info
                - file_info: dict with region, universe, dataset, type
                - op_list is NOT needed, loaded from operators_desc.json automatically

        Returns:
            List of ideas with operator codes
        """
        # Validate input
        self.validate_input(input_data, ["class_list", "file_info"])

        class_list = input_data["class_list"]
        file_info = input_data["file_info"]

        region = file_info["region"]
        universe = file_info["universe"]
        dataset = file_info["dataset"]
        data_type = file_info["type"]
        generate_num = input_data.get("generate_num", self.generate_num)

        print(f"[IdeaGenerator] Generating ideas for {region}/{universe}/{dataset}/{data_type}")
        print(f"[IdeaGenerator] Classes: {len(class_list)}, Operators: {len(self.all_op_descriptions)}")

        # Build context key for deduplication
        context_key = f"{region}_{universe}_{dataset}_{data_type}"

        # Load existing history and embeddings for this context (outside loop)
        existing_history = self._load_history(context_key)
        existing_embeddings = self._load_embeddings(context_key) if self.rag else None

        # Generate ideas with retry mechanism until at least one valid idea is generated
        retry_count = 0
        ideas = []
        new_embeddings = None

        while True:
            retry_count += 1
            if retry_count > 1:
                print(f"[IdeaGenerator] Retry {retry_count}: Regenerating ideas...")

            # Generate ideas using LLM with all operator descriptions
            ideas = self._generate_with_llm(
                class_list=class_list,
                num=generate_num,
                history=existing_history
            )

            if not ideas:
                print(f"[IdeaGenerator] Warning: LLM returned no ideas")
                continue

            # Validate and convert ideas (remove hallucinations)
            raw_idea_count = len(ideas)
            ideas = self._validate_and_convert_ideas(ideas, class_list)
            valid_idea_count = len(ideas)

            # Check if too many ideas were filtered
            if valid_idea_count < raw_idea_count:
                print(f"[IdeaGenerator] Warning: {raw_idea_count - valid_idea_count} ideas filtered due to invalid classes/operators")

            # Deduplicate using RAG semantic similarity (inside loop)
            if self.rag:
                ideas, new_embeddings = self.rag.filter_duplicates(
                    ideas,
                    existing_embeddings,
                    threshold=self.rag.similarity_threshold
                )

            final_count = len(ideas)

            # If we have ideas, update history/embeddings and break the loop
            if final_count > 0:
                # Update history (in memory, not saved yet)
                new_contents = [idea.get("content", "").strip() for idea in ideas]
                existing_history.extend(new_contents)

                # Update embeddings (in memory, not saved yet)
                if self.rag and new_embeddings is not None and new_embeddings.size > 0:
                    if existing_embeddings is not None and existing_embeddings.size > 0:
                        existing_embeddings = np.vstack([existing_embeddings, new_embeddings])
                    else:
                        existing_embeddings = new_embeddings
                break

            print(f"[IdeaGenerator] Warning: No valid ideas after filtering, will retry immediately")



        # Save final history and embeddings to files (after loop ends)
        if ideas:
            self._save_history(context_key, existing_history)

            if self.rag and existing_embeddings is not None and existing_embeddings.size > 0:
                self._save_embeddings(context_key, existing_embeddings)
        
        # Check if generated enough ideas
        final_count = len(ideas)
        if final_count < generate_num:
            print(f"[IdeaGenerator] Warning: Only generated {final_count}/{generate_num} ideas")

        print(f"[IdeaGenerator] Generated {final_count} unique ideas")

        return {
            "status": "success",
            "ideas": ideas,
            "count": final_count,
            "expected_count": generate_num,
            "context": context_key,
            "file_info": file_info,
            "retry_count": retry_count
        }

    def _validate_and_convert_ideas(self, ideas: List[Dict[str, Any]],
                                     valid_classes: List[str]) -> List[Dict[str, Any]]:
        """
        Validate ideas and convert operator descriptions to codes.
        Removes hallucinated use_class and use_op values.
        Attempts to correct class names with similarity >= 0.95.
        """
        valid_class_set = set(valid_classes)
        validated_ideas = []

        for idea in ideas:
            original_classes = idea.get("use_class", [])
            original_ops = idea.get("use_op", [])

            # Filter and correct use_class
            valid_classes_in_idea = []
            removed_classes = []
            for c in original_classes:
                if c in valid_class_set:
                    valid_classes_in_idea.append(c)
                else:
                    # Try to find similar class with similarity >= 0.95
                    corrected = self._find_similar_class(c, valid_classes)
                    if corrected:
                        print(f"[IdeaGenerator] Corrected class '{c}' -> '{corrected}'")
                        valid_classes_in_idea.append(corrected)
                    else:
                        removed_classes.append(c)

            # Filter use_op - keep only valid operator descriptions and convert to codes
            valid_op_codes = []
            removed_ops = []
            for desc in original_ops:
                if desc in self.desc_to_code:
                    valid_op_codes.append(self.desc_to_code[desc])
                else:
                    removed_ops.append(desc)
                    print(f"[IdeaGenerator] Warning: Unknown operator description '{desc}'")

            # Log removed items
            if removed_classes:
                print(f"[IdeaGenerator] Filtered invalid classes from {idea.get('id')}: {removed_classes}")
            if removed_ops:
                print(f"[IdeaGenerator] Filtered invalid operators from {idea.get('id')}: {removed_ops}")

            # Only keep idea if it still has valid classes and operators
            if valid_classes_in_idea and valid_op_codes:
                idea["use_class"] = valid_classes_in_idea
                idea["use_op"] = valid_op_codes
                validated_ideas.append(idea)
            else:
                print(f"[IdeaGenerator] Discarded {idea.get('id')}: no valid classes or operators remaining")

        return validated_ideas

    def _find_similar_class(self, invalid_class: str, valid_classes: List[str], threshold: float = 0.95) -> str:
        """
        Find the most similar valid class name using edit distance.
        Returns the match if similarity >= threshold, otherwise None.
        """
        best_match = None
        best_ratio = 0.0

        for valid_class in valid_classes:
            ratio = SequenceMatcher(None, invalid_class, valid_class).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = valid_class

        if best_match and best_ratio >= threshold:
            return best_match
        return None

    def _generate_with_llm(self, class_list: List[str],
                           num: int,
                           history: List[str]) -> List[Dict[str, Any]]:
        """Generate ideas using LLM"""
        # Use PromptTemplates to generate prompt with all operator descriptions
        prompt = PromptTemplates.idea_generation(
            class_list=class_list,
            op_descriptions=self.all_op_descriptions,
            num_ideas=num,
            history=history
        )

        try:
            response = self.llm.client.complete(prompt)
            return self._parse_ideas_response(response, num)
        except Exception as e:
            print(f"[IdeaGenerator] LLM generation failed: {e}")
            return []

    def _parse_ideas_response(self, response: str, expected_num: int) -> List[Dict[str, Any]]:
        """Parse ideas from LLM response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"[IdeaGenerator] Extracted JSON (first 300 chars): {json_str[:300]}")
                data = json.loads(json_str)
                ideas = data.get("ideas", [])
                print(f"[IdeaGenerator] Found {len(ideas)} ideas in response")

                # Check if LLM returned enough ideas
                if len(ideas) < expected_num:
                    print(f"[IdeaGenerator] Warning: LLM returned only {len(ideas)}/{expected_num} ideas")

                # Format and validate ideas
                formatted_ideas = []
                skipped_count = 0
                for i, idea in enumerate(ideas):
                    formatted_idea = {
                        "id": f"idea_{i}",
                        "content": idea.get("content", "").strip(),
                        "use_class": idea.get("use_class", []),
                        "use_op": idea.get("use_op", [])  # Still descriptions at this point
                    }

                    # Check for empty fields
                    skip_reason = None
                    if not formatted_idea["content"]:
                        skip_reason = "empty content"
                    elif not formatted_idea["use_class"]:
                        skip_reason = "empty use_class"
                    elif not formatted_idea["use_op"]:
                        skip_reason = "empty use_op"

                    if skip_reason:
                        skipped_count += 1
                    else:
                        formatted_ideas.append(formatted_idea)

                # Print summary instead of individual ideas
                if skipped_count > 0:
                    print(f"[IdeaGenerator] Skipped {skipped_count} ideas with empty fields")
                print(f"[IdeaGenerator] Valid ideas: {len(formatted_ideas)}/{len(ideas)}")

                return formatted_ideas
            else:
                print(f"[IdeaGenerator] No JSON found in response")
        except Exception as e:
            print(f"[IdeaGenerator] Failed to parse ideas: {e}")

        return []
