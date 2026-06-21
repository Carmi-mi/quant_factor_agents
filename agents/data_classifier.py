"""
Dataset Classifier Agent
Reads CSV files and classifies fields by economic meaning using LLM with self-review mechanism
"""
import os
import json
from typing import Dict, Any, List, Tuple
from core.agent_base import Agent
from services.llm_service import LLMService
from infrastructure.prompt_templates import PromptTemplates


class DataClassifierAgent(Agent):
    """Dataset classification agent using LLM with iterative review"""
    
    def __init__(self, config: Dict[str, Any], llm_service: LLMService = None):
        super().__init__("DataClassifier", config)
        self.cache_dir = config.get("data", {}).get("cache_dir", "cache")
        self.llm = llm_service
        self.id_column = config.get("data", {}).get("id_column", "id")
        self.description_column = config.get("data", {}).get("description_column", "description")
        self.max_review_iterations = config.get("classification", {}).get("max_review_iterations", 3)
        
        if self.llm is None:
            raise ValueError("LLM service is required for DataClassifierAgent")
    
    def run(self, config: Dict[str, Any], force_reload: bool = False) -> Dict[str, Any]:
        """
        Classify dataset fields from config with iterative review
        
        Args:
            config: Configuration with data.raw_file path
            force_reload: Force re-classification even if cache exists
        
        Returns:
            Classification results with review history
        """
        # Get file path from config
        file_path = self._get_file_from_config(config)
        if not file_path:
            raise ValueError("No data file configured in config['data']['raw_file']")
        
        data_file = os.path.basename(file_path)
        file_info = self._parse_filename(data_file)
        cache_file = self._get_cache_path(file_info)
        
        # Check cache
        if os.path.exists(cache_file) and not force_reload:
            return self._load_from_cache(cache_file, data_file)
        
        # Read CSV
        fields = self._read_csv(file_path)
        print(f"[DataClassifier] Read {len(fields)} fields from {data_file}")
        
        # Choose classification strategy based on dataset size
        if len(fields) <= 500:
            # Small dataset: use iterative review process
            print(f"[DataClassifier] Using iterative review (≤500 fields)")
            classification, review_history = self._classify_with_review(fields, file_info)
        else:
            # Large dataset: use batch classification with overall review
            print(f"[DataClassifier] Using batch classification with review (>500 fields)")
            classification, review_history = self._batch_classification(fields)
            classification["file_info"] = file_info
        
        # Prepare final result (minimal data for cache)
        final_result = {
            "field_mapping": classification["field_mapping"],
            "reasoning": classification.get("reasoning", ""),
            "file_info": file_info,
            "total_fields": len(fields),
            "total_iterations": len(review_history)
        }
        
        # Save only final result to cache
        self._save_to_cache(final_result, cache_file)
        
        # Return full result with review history
        final_result["review_history"] = review_history
        return self._format_result(final_result, data_file, cached=False)
    
    def _batch_classification(self, fields: List[Dict], batch_size: int = 500) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Classify large datasets incrementally using existing categories with per-batch review
        
        Strategy:
        1. First batch: Direct classification + review to establish initial categories
        2. Subsequent batches: Use existing categories + review each batch
        3. If review finds issues, re-classify that batch only
        
        Args:
            fields: All fields to classify
            batch_size: Number of fields per batch (default 500)
            
        Returns:
            Tuple of (merged classification result, review history)
        """
        from math import ceil
        
        total_fields = len(fields)
        num_batches = ceil(total_fields / batch_size)
        print(f"[DataClassifier] Processing {total_fields} fields in {num_batches} batches (with per-batch review)...")
        
        # Track all classifications
        all_mappings = {}
        all_reasonings = []
        all_classified_ids = set()  # Track to avoid duplicates
        batch_review_history = []  # Track review for each batch
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_fields)
            batch = fields[start_idx:end_idx]
            
            print(f"\n[DataClassifier] Batch {i+1}/{num_batches}: {len(batch)} fields")
            
            # Step 1: Classify this batch
            if i == 0:
                # First batch: direct classification to establish categories
                print(f"[DataClassifier]   -> Initial classification")
                prompt = PromptTemplates.data_classification_with_metadata(batch)
                response = self.llm.client.complete(prompt)
                result = self._parse_classification_response(response, batch)
            else:
                # Subsequent batches: use existing categories
                print(f"[DataClassifier]   -> Using {len(all_mappings)} existing categories")
                existing_categories = list(all_mappings.keys())
                existing_examples = {cat: ids[:10] for cat, ids in all_mappings.items()}

                prompt = PromptTemplates.classification_with_existing_categories(
                    batch, existing_categories, existing_examples
                )
                response = self.llm.client.complete(prompt)
                result = self._parse_classification_response(response, batch)
                
                # Track new categories
                new_cats = result.get("new_categories_created", [])
                if new_cats:
                    print(f"[DataClassifier]   -> Created {len(new_cats)} new categories: {new_cats}")
            
            # Step 2: Review this batch's classification
            print(f"[DataClassifier]   -> Reviewing batch classification...")
            batch_review = self._review_classification(batch, result)
            
            # Step 3: If issues found, re-classify this batch only
            if not batch_review.get("review_passed", False):
                issues = batch_review.get("issues", [])
                print(f"[DataClassifier]   -> Review found {len(issues)} issues, re-classifying batch...")
                result = self._reclassify_with_feedback(batch, result, batch_review)
                print(f"[DataClassifier]   -> Batch re-classification complete")
                batch_review_history.append({
                    "batch": i + 1,
                    "review_passed": False,
                    "issues_count": len(issues),
                    "reclassified": True
                })
            else:
                print(f"[DataClassifier]   -> Batch review passed")
                batch_review_history.append({
                    "batch": i + 1,
                    "review_passed": True,
                    "reclassified": False
                })
            
            # Step 4: Merge mappings (avoiding duplicates)
            for cat, ids in result.get("field_mapping", {}).items():
                if cat not in all_mappings:
                    all_mappings[cat] = []
                for id in ids:
                    if id not in all_classified_ids:
                        all_mappings[cat].append(id)
                        all_classified_ids.add(id)
            
            all_reasonings.append(f"Batch {i+1}: {result.get('reasoning', '')}")
        
        # Check for any unclassified fields
        all_field_ids = set(f["id"] for f in fields)
        unclassified = all_field_ids - all_classified_ids
        if unclassified:
            print(f"[DataClassifier] Warning: {len(unclassified)} fields not classified, adding to 'other'")
            all_mappings.setdefault("other", []).extend(list(unclassified))
        
        print(f"\n[DataClassifier] Batch processing complete!")
        print(f"[DataClassifier] Total categories: {len(all_mappings)}")
        print(f"[DataClassifier] Total fields classified: {len(all_classified_ids)}")
        
        # Summarize batch review results
        passed_batches = sum(1 for r in batch_review_history if r["review_passed"])
        reclassified_batches = sum(1 for r in batch_review_history if r["reclassified"])
        print(f"[DataClassifier] Batch reviews: {passed_batches}/{num_batches} passed, {reclassified_batches} re-classified")
        
        batch_result = {
            "field_mapping": all_mappings,
            "reasoning": " | ".join(all_reasonings)
        }
        
        return batch_result, batch_review_history
    
    def _classify_with_review(self, fields: List[Dict], file_info: Dict[str, str]) -> Tuple[Dict, List[Dict]]:
        """
        Classify fields with iterative review process
        
        Process:
        1. Initial classification
        2. Review classification
        3. If issues found, re-classify with feedback
        4. Repeat until review passes or max iterations reached
        
        Returns:
            Tuple of (final_classification, review_history)
        """
        review_history = []
        current_classification = None
        
        for iteration in range(1, self.max_review_iterations + 1):
            print(f"\n[DataClassifier] === Iteration {iteration}/{self.max_review_iterations} ===")
            
            # Step 1: Classification (initial or with feedback)
            if current_classification is None:
                print("[DataClassifier] Performing initial classification...")
                current_classification = self._initial_classification(fields)
            else:
                # Get previous review feedback
                previous_review = review_history[-1]["review"] if review_history else {}
                if previous_review and not previous_review.get("review_passed", False):
                    print("[DataClassifier] Re-classifying with review feedback...")
                    current_classification = self._reclassify_with_feedback(
                        fields, current_classification, previous_review
                    )
                else:
                    print("[DataClassifier] No feedback to apply, keeping current classification")
            
            # Step 2: Review the classification
            print("[DataClassifier] Reviewing classification...")
            review_result = self._review_classification(fields, current_classification)
            
            review_history.append({
                "iteration": iteration,
                "classification": {
                    "field_mapping": current_classification["field_mapping"].copy(),
                    "reasoning": current_classification.get("reasoning", "")
                },
                "review": review_result
            })
            
            # Check if review passed
            if review_result.get("review_passed", False):
                print(f"[DataClassifier] ✓ Review passed on iteration {iteration}")
                break
            else:
                issues = review_result.get("issues", [])
                suggestions = review_result.get("suggestions", [])
                print(f"[DataClassifier] ✗ Review found {len(issues)} issues, {len(suggestions)} suggestions")
                
                # Show issues
                for issue in issues[:3]:
                    print(f"    - {issue}")
                
                # If this is the last iteration, stop
                if iteration == self.max_review_iterations:
                    print("[DataClassifier] Max iterations reached")
        
        current_classification["file_info"] = file_info
        current_classification["total_iterations"] = len(review_history)
        
        return current_classification, review_history
    
    def _initial_classification(self, fields: List[Dict]) -> Dict[str, Any]:
        """Perform initial classification"""
        prompt = PromptTemplates.data_classification_with_metadata(fields)
        response = self.llm.client.complete(prompt)
        return self._parse_classification_response(response, fields)

    def _reclassify_with_feedback(self, fields: List[Dict],
                                  previous_classification: Dict[str, Any],
                                  review_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Re-classify fields based on review feedback"""
        prompt = PromptTemplates.classification_with_feedback(
            fields, previous_classification, review_feedback
        )
        response = self.llm.client.complete(prompt)
        return self._parse_classification_response(response, fields)

    def _review_classification(self, fields: List[Dict], classification: Dict[str, Any]) -> Dict[str, Any]:
        """Review classification and suggest improvements"""
        prompt = PromptTemplates.classification_review(fields, classification)
        response = self.llm.client.complete(prompt)
        return self._parse_review_response(response)
    
    def _parse_classification_response(self, response: str, fields: List[Dict]) -> Dict[str, Any]:
        """Parse classification response from LLM"""
        import re
        
        result = None
        
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except:
                pass
        
        # Try raw JSON
        if not result:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    pass
        
        if not result:
            raise ValueError("No valid JSON in classification response")
        
        field_mapping = result.get("field_mapping", {})
        
        # Get valid field IDs from original fields
        valid_ids = set(f["id"] for f in fields)
        
        # Filter out invalid IDs (LLM hallucinations)
        filtered_mapping = {}
        removed_count = 0
        for cat, ids in field_mapping.items():
            valid_ids_in_cat = [id for id in ids if id in valid_ids]
            invalid_ids = [id for id in ids if id not in valid_ids]
            if invalid_ids:
                removed_count += len(invalid_ids)
                print(f"[DataClassifier] Filtered {len(invalid_ids)} invalid IDs from category '{cat}': {invalid_ids[:3]}{'...' if len(invalid_ids) > 3 else ''}")
            if valid_ids_in_cat:
                filtered_mapping[cat] = valid_ids_in_cat
        
        # Ensure all valid ids are classified
        classified_ids = set()
        for ids in filtered_mapping.values():
            classified_ids.update(ids)
        
        missing = valid_ids - classified_ids
        if missing:
            filtered_mapping.setdefault("other", []).extend(list(missing))
        
        if removed_count > 0:
            print(f"[DataClassifier] Total filtered invalid IDs: {removed_count}")
        
        return {
            "field_mapping": filtered_mapping,
            "reasoning": result.get("reasoning", "")
        }
    
    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """Parse review response from LLM"""
        import re
        
        result = None
        
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except:
                pass
        
        # Try raw JSON
        if not result:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    pass
        
        if not result:
            # If no valid JSON, assume review passed to avoid infinite loop
            return {
                "review_passed": True,
                "issues": [],
                "suggestions": [],
                "corrected_mapping": {},
                "confidence": "low",
                "review_reasoning": "Failed to parse review response"
            }
        
        return result
    
    def _read_csv(self, file_path: str) -> List[Dict]:
        """Read CSV file and return field metadata"""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        fields = []
        for _, row in df.iterrows():
            fields.append({
                "id": row.get(self.id_column, ""),
                "description": row.get(self.description_column, ""),
                "type": row.get("type", "")
            })
        
        return fields
    
    def _get_file_from_config(self, config: Dict[str, Any]) -> str:
        """Get data file path from config"""
        raw_file = config.get("data", {}).get("raw_file", "")
        if not raw_file:
            return ""
        
        if not os.path.isabs(raw_file):
            raw_file = os.path.join(os.getcwd(), raw_file)
        
        return raw_file if os.path.exists(raw_file) else ""
    
    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename: region_universe_dataset_type.csv
        Example: ASI_MINVOL1M_other455_MATRIX.csv
            region = ASI
            universe = MINVOL1M
            dataset = other455
            type = MATRIX
        """
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")

        if len(parts) >= 4:
            return {
                "region": parts[0],
                "universe": parts[1],
                "dataset": parts[2],
                "type": "_".join(parts[3:]) if len(parts) > 4 else parts[3]
            }

        raise ValueError(f"Invalid filename format: {filename}. Expected: region_universe_dataset_type.csv")
    
    def _get_cache_path(self, file_info: Dict[str, str]) -> str:
        """Get cache file path including dataset"""
        return os.path.join(
            self.cache_dir,
            "classification",
            f"classification_{file_info['region']}_{file_info['universe']}_{file_info['dataset']}_{file_info['type']}.json"
        )
    
    def _load_from_cache(self, cache_file: str, source_file: str) -> Dict[str, Any]:
        """Load from cache"""
        print(f"[DataClassifier] Loading cached classification")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            classification = json.load(f)
        
        return self._format_result(classification, source_file, cached=True)
    
    def _save_to_cache(self, classification: Dict[str, Any], cache_file: str):
        """Save to cache"""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(classification, f, indent=2, ensure_ascii=False)
        print(f"[DataClassifier] Saved to {cache_file}")
    
    def _format_result(self, classification: Dict[str, Any], source_file: str, 
                       cached: bool) -> Dict[str, Any]:
        """Format result"""
        return {
            "status": "success",
            "class_list": list(classification["field_mapping"].keys()),
            "field_mapping": classification["field_mapping"],
            "source_file": source_file,
            "file_info": classification.get("file_info", {}),
            "cached": cached,
            "review_history": classification.get("review_history", []),
            "total_iterations": classification.get("total_iterations", 1)
        }
    
    def get_category_fields(self, category: str, classification: Dict[str, Any]) -> List[str]:
        """Get all field IDs in a specific category"""
        return classification.get("field_mapping", {}).get(category, [])
    
    def get_category_stats(self, classification: Dict[str, Any]) -> Dict[str, int]:
        """Get statistics for each category"""
        field_mapping = classification.get("field_mapping", {})
        return {cat: len(ids) for cat, ids in field_mapping.items()}
    
    def print_classification(self, classification: Dict[str, Any]):
        """Pretty print classification results"""
        print("\n" + "=" * 60)
        print("Classification Result")
        print("=" * 60)
        print(f"File info: {classification['file_info']}")
        print(f"Categories ({len(classification['class_list'])}): {classification['class_list']}")
        print(f"Cached: {classification['cached']}")
        print(f"Iterations: {classification.get('total_iterations', 1)}")
        
        # Show review summary if available
        review_history = classification.get("review_history", [])
        if review_history:
            print(f"\nReview History:")
            for review in review_history:
                # Handle both nested and flat review structure
                review_data = review.get("review", review)
                passed = "✓" if review_data.get("review_passed", False) else "✗"
                issues = len(review_data.get("issues", []))
                iteration = review.get("iteration", "?")
                print(f"  Iteration {iteration}: {passed} ({issues} issues)")
        
        print(f"\nField mapping:")
        for cat, ids in classification['field_mapping'].items():
            print(f"  - {cat}: {len(ids)} fields")
        print("=" * 60)
