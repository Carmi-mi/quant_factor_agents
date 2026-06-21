"""
Expression Generator Agent
Generates alpha expressions from ideas using LLM
"""
import re
import json
import os
import itertools
from typing import Dict, Any, List
from core.agent_base import Agent
from services.llm_service import LLMService
from infrastructure.prompt_templates import PromptTemplates


class ExpressionGeneratorAgent(Agent):
    """Expression generation agent with LLM support"""

    # Exempt fields: basic market data fields that LLM can use but not required
    # Maps correct field names to common mistakes/aliases
    EXEMPT_FIELDS = {
        'returns': ['ret', 'return', 'daily_return', 'daily_returns'],
        'volume': ['vol', 'volm'],
        'adv20': [],
        'cap': [],
        'close': [],
        'dividend': [],
        'high': [],
        'low': [],
        'open': [],
        'sharesout': [],
        'split': [],
        'vwap': [],
        'gaussian': [],
        'uniform': [],
        'cauchy': []
    }

    # Standard group fields for MATRIX type data
    # When type is MATRIX, group operators can only use these fields
    STANDARD_GROUP_FIELDS = ['sector', 'country', 'market', 'industry', 'subindustry']

    def __init__(self, config: Dict[str, Any],
                 llm_service: LLMService = None):
        """
        Initialize

        Args:
            config: Configuration dictionary
            llm_service: LLM service for intelligent generation
        """
        super().__init__("ExpressionGenerator", config)
        self.llm = llm_service
        self.use_llm = llm_service is not None
        self.operators_doc = self._load_operators_doc()

        # Initialize template cache
        self.cache_dir = config.get("data", {}).get("cache_dir", "cache")
        self.template_cache_dir = os.path.join(self.cache_dir, 'templates')
        os.makedirs(self.template_cache_dir, exist_ok=True)
        # Cache file will be set dynamically based on file_info
        self.template_cache_file = None
        self.template_cache = []

    def _correct_field_names(self, expr_code: str) -> str:
        """
        Correct common field name mistakes in expressions
        
        Args:
            expr_code: The expression code to correct
            
        Returns:
            Corrected expression code
        """
        corrected = expr_code
        corrections_made = []
        
        # Build reverse mapping: mistake -> correct field
        mistake_to_correct = {}
        for correct_field, mistakes in self.EXEMPT_FIELDS.items():
            for mistake in mistakes:
                mistake_to_correct[mistake] = correct_field
        
        # Find all identifiers in the expression
        tokens = re.findall(r'\b[a-zA-Z_]+\b', corrected)
        
        # Replace mistakes with correct field names
        for token in set(tokens):
            if token in mistake_to_correct:
                correct_field = mistake_to_correct[token]
                # Use word boundary to ensure whole word replacement
                corrected = re.sub(r'\b' + token + r'\b', correct_field, corrected)
                corrections_made.append(f"{token} -> {correct_field}")
        
        if corrections_made:
            print(f"[ExpressionGenerator] Field corrections: {', '.join(corrections_made)}")

        return corrected

    def _validate_template_operators(self, template: str) -> str:
        """
        Validate operators in template before expansion.
        Remove invalid operators (not in operators_desc.json) along with their parentheses and arguments.

        Args:
            template: The template string with {placeholders}

        Returns:
            Cleaned template string
        """
        # Load valid operators from operators_desc.json (simple names without signatures)
        valid_ops = self._load_valid_operators()

        cleaned = template
        removed_ops = []

        # Find all function calls (identifier followed by opening parenthesis)
        func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('

        i = 0
        while i < len(cleaned):
            match = re.search(func_pattern, cleaned[i:])
            if not match:
                break

            func_name = match.group(1)
            start_pos = i + match.start()
            paren_start = i + match.end() - 1  # Position of '('

            # Check if this is inside a placeholder {var}
            # We're inside if there's an unmatched { before us
            before = cleaned[:start_pos]
            last_open = before.rfind('{')
            last_close = before.rfind('}')

            if last_open > last_close:
                # Inside placeholder, skip
                i = paren_start + 1
                continue

            # Check if operator is valid
            if func_name not in valid_ops:
                # Find matching closing parenthesis (handle nesting)
                paren_count = 1
                paren_end = paren_start + 1
                while paren_end < len(cleaned) and paren_count > 0:
                    if cleaned[paren_end] == '(':
                        paren_count += 1
                    elif cleaned[paren_end] == ')':
                        paren_count -= 1
                    paren_end += 1

                # Remove the entire function call
                removed_ops.append(func_name)
                cleaned = cleaned[:start_pos] + cleaned[paren_end:]
                # Don't advance i, check from same position again
            else:
                i = paren_start + 1

        if removed_ops:
            print(f"[ExpressionGenerator] Removed invalid operators from template: {', '.join(removed_ops)}")

        return cleaned

    def _identify_variable_type(self, var_name: str) -> str:
        """
        根据变量名识别类型（基于提示词规范）
        
        Returns:
            'field', 'operator', 'number', 'group', 'unknown'
        """
        var_lower = var_name.lower()
        
        # 字段：以 field 开头
        if var_lower.startswith('field'):
            return 'field'
        
        # 操作符：以 op 或 operator 开头
        if var_lower.startswith('op') or var_lower.startswith('operator'):
            return 'operator'
        
        # 窗口：以 window/period/lookback/days 开头
        if any(var_lower.startswith(x) for x in ['window', 'period', 'lookback', 'days']):
            return 'number'
        
        # 分组：以 group/category/cat 开头
        if var_lower.startswith('group') or var_lower.startswith('category') or var_lower.startswith('cat'):
            return 'group'
        
        return 'unknown'
    
    def _validate_and_correct_variables(self, variables: Dict[str, List[str]], 
                                        field_mapping: Dict[str, List[str]],
                                        valid_operators: set) -> tuple:
        """
        验证并纠正变量中的字段名
        
        策略：
        1. 根据变量名识别类型
        2. 字段类型：合并所有可用字段（包括豁免字段），子串匹配纠正
        3. 操作符类型：验证是否在有效操作符集合中
        4. 数值类型：直接保留
        5. 分组类型：直接保留（暂不验证）
        6. 无匹配则丢弃，变量为空则返回失败
        
        Args:
            variables: 变量字典
            field_mapping: 可用字段映射
            valid_operators: 有效操作符集合
            
        Returns:
            (纠正后的变量字典, 是否有效)
        """
        # 合并所有可用字段（普通字段 + 豁免字段）
        all_fields = set()
        for fields in field_mapping.values():
            all_fields.update(fields)
        all_fields.update(self.EXEMPT_FIELDS.keys())
        all_fields = list(all_fields)
        
        corrected_vars = {}
        
        for var_name, values in variables.items():
            var_type = self._identify_variable_type(var_name)
            corrected_values = []
            
            if var_type == 'operator':
                # 操作符类型：验证有效性
                for val in values:
                    if val in valid_operators:
                        corrected_values.append(val)
                    else:
                        print(f"[ExpressionGenerator] Discarded invalid operator: {val}")
                
            elif var_type == 'number':
                # 数值类型：直接保留
                corrected_values = values
                
            elif var_type == 'group':
                # 分组类型：直接保留（暂不验证）
                corrected_values = values
                
            elif var_type == 'field' or var_type == 'unknown':
                # 字段类型或未知类型：进行子串匹配纠正
                for val in values:
                    if val in all_fields:
                        # 完全匹配
                        corrected_values.append(val)
                    else:
                        # 子串匹配
                        val_lower = val.lower()
                        candidates = []
                        
                        for field in all_fields:
                            field_lower = field.lower()
                            # 提取关键词（去掉前缀）
                            field_keyword = field_lower.split('_')[-1] if '_' in field_lower else field_lower
                            
                            # 双向子串匹配
                            if val_lower in field_keyword or field_keyword in val_lower:
                                candidates.append(field)
                        
                        if candidates:
                            # 选最短的字段
                            best_match = min(candidates, key=len)
                            corrected_values.append(best_match)
                            print(f"[ExpressionGenerator] Corrected field: {val} -> {best_match}")
                        else:
                            print(f"[ExpressionGenerator] Discarded unknown field: {val}")
            
            # 去重保持顺序
            seen = set()
            unique_values = []
            for v in corrected_values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)
            
            # 检查变量是否为空
            if not unique_values:
                print(f"[ExpressionGenerator] WARNING: Variable '{var_name}' is empty after validation")
                return {}, False
            
            corrected_vars[var_name] = unique_values
        
        return corrected_vars, True

    def _load_operators_doc(self) -> Dict[str, str]:
        """Load operators documentation from config file (operators.json with signatures)"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'operators.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ExpressionGenerator] Warning: Failed to load operators.json: {e}")
            return {}

    def _load_valid_operators(self) -> set:
        """Load valid operator names from operators_desc.json for template validation"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'operators_desc.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                ops_desc = json.load(f)
                return set(ops_desc.keys())
        except Exception as e:
            print(f"[ExpressionGenerator] Warning: Failed to load operators_desc.json: {e}")
            return set()

    def _setup_template_cache(self, file_info: Dict[str, Any]):
        """Setup template cache file path based on file_info"""
        region = file_info.get('region', 'UNKNOWN')
        universe = file_info.get('universe', 'UNKNOWN')
        dataset = file_info.get('dataset', 'UNKNOWN')
        data_type = file_info.get('type', 'MATRIX')

        cache_key = f"{region}_{universe}_{dataset}_{data_type}"
        self.template_cache_file = os.path.join(self.template_cache_dir, f'templates_{cache_key}.json')
        self.template_cache = self._load_template_cache()
        print(f"[ExpressionGenerator] Template cache file: {self.template_cache_file}")

    def _load_template_cache(self) -> List[Dict[str, Any]]:
        """Load template cache from file"""
        if self.template_cache_file and os.path.exists(self.template_cache_file):
            try:
                with open(self.template_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ExpressionGenerator] Warning: Failed to load template cache: {e}")
        return []

    def _save_template_cache(self):
        """Save template cache to file"""
        try:
            with open(self.template_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.template_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ExpressionGenerator] Warning: Failed to save template cache: {e}")

    def _cache_template(self, idea_content: str, template: str):
        """
        Cache a generated template

        Args:
            idea_content: The idea content (description)
            template: The expression template
        """
        template_entry = {
            "description": idea_content,
            "template": template
        }

        # Check if template already exists (by template string)
        existing_idx = None
        for idx, entry in enumerate(self.template_cache):
            if entry.get("template") == template:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Update existing entry
            self.template_cache[existing_idx] = template_entry
            print(f"[ExpressionGenerator] Updated template cache for template: {template[:50]}...")
        else:
            # Add new entry
            self.template_cache.append(template_entry)
            print(f"[ExpressionGenerator] Added template to cache: {template[:50]}...")

        # Save cache
        self._save_template_cache()

    def _get_operator_docs(self, use_ops: List[str]) -> Dict[str, str]:
        """
        Get operator documentation for specified operators

        Args:
            use_ops: List of operator names from idea's use_op

        Returns:
            Dictionary mapping operator names to their documentation
        """
        op_docs = {}

        for op in use_ops:
            op_clean = op.strip().lower()

            # Try exact match first
            if op_clean in self.operators_doc:
                op_docs[op] = self.operators_doc[op_clean]
                continue

            # Try partial match (operator name appears in key)
            for key, doc in self.operators_doc.items():
                key_clean = key.strip().lower()
                # Match operator name at start of key (e.g., "ts_sum" matches "ts_sum(x, d)")
                if key_clean.startswith(op_clean + '(') or key_clean == op_clean:
                    # Use the original key from operators.json as the operator name
                    op_docs[key] = doc
                    break

        return op_docs

    def _correct_operator_names(self, use_ops: List[str]) -> List[str]:
        """
        Correct operator names to match operators.json keys

        Args:
            use_ops: List of operator names from idea's use_op

        Returns:
            List of corrected operator names (matching operators.json keys)
        """
        corrected_ops = []

        for op in use_ops:
            op_clean = op.strip().lower()

            # Try exact match first
            if op_clean in self.operators_doc:
                # Find the original key that matches
                for key in self.operators_doc.keys():
                    if key.strip().lower() == op_clean:
                        corrected_ops.append(key)
                        break
                continue

            # Try partial match (operator name appears in key)
            found = False
            for key in self.operators_doc.keys():
                key_clean = key.strip().lower()
                # Match operator name at start of key (e.g., "ts_sum" matches "ts_sum(x, d)")
                if key_clean.startswith(op_clean + '(') or key_clean == op_clean:
                    corrected_ops.append(key)
                    found = True
                    break

            if not found:
                # If no match found, keep original
                corrected_ops.append(op)

        return corrected_ops

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate expressions from ideas

        Args:
            input_data: {"ideas": [...], "field_mapping": {...}, "file_info": {...}}

        Returns:
            Generated expressions with file_info preserved
        """
        # Validate input
        self.validate_input(input_data, ["ideas", "field_mapping"])

        ideas = input_data["ideas"]
        field_mapping = input_data["field_mapping"]
        file_info = input_data.get("file_info", {})
        data_type = file_info.get("type", "MATRIX")  # Default to MATRIX

        # Set up template cache file based on file_info
        self._setup_template_cache(file_info)

        print(f"[ExpressionGenerator] Generating expressions for {len(ideas)} ideas")
        print(f"[ExpressionGenerator] Data type: {data_type}")
        print(f"[ExpressionGenerator] Using LLM: {self.use_llm}")

        idea_results = []
        total_count = 0

        for idea in ideas:
            idea_result = self._generate_for_idea(idea, field_mapping, data_type)
            idea_results.append(idea_result)
            total_count += len(idea_result.get("expressions", []))

        print(f"[ExpressionGenerator] Generated {total_count} expressions for {len(ideas)} ideas")

        return {
            "status": "success",
            "ideas": idea_results,
            "count": total_count,
            "file_info": file_info
        }
    
    def _generate_for_idea(self, idea: Dict[str, Any],
                          field_mapping: Dict[str, List[str]],
                          data_type: str = "MATRIX") -> Dict[str, Any]:
        """Generate expressions for a single idea using LLM only"""
        expressions = []

        if self.use_llm:
            try:
                llm_exprs = self._generate_with_llm(idea, field_mapping, data_type)
                expressions.extend(llm_exprs)
            except Exception as e:
                print(f"[ExpressionGenerator] LLM generation failed for idea {idea.get('id', 'unknown')}: {e}")

        return {
            "idea_id": idea.get("id", ""),
            "idea_content": idea.get("content", ""),
            "expressions": expressions[:100],
            "count": len(expressions[:100])
        }

    def _generate_with_llm(self, idea: Dict[str, Any],
                          field_mapping: Dict[str, List[str]],
                          data_type: str = "MATRIX") -> List[Dict[str, Any]]:
        """Generate expressions using LLM template"""
        expressions = []

        # Filter field_mapping to only include fields from idea's use_class
        use_classes = idea.get("use_class", [])
        filtered_field_mapping = {}
        for cls in use_classes:
            if cls in field_mapping:
                filtered_field_mapping[cls] = field_mapping[cls]

        # If no matching classes found, use all field_mapping as fallback
        if not filtered_field_mapping:
            filtered_field_mapping = field_mapping
            print(f"[ExpressionGenerator] Warning: No matching fields for use_class {use_classes}, using all fields")

        print(f"[ExpressionGenerator] LLM using fields from categories: {list(filtered_field_mapping.keys())}")

        # Get operator documentation for idea's use_op
        use_ops = idea.get("use_op", [])

        # Correct operator names to match operators.json keys
        use_ops = self._correct_operator_names(use_ops)
        print(f"[ExpressionGenerator] Corrected operators: {use_ops}")

        op_docs = self._get_operator_docs(use_ops)
        print(f"[ExpressionGenerator] LLM using operators: {list(op_docs.keys())}")

        # Generate prompt for template with data_type
        prompt = PromptTemplates.expression_template_generation(idea, filtered_field_mapping, op_docs, data_type)

        # Call LLM
        response = self.llm.client.complete(prompt)

        # Store raw response for inspection
        self._last_llm_response = response
        self._last_llm_prompt = prompt

        # Parse template response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                template = result.get("template", "")
                variables = result.get("variables", {})
                logic = result.get("logic", "")

                if template and variables:
                    # Check if any variable list is empty
                    empty_vars = [var_name for var_name, var_values in variables.items() if not var_values]
                    if empty_vars:
                        print(f"[ExpressionGenerator] WARNING: Empty variable list(s) for {empty_vars}, skipping this idea")
                        return []  # Return empty list to abandon this idea

                    # Validate and clean template before expansion
                    template = self._validate_template_operators(template)
                    if not template:
                        print(f"[ExpressionGenerator] WARNING: Template became empty after operator validation, skipping this idea")
                        return []

                    # Validate and correct variables before expansion
                    valid_operators = self._load_valid_operators()
                    variables, is_valid = self._validate_and_correct_variables(
                        variables, filtered_field_mapping, valid_operators
                    )
                    if not is_valid:
                        print(f"[ExpressionGenerator] WARNING: Variable validation failed, skipping this idea")
                        return []

                    # Expand template into multiple expressions
                    expanded = self._expand_template(template, variables)

                    # Correct field names and remove spaces in expanded expressions
                    corrected = [self._correct_field_names(expr).replace(" ", "") for expr in expanded]
                    expressions.extend(corrected)

                    print(f"[ExpressionGenerator] Expanded template to {len(expanded)} expressions")

                    # Cache the template
                    self._cache_template(
                        idea_content=idea.get("content", ""),
                        template=template
                    )
        except Exception as e:
            print(f"[ExpressionGenerator] Failed to parse LLM response: {e}")

        return expressions

    def _expand_template(self, template: str, variables: Dict[str, List[Any]]) -> List[str]:
        """
        Expand template into multiple expressions using variable combinations
        
        Args:
            template: The template string with {{placeholder}}
            variables: Dictionary mapping placeholders to possible values
            
        Returns:
            List of expanded expressions
        """
        # Convert all values to strings
        str_vars = {}
        for key, values in variables.items():
            str_vars[key] = [str(v) for v in values]

        # Get all variable names in template
        var_names = list(str_vars.keys())
        
        # Generate all combinations using Cartesian product
        if not var_names:
            return [template]

        # Create list of value lists
        value_lists = [str_vars[name] for name in var_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*value_lists))

        # Generate expressions from each combination
        expressions = []
        for combo in combinations:
            expr = template
            for name, value in zip(var_names, combo):
                # Support both {name} and {{name}} formats
                expr = expr.replace(f"{{{name}}}", value)
                expr = expr.replace(f"{{{{{name}}}}}", value)
            expressions.append(expr)

        return expressions
