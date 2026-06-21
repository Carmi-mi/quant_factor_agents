"""
Prompt Templates for LLM Agents
Centralized management of all prompts used in the system
"""
import json
from typing import Dict, Any, List


class PromptTemplates:
    """Prompt template manager"""
    
    # ==================== DataClassifier Prompts ====================

    @staticmethod
    def data_classification_with_metadata(field_metadata: List[Dict[str, Any]]) -> str:
        """
        Prompt for classifying fields using id + description metadata
        All fields are included in the prompt for direct classification

        Args:
            field_metadata: List of field metadata with id, description, type, etc.
        """
        total_fields = len(field_metadata)
        
        # Build fields description - include ALL fields
        fields_str = "\n".join([
            f"- ID: {f['id']}\n  Description: {f['description']}\n  Type: {f.get('type', 'N/A')}"
            for f in field_metadata
        ])

        return f"""You are a data classification expert. Please classify ALL the following fields into appropriate categories based on their descriptions.

Total Fields to Classify: {total_fields}

Fields:
{fields_str}

Your Task:
1. Analyze each field's ID and description
2. Identify logical groupings based on the data characteristics
3. Create appropriate category names (use English, lowercase, underscores)
4. Assign EACH field ID to exactly one category

Category Guidelines:
- Create 5-15 categories based on SEMANTIC/CONCEPTUAL meaning, NOT data source or naming patterns
- Categories should be mutually exclusive and collectively exhaustive
- Use descriptive names that reflect what the data MEASURES or REPRESENTS
- Group by: conceptual similarity, logical relationships, domain-specific meaning
- IGNORE data source prefixes, version numbers, or technical identifiers in field names
- Consider the actual meaning described in the field descriptions

Please output in JSON format:
{{
    "field_mapping": {{
        "category_name_1": ["id1", "id2", "id3"],
        "category_name_2": ["id4", "id5"],
        "category_name_3": ["id6", "id7", "id8"]
    }},
    "reasoning": "Brief explanation of your classification logic"
}}

IMPORTANT RULES:
1. EVERY field ID must be assigned to exactly one category
2. ALL {total_fields} field IDs must be classified (no missing fields)
3. Category names should be lowercase with underscores, describing the CONCEPTUAL meaning
4. Provide clear reasoning explaining the classification logic
5. DO NOT group by data source prefixes or technical identifiers - group by what the data REPRESENTS conceptually"""

    @staticmethod
    def data_classification(columns: List[str], sample_data: List[Dict] = None) -> str:
        """
        Prompt for classifying data columns - LLM decides categories
        
        Args:
            columns: List of column names
            sample_data: Optional sample rows for context
        """
        sample_str = ""
        if sample_data:
            sample_str = f"\nSample data (first 3 rows):\n{sample_data[:3]}"
        
        return f"""You are a data classification expert. Please analyze and classify the following columns into appropriate categories.

Columns to classify:
{', '.join(columns)}{sample_str}

Your Task:
1. Analyze the column names and sample data
2. Identify logical groupings based on the data characteristics
3. Create appropriate category names (use English, lowercase, no spaces)
4. Assign each column to exactly one category

Category Guidelines:
- Create 3-8 categories that make sense for this dataset
- Categories should be mutually exclusive
- Use descriptive names that reflect the conceptual meaning of the data
- Consider domain-specific conventions but adapt to the actual data

Please output in JSON format:
{{
    "field_mapping": {{
        "category_name_1": ["field1", "field2"],
        "category_name_2": ["field3", "field4"],
        "category_name_3": ["field5"]
    }},
    "reasoning": "Brief explanation of your classification logic and why you chose these categories"
}}

Rules:
1. Each field must be assigned to exactly one category
2. All columns must be classified (no missing fields)
3. Category names should be lowercase with underscores if needed
4. Provide clear reasoning for your categorization choices"""

    @staticmethod
    def classification_review(field_metadata: List[Dict[str, Any]], 
                             current_classification: Dict[str, Any]) -> str:
        """
        Prompt for reviewing and critiquing classification results
        
        Args:
            field_metadata: Original field metadata
            current_classification: Current classification result to review
        """
        total_fields = len(field_metadata)
        field_mapping = current_classification.get("field_mapping", {})
        reasoning = current_classification.get("reasoning", "")
        
        # Build detailed current classification with all field IDs
        mapping_str = "\n\n".join([
            f"Category: {cat}\nFields ({len(ids)}): {', '.join(ids[:20])}{'...' if len(ids) > 20 else ''}"
            for cat, ids in field_mapping.items()
        ])
        
        # Build original fields with descriptions for reference
        fields_str = "\n".join([
            f"- ID: {f['id']} | Description: {f['description']}"
            for f in field_metadata[:50]  # Show first 50 to avoid too long prompt
        ])
        if len(field_metadata) > 50:
            fields_str += f"\n... and {len(field_metadata) - 50} more fields"
        
        # Check for potential issues
        all_ids = set(f["id"] for f in field_metadata)
        classified_ids = set()
        for ids in field_mapping.values():
            classified_ids.update(ids)
        
        missing = all_ids - classified_ids
        extra = classified_ids - all_ids
        
        issues_str = ""
        if missing:
            issues_str += f"\n- Missing fields: {list(missing)[:10]}"
        if extra:
            issues_str += f"\n- Extra fields not in original: {list(extra)[:10]}"
        
        return f"""You are a senior data classification expert reviewing a classification result. Your task is to critically evaluate the classification and suggest improvements.

Original Fields: {total_fields} total

Sample of Original Fields with Descriptions:
{fields_str}

Current Classification:
{mapping_str}

Original Reasoning:
{reasoning}

Potential Issues Detected:{issues_str if issues_str else " None detected"}

Your Task - Review and Critique:

1. **Completeness Check**:
   - Are ALL {total_fields} fields classified?
   - Are there any fields that should be moved to different categories?

2. **Consistency Check**:
   - Are similar fields grouped together logically?
   - Are there fields with similar economic meaning in different categories?
   - Are category boundaries clear and meaningful?

3. **Economic Logic Check**:
   - Does each category represent a coherent economic concept?
   - Are the categories mutually exclusive in terms of economic meaning?
   - Would a financial analyst agree with these groupings?

4. **Common Issues to Look For**:
   - Fields with similar conceptual meaning should be in the same category
   - Related concepts should be grouped together (e.g., inputs and outputs of the same process)
   - Fields representing different aspects of the same entity should be logically organized
   - Any overly broad "other" or "miscellaneous" category should be minimized

5. **Specific Review Questions**:
   - Are fields with similar descriptions grouped together?
   - Do the categories reflect the actual semantic relationships?
   - Are there fields that clearly belong to a different category?
   - Is the categorization consistent across all fields?

Output Format - JSON:
{{
    "review_passed": false,
    "issues": [
        "Issue 1: description of the problem",
        "Issue 2: description of the problem"
    ],
    "suggestions": [
        "Move field X from category A to category B because...",
        "Merge categories C and D because...",
        "Split category E into F and G because..."
    ],
    "confidence": "high/medium/low",
    "review_reasoning": "Detailed explanation of your review findings"
}}

IMPORTANT:
- Set "review_passed" to true ONLY if the classification is perfect
- If there are ANY issues, set it to false and provide specific suggestions
- Focus on giving clear, actionable suggestions for improvement
- DO NOT provide corrected_mapping - the system will re-classify based on your suggestions"""

    @staticmethod
    def classification_with_feedback(field_metadata: List[Dict[str, Any]], 
                                     previous_classification: Dict[str, Any],
                                     review_feedback: Dict[str, Any]) -> str:
        """
        Prompt for re-classifying fields based on review feedback
        
        Args:
            field_metadata: Original field metadata
            previous_classification: Previous classification attempt
            review_feedback: Review feedback with suggestions
        """
        total_fields = len(field_metadata)
        field_mapping = previous_classification.get("field_mapping", {})
        suggestions = review_feedback.get("suggestions", [])
        issues = review_feedback.get("issues", [])
        
        # Build previous classification
        mapping_str = "\n\n".join([
            f"Category: {cat}\nFields ({len(ids)}): {', '.join(ids[:15])}{'...' if len(ids) > 15 else ''}"
            for cat, ids in field_mapping.items()
        ])
        
        # Build suggestions
        suggestions_str = "\n".join([f"- {s}" for s in suggestions]) if suggestions else "No specific suggestions provided."
        
        # Build issues
        issues_str = "\n".join([f"- {i}" for i in issues]) if issues else "No specific issues identified."
        
        # Build all fields
        fields_str = "\n".join([
            f"- ID: {f['id']} | Description: {f['description']}"
            for f in field_metadata
        ])
        
        return f"""You are a data classification expert. You previously classified a set of fields, but a senior reviewer provided feedback. Your task is to re-classify ALL fields based on the review feedback.

Total Fields to Classify: {total_fields}

ALL Fields with Descriptions:
{fields_str}

Previous Classification:
{mapping_str}

Review Issues Identified:
{issues_str}

Review Suggestions:
{suggestions_str}

Your Task:
1. Carefully read all the review feedback
2. Address EVERY issue mentioned in the review
3. Follow ALL suggestions provided
4. Re-classify ALL {total_fields} fields into appropriate categories
5. Ensure the new classification fixes all identified problems

Key Principles:
- Group by SEMANTIC/CONCEPTUAL meaning, not data source or naming patterns
- Ensure fields with similar descriptions or purposes are in the same category
- Related concepts (e.g., inputs and outputs, causes and effects) should be logically organized
- Categories should reflect the actual relationships described in the field descriptions
- Minimize overly broad "other" or "miscellaneous" categories

Please output in JSON format:
{{
    "field_mapping": {{
        "category_name_1": ["id1", "id2", "id3"],
        "category_name_2": ["id4", "id5"],
        "category_name_3": ["id6", "id7", "id8"]
    }},
    "reasoning": "Explain how you addressed the review feedback and your classification logic"
}}

IMPORTANT RULES:
1. EVERY field ID must be assigned to exactly one category
2. ALL {total_fields} field IDs must be classified (no missing fields)
3. Address ALL issues mentioned in the review
4. Follow ALL suggestions from the review
5. Category names should be lowercase with underscores
6. Be careful to classify based on economic meaning, not just field names"""

    @staticmethod
    def classification_with_existing_categories(field_metadata: List[Dict[str, Any]],
                                               existing_categories: List[str],
                                               existing_examples: Dict[str, List[str]]) -> str:
        """
        Prompt for classifying fields using existing categories
        
        Args:
            field_metadata: Fields to classify
            existing_categories: List of existing category names
            existing_examples: Examples of fields in each category
        """
        total_fields = len(field_metadata)
        
        # Build existing categories with examples
        categories_str = "\n\n".join([
            f"Category: {cat}\nExamples: {', '.join(existing_examples.get(cat, [])[:10])}"
            for cat in existing_categories
        ])
        
        # Build fields to classify
        fields_str = "\n".join([
            f"- ID: {f['id']} | Description: {f['description']}"
            for f in field_metadata
        ])
        
        return f"""You are a data classification expert. Your task is to classify new fields into EXISTING categories, or create new categories if necessary.

Existing Categories (with examples):
{categories_str}

New Fields to Classify ({total_fields} total):
{fields_str}

Your Task:
1. For each new field, determine if it fits into one of the EXISTING categories
2. If a field fits an existing category, assign it there
3. If a field does NOT fit any existing category, create a NEW category with a descriptive name
4. Ensure each field is assigned to exactly one category

Rules:
- Use existing categories whenever possible
- Only create new categories for fields that truly don't fit existing ones
- New category names should be descriptive and follow the same naming style (lowercase with underscores)
- Consider the semantic meaning, not just the field name

Output Format - JSON:
{{
    "field_mapping": {{
        "existing_category_1": ["id1", "id2"],
        "existing_category_2": ["id3"],
        "new_category_name": ["id4", "id5"]
    }},
    "new_categories_created": ["new_category_name"],
    "reasoning": "Explain your classification decisions, especially for new categories"
}}

IMPORTANT:
1. Assign EACH field to exactly one category
2. Prefer existing categories over creating new ones
3. Create new categories only when necessary
4. All {total_fields} fields must be classified"""
    
    @staticmethod
    def idea_generation(class_list: List[str], op_descriptions: List[str],
                       num_ideas: int = 4, history: List[str] = None) -> str:
        """
        Prompt for generating factor ideas using operator descriptions
        """
        # Format operator descriptions WITHOUT numbers to avoid LLM copying them
        op_desc_str = "\n".join([f"- {desc}" for desc in op_descriptions])

        return f"""You are a quantitative finance expert specialized in alpha factor research. Generate EXACTLY {num_ideas} innovative and mutually distinct quantitative factor ideas.

Available Data Categories:
{', '.join(class_list)}

Available Operations (MUST use exact descriptions from the list below):
{op_desc_str}

Requirements:
1. EXACTLY {num_ideas} ideas - output precisely this number in the ideas array
2. Each idea MUST use at least 1 data category and 1 operation from the lists above
3. For operations, COPY the description EXACTLY as shown (without the leading dash)
4. Ideas must be MUTUALLY DISTINCT - different economic logic, different data combinations, or different operations
5. Each idea should capture a specific market anomaly with clear economic intuition

Operation Combination Rules:
- Simple ideas: 1-2 operations
- Complex ideas: 3-4 operations
- Avoid using more than 4 operations (overfitting risk)
- Prefer meaningful combinations (operations should build on each other sequentially)
- Distribute ideas across complexity levels
- BAD: Using many operations without clear purpose
- BAD: Nesting operations randomly without logical flow
- BAD: Applying the same type of operation repeatedly

Data Category Combination Guidelines:
- Single category: Use for simple, focused factors
- Two categories: Good for relative value or spread trades
- Three+ categories: Only when necessary for complex interactions
- Avoid combining unrelated categories without clear economic logic
- Ensure each idea can be expressed with the available operations

Economic Logic Framework (address all three):
- What pattern does it capture? (e.g., momentum, mean-reversion, value premium)
- Why does it exist? (e.g., behavioral bias, information asymmetry, liquidity constraints)
- How to monetize? (e.g., long/short portfolio construction, timing)

Diversity Guidelines:
- Vary data category combinations (don't use the same single category for all ideas)
- Mix operation types (time-series vs cross-sectional, momentum vs mean-reversion)
- Cover different market mechanisms (information diffusion, behavioral biases, risk premia)

Output Format (STRICT JSON):
{{
    "ideas": [
        {{
            "content": "Detailed factor description including economic logic and expected behavior",
            "use_class": ["category1", "category2"],
            "use_op": ["operation description 1", "operation description 2"]
        }}
    ]
}}

CRITICAL:
- Output EXACTLY {num_ideas} ideas
- Operation descriptions in use_op MUST match the Available Operations list exactly (copy only the text, not the leading dash)
- Ideas must be fundamentally different from each other"""
    
    # ==================== ExpressionGenerator Prompts ====================

    @staticmethod
    def expression_template_generation(idea: Dict[str, Any], field_mapping: Dict[str, List[str]],
                                      op_docs: Dict[str, str], data_type: str = "MATRIX") -> str:
        """
        Prompt for generating a compact expression template with variables

        Args:
            idea: The factor idea
            field_mapping: Available fields per category (already filtered by use_class)
            op_docs: Operator documentation dictionary {operator_name: documentation}
            data_type: Type of data ("MATRIX" or "GROUP")
        """
        # Build available fields string (显示所有字段)
        fields_str = "\n".join([f"- {cat}: {', '.join(fields)}"
                               for cat, fields in field_mapping.items()])

        # Build operator documentation string
        ops_doc_str = "\n".join([f"- {op}: {doc}" for op, doc in op_docs.items()])

        # 判断是否只使用一个类别
        use_classes = idea.get('use_class', [])
        single_category = len(use_classes) == 1

        # Exempt fields that can be used but not required
        # 当只使用一个类别时，不允许使用豁免字段
        exempt_fields = ["adv20", "cap", "close", "dividend", "high", "low", "open", 
                        "returns", "sharesout", "split", "volume", "vwap"]

        if single_category:
            exempt_fields_section = """Basic Market Data Fields:
- NOT ALLOWED: When using only one data category, you MUST ONLY use fields from that category
- Do NOT use any basic market data fields like close, volume, returns, etc."""
        else:
            exempt_fields_section = f"""Basic Market Data Fields (use ONLY these exact names if needed):
- {', '.join(exempt_fields)}"""

        # Group field rules based on data type
        if data_type == "MATRIX":
            group_field_rule = """Group Operator Rules (CRITICAL for MATRIX data):
- group_median, group_mean, group_zscore, etc. SECOND parameter MUST be one of: sector, country, market, industry, subindustry
- Do NOT use dataset-specific group fields (like oth455_xxx_cluster_xxx) in group operators"""
        else:
            group_field_rule = """Group Operator Rules for GROUP data:
- group_median, group_mean, group_zscore, etc. can use the group fields from Available Fields above"""

        return f"""You are a quantitative developer specializing in alpha expression generation.

Factor Idea:
{idea.get('content', '')}

Data Type: {data_type}
Required Data Categories: {', '.join(use_classes)}
Required Operators: {', '.join(idea.get('use_op', []))}

Available Fields (by Category):
{fields_str}

{exempt_fields_section}

Operator Documentation (MUST use these operators):
{ops_doc_str}

{group_field_rule}

CRITICAL FIELD USAGE RULES:
1. ONLY use field names listed above
2. For returns, use "returns" (NOT "ret", "return", or "daily_return")
3. For volume, use "volume" (NOT "vol" or "volm")
4. For close price, use "close" (NOT "adj_close" or "price")
5. Do NOT invent field names not in the lists above

Your Task:
Generate a compact template that can be used to create multiple variations of the same core idea.

Output Format (JSON):
{{
    "logic": "Brief description of the economic logic this template implements",
    "template": "The expression template with {{placeholders}} for variables",
    "variables": {{
        "field": ["field1", "field2", ...],
        "op": ["operator1", "operator2", ...],
        "window": [5, 10, 20],
        "group": ["sector"],
        ...
    }}
}}

Variable Naming Rules (MUST FOLLOW):
1. "field" - for data field names (from Available Fields)
2. "op" or "operator" - for operator names (from Required Operators)
3. "window", "period", "lookback", "days" - for numeric time windows
4. "group", "category", "cat" - for group/category names
5. Use descriptive names that clearly indicate the variable type

Time Window Values (MUST ONLY use these values):
- Valid windows: [1, 2, 3, 5, 10, 20, 40, 60, 120, 240, 504]
- DO NOT use any other values for time windows

Multiple Variables of Same Type:
- Use numbered suffix: "field1", "field2" or "op1", "op2"
- OR use descriptive suffix: "field_long", "field_short" or "op_agg", "op_transform"
- Examples: "window_short", "window_long", "group_sector", "group_industry"

Example (single variables):
{{
    "logic": "Mean reversion within customer cluster",
    "template": "rank(reverse(subtract({{op}}({{field}},{{window}}), group_median({{op}}({{field}},{{window}}), {{group}}))))",
    "variables": {{
        "op": ["ts_sum", "ts_mean"],
        "field": ["close"],
        "window": [5, 10, 20],
        "group": ["customer_roam_kmeans_5"]
    }}
}}

Example (multiple same-type variables):
{{
    "logic": "Price momentum vs volume confirmation",
    "template": "correlation({{op1}}({{field_price}},{{window_short}}), {{op2}}({{field_volume}},{{window_long}}))",
    "variables": {{
        "op1": ["ts_sum", "ts_mean"],
        "op2": ["ts_sum", "ts_mean"],
        "field_price": ["close", "vwap"],
        "field_volume": ["volume", "adv20"],
        "window_short": [5, 10],
        "window_long": [20, 40]
    }}
}}

Requirements:
1. Use {{placeholder}} syntax for variables
2. Variable list names MUST follow the naming rules above
3. Variable values must be from the Available Fields and Required Operators
4. Template must implement the core economic logic
5. Keep it simple - no more than 4-5 variables

Generate the template now:"""

    # ==================== Alpha Improvement Prompts (ImprovementAgent使用) ====================

    @staticmethod
    def alpha_improvement_conversation(expr_code: str, source_idea: str,
                                        defect_reason: str, metrics: Dict[str, float],
                                        strategy: str, target_sharpe: float = 1.25,
                                        conversation_history: List[Dict] = None,
                                        backtest_history: List[Dict] = None,
                                        current_settings: Dict[str, Any] = None,
                                        current_reflection: Dict[str, Any] = None,
                                        operator_docs: Dict[str, str] = None,
                                        region: str = "USA") -> List[Dict]:
        """
        Build conversation messages for alpha improvement with LLM
        Supports both first round (no history) and subsequent rounds (with history and reflection)

        Args:
            expr_code: Current expression code
            source_idea: Original factor idea description
            defect_reason: Why it needs improvement (test failures)
            metrics: Current performance metrics
            strategy: Improvement strategy to apply
            target_sharpe: Target Sharpe ratio
            conversation_history: Previous conversation messages
            backtest_history: Previous backtest results
            current_settings: Current simulation settings
            current_reflection: Reflection from last round (for subsequent rounds)
            operator_docs: Detailed operator documentation from operators.json
            region: Market region for valid neutralization options

        Returns:
            List of message dicts for LLM chat
        """
        # Get valid neutralization options based on region
        region = region.upper() if region else "USA"
        if region in ['ASI', 'EUR', 'GLB']:
            valid_neutral = "COUNTRY, MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST"
        elif region in ['CHN', 'KOR', 'IND']:
            valid_neutral = "MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, CROWDING, FAST, SLOW, SLOW_AND_FAST"
        elif region == 'MEA':
            valid_neutral = "COUNTRY, MARKET, INDUSTRY, SUBINDUSTRY, SECTOR"
        else:  # USA and others
            valid_neutral = "MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST"
        # Determine if this is the first round
        is_first_round = not backtest_history or len(backtest_history) == 0
        strategy_descriptions = {
            "adjust_param": "[Parameter Tuning] Fine-tune operator internal parameters - window sizes (5/10/20/60/120/252), weight parameters, smoothing coefficients. Best for: Expressions with decent returns but high volatility that need refinement.",
            "replace_op": "[Operator Replacement] Replace with equivalent quantitative operators - ts_mean↔ts_median↔ts_weighted_mean, ts_zscore↔ts_rank↔ts_percentile, ts_corr↔ts_covariance, rank↔group_rank. Best for: Logic flaws from inappropriate operator selection. Be bold in trying different operators!",
            "restructure": "[Structure Refactoring] Restructure overall nesting structure and calculation logic - change operation order, reduce/increase nesting levels, split complex expressions, change grouping methods. Best for: Overly complex structures, overfitting, or unclear logic.",
            "transform": "[Mathematical Transformation] Transform the factor's mathematical form - convert price factor to momentum factor, cross-sectional to time-series, add lag terms, construct differences/ratios/log transforms. Best for: Fundamental changes to factor logic.",
            "combine": "[Combination Enhancement] Combine multiple complementary factors or add interaction terms - weighted combinations, conditional combinations, residual extraction, orthogonalization. Best for: Insufficient single-factor information, need multi-dimensional enhancement.",
            "auto": "[Auto Select] Automatically select the most aggressive improvement strategy based on defect analysis"
        }
        strategy_desc = strategy_descriptions.get(strategy, strategy_descriptions["auto"])

        messages = []

        # Build operator documentation string (include all operators)
        if operator_docs:
            ops_doc_str = "\n".join([f"- {op}: {doc}" for op, doc in operator_docs.items()])


        # System message
        messages.append({
            "role": "system",
            "content": f"""You are an expert quantitative researcher specializing in alpha factor optimization.
Your goal is to iteratively improve alpha expressions through conversation.

Key Principles:
1. Analyze why the current expression underperforms based on test failures
2. Maintain the original economic logic from the factor idea
3. Generate valid FASTEXPR code with proper syntax
4. Learn from previous attempts and avoid repeating failures
5. Be specific about what changes you make and why

Available Operators:
{ops_doc_str}

Always respond in JSON format with the improved expression."""
        })

        if is_first_round:
            # 第一轮：初始prompt
            # Build current settings string
            settings_str = ""
            if current_settings:
                settings_str = f"""
Current Settings:
- neutralization: {current_settings.get('neutralization', 'INDUSTRY')}
- decay: {current_settings.get('decay', 0)}
- truncation: {current_settings.get('truncation', 0.08)}
- delay: {current_settings.get('delay', 1)}
- maxTrade: {current_settings.get('maxTrade', 'OFF')}"""

            initial_content = f"""Please improve the following alpha expression.

Original Factor Idea:
{source_idea}

Current Expression:
```
{expr_code}
```

Test Failures / Issues:
{defect_reason if defect_reason else "Underperforming - needs optimization"}

Current Performance Metrics:
- Sharpe Ratio: {metrics.get('sharpe', 0):.4f}
- Annual Return: {metrics.get('returns', 0):.4f}
- Max Drawdown: {metrics.get('drawdown', 0):.4f}
- Turnover: {metrics.get('turnover', 0):.4f}
- Fitness: {metrics.get('fitness', 0):.4f}
{settings_str}

Current Settings:
- neutralization: {current_settings.get('neutralization', 'INDUSTRY') if current_settings else 'INDUSTRY'}
- decay: {current_settings.get('decay', 0) if current_settings else 0}
- truncation: {current_settings.get('truncation', 0.08) if current_settings else 0.08}

Target:
- Sharpe Ratio: > {target_sharpe}
- Max Drawdown: < 0.15

Suggested Strategy: {strategy_desc}

You can also suggest settings adjustments (neutralization, decay, truncation, delay, maxTrade) if they would improve performance.

Your Task:
1. Analyze the test failures and understand why the expression underperforms
2. Apply the suggested strategy or choose a better one
3. Generate an improved FASTEXPR expression
4. Explain your changes and expected improvement

Requirements:
1. Maintain the core economic logic from the original idea
2. Address the specific test failures mentioned
3. Ensure syntactic validity (balanced parentheses, valid operators)
4. Use appropriate time windows (5, 10, 20, 60, 120, 252)
5. Consider adding neutralization (group_neutral, group_rank) if needed

Settings Configuration:
You can also suggest improvements to the simulation settings:

REGION: {region}

VALID NEUTRALIZATION OPTIONS for {region}:
{valid_neutral}

OTHER SETTINGS:
- decay: 0-512 (smoothing factor, default: 0)
- truncation: 0-0.1 (weight truncation, default: 0.08)
- delay: 1 (data delay, default: 1)
- maxTrade: ON or OFF (max trade constraint, default: OFF)

Only suggest settings changes if you believe they will meaningfully improve performance.

Output Format (JSON):
{{
    "improved_expressions": [
        "first_improved_fastexpr_code_here",
        "second_improved_fastexpr_code_here (optional)",
        "... more variants (optional)"
    ],
    "improved_settings": {{
        "neutralization": "INDUSTRY",
        "decay": 3,
        "truncation": 0.06
    }},
    "changes_made": "Specific description of what you changed in expressions and settings",
    "expected_improvement": "Why this should address the test failures",
    "confidence": 0.8
}}

Notes:
- improved_expressions: Array of expression variants, at least one required, no upper limit
- improved_settings: Single settings configuration (optional)
- All expressions will use the same settings configuration
- Generate multiple variants to increase chances of finding a valid expression"""
            messages.append({"role": "user", "content": initial_content})
        else:
            # 后续轮次：包含完整改进历史、反思和维度分析
            # Build history with settings
            history_str = ""
            for i, h in enumerate(backtest_history, 1):
                hist_metrics = h.get('metrics', {})
                hist_settings = h.get('settings', {})
                settings_part = ""
                if hist_settings:
                    settings_part = f"""
- Settings: decay={hist_settings.get('decay', 0)}, neutralization={hist_settings.get('neutralization', 'INDUSTRY')}"""
                history_str += f"""
Round {i}:
- Expression: {h.get('expression_code', h.get('expression', 'N/A'))}{settings_part}
- Sharpe: {hist_metrics.get('sharpe', 0):.4f}
- Turnover: {hist_metrics.get('turnover', 0):.4f}
- Changes: {h.get('changes_made', 'N/A')}
"""

            # 计算改进趋势
            initial_sharpe = backtest_history[0].get('metrics', {}).get('sharpe', 0) if backtest_history else 0
            last_sharpe = backtest_history[-1].get('metrics', {}).get('sharpe', 0) if backtest_history else 0
            trend = "improving" if last_sharpe > initial_sharpe else "declining" if last_sharpe < initial_sharpe else "stable"

            # 反思结果（如果提供）
            reflection = current_reflection.get('reflection', 'No reflection provided') if current_reflection else 'No reflection provided'
            what_worked = ', '.join(current_reflection.get('what_worked', [])) if current_reflection else 'N/A'
            what_failed = ', '.join(current_reflection.get('what_failed', [])) if current_reflection else 'N/A'
            key_learnings = ', '.join(current_reflection.get('key_learnings', [])) if current_reflection else 'N/A'
            next_time = current_reflection.get('next_time', '') if current_reflection else ''
            loop_recovery_warning = current_reflection.get('loop_recovery_warning', '') if current_reflection else ''

            # 维度分析
            dim_analysis = current_reflection.get('dimension_analysis', {}) if current_reflection else {}
            econ_logic = dim_analysis.get('economic_logic', {})
            trading_cost = dim_analysis.get('trading_cost', {})
            overfitting = dim_analysis.get('overfitting_risk', {})
            param_sens = dim_analysis.get('parameter_sensitivity', {})

            feedback_content = f"""Please improve the alpha expression based on the following comprehensive analysis.

Original Factor Idea:
{source_idea}

Complete Improvement History ({len(backtest_history)} rounds):
{history_str}

Current Status:
- Initial Sharpe: {initial_sharpe:.4f}
- Latest Sharpe: {last_sharpe:.4f}
- Trend: {trend}
- Target: {target_sharpe}

Current Expression:
```
{expr_code}
```

=== REFLECTION FROM LAST ROUND ===
Overall Reflection: {reflection}
What Worked: {what_worked}
What Failed: {what_failed}
Key Learnings: {key_learnings}
Next Time: {next_time}
{f"🚨 LOOP RECOVERY WARNING: {loop_recovery_warning}\\n" if loop_recovery_warning else ""}===================================

--- Dimensional Analysis ---
Economic Logic Consistency:
- Consistent: {econ_logic.get('consistent', 'N/A')}
- Deviation: {econ_logic.get('deviation', 'N/A')}
- Justified: {econ_logic.get('justified', 'N/A')}

Trading Cost Impact:
- Turnover Concern: {trading_cost.get('turnover_concern', 'N/A')}
- Cost Impact: {trading_cost.get('cost_impact', 'N/A')}
- Practical: {trading_cost.get('practical', 'N/A')}

Overfitting Risk:
- Risk Level: {overfitting.get('risk_level', 'N/A')}
- Concerns: {', '.join(overfitting.get('concerns', []))}
- Generalizable: {overfitting.get('generalizable', 'N/A')}

Parameter Sensitivity:
- Robust: {param_sens.get('robust', 'N/A')}
- Sensitive Params: {', '.join(param_sens.get('sensitive_params', []))}
- Recommendation: {param_sens.get('recommendation', 'N/A')}
===================================

Current Settings:
- neutralization: {current_settings.get('neutralization', 'INDUSTRY') if current_settings else 'INDUSTRY'}
- decay: {current_settings.get('decay', 0) if current_settings else 0}
- truncation: {current_settings.get('truncation', 0.08) if current_settings else 0.08}

Target:
- Sharpe Ratio: > {target_sharpe}
- Max Drawdown: < 0.15

Suggested Strategy: {strategy_desc}

Your Task:
1. Review the complete improvement history above
2. Consider the reflection and dimensional analysis from the last round
3. Identify patterns: what consistently works vs what consistently fails
4. Address specific concerns from dimensional analysis:
   - Ensure economic logic consistency
   - Control trading costs (keep turnover reasonable)
   - Avoid overfitting (use standard windows, avoid excessive complexity)
   - Use robust, non-sensitive parameters
5. Propose an improved expression that applies all learnings
6. Explain how this new attempt addresses previous shortcomings

Requirements:
1. Maintain the core economic logic from the original idea
2. Build on successful elements from previous rounds
3. Avoid strategies that have consistently failed
4. Ensure syntactic validity
5. Use standard time windows (5, 10, 20, 60, 120, 252) - avoid arbitrary numbers
6. Keep turnover reasonable (< 0.5) for practical trading
7. Avoid excessive nesting and complexity
8. Ensure the expression would generalize to out-of-sample data

Settings Configuration (optional):

REGION: {region}

VALID NEUTRALIZATION OPTIONS for {region}:
{valid_neutral}

OTHER SETTINGS:
- decay: 0-512 (smoothing factor)
- truncation: 0-1 (weight truncation)
- delay: 0 or 1 (data delay)
- maxTrade: ON or OFF

Output Format (JSON):
{{
    "improved_expressions": [
        "first_improved_fastexpr_code_here",
        "second_improved_fastexpr_code_here (optional)",
        "... more variants (optional)"
    ],
    "improved_settings": {{
        "neutralization": "INDUSTRY",
        "decay": 3,
        "truncation": 0.06
    }},
    "changes_made": "Specific changes to expressions and settings",
    "expected_improvement": "Expected impact",
    "confidence": 0.8
}}

Notes:
- improved_expressions: Array of expression variants, at least one required, no upper limit
- improved_settings: Single settings configuration (optional)
- All expressions will use the same settings configuration
- Generate multiple variants to increase chances of finding a valid expression"""
            messages.append({"role": "user", "content": feedback_content})

        return messages

    @staticmethod
    def alpha_round_reflection(expr_code: str, backtest_result: Dict[str, Any],
                                previous_changes: str, expected_improvement: str) -> str:
        """
        单独对当前轮次的回测结果进行反思（不考虑历史）

        Args:
            expr_code: 当前轮次测试的表达式
            backtest_result: 回测结果（包含所有指标和失败测试）
            previous_changes: 上一轮做的修改
            expected_improvement: 上一轮预期的改进

        Returns:
            反思prompt
        """
        metrics = backtest_result.get('metrics', {})
        is_tests = backtest_result.get('is_tests', {})

        # 构建指标字符串
        metrics_str = f"""- Sharpe Ratio: {metrics.get('sharpe', 0):.4f}
- Annual Return: {metrics.get('returns', 0):.4f}
- Max Drawdown: {metrics.get('drawdown', 0):.4f}
- Turnover: {metrics.get('turnover', 0):.4f}
- Fitness: {metrics.get('fitness', 0):.4f}
- Margin: {metrics.get('margin', 0):.6f}
- Long Count: {metrics.get('long_count', 0)}
- Short Count: {metrics.get('short_count', 0)}"""

        # 构建失败测试字符串
        failed_tests_str = ""
        if is_tests:
            names = is_tests.get('name', {})
            results = is_tests.get('result', {})
            limits = is_tests.get('limit', {})
            values = is_tests.get('value', {})

            failed_tests = []
            for idx, name in names.items():
                if results.get(idx) == 'FAIL':
                    failed_tests.append(
                        f"- {name}: {values.get(idx)} (limit: {limits.get(idx)})"
                    )
            failed_tests_str = "\n".join(failed_tests) if failed_tests else "No failed tests"

        return f"""You are evaluating the results of a single improvement attempt. Focus ONLY on this round's results.

Expression Tested:
```
{expr_code}
```

Changes Made in This Round:
{previous_changes if previous_changes else "Initial expression"}

Expected Improvement:
{expected_improvement if expected_improvement else "N/A"}

Backtest Results:
{metrics_str}

Failed Tests:
{failed_tests_str}

Your Task - Reflect on THIS ROUND ONLY:

**1. Performance Analysis**
- Did the changes produce the expected results? Why or why not?
- Which specific metrics improved or deteriorated?

**2. Economic Logic Consistency**
- Do the modifications maintain the original economic/financial logic?
- Are the changes economically interpretable and sound?
- Did we deviate from the core factor idea? If so, was it justified?

**3. Trading Cost Impact**
- Is the turnover rate practical for real trading?
- Consider transaction costs: high turnover (>0.5) may erode profits
- Is the margin sufficient to cover trading costs?

**4. Overfitting Risk Assessment**
- Do the parameters look overly optimized for historical data?
- Are the time windows reasonable (not too specific like 17 days)?
- Is the expression too complex with excessive nesting?
- Would this likely generalize to out-of-sample data?

**5. Parameter Sensitivity**
- Are the current parameters robust or overly sensitive?
- Would small changes in windows/operators significantly alter results?
- Are we using standard windows (5, 10, 20, 60, 120, 252) or arbitrary ones?

**6. Synthesis**
- What worked well in this attempt?
- What went wrong or underperformed?
- What did you learn from this specific round?
- What would you do differently next time?

Provide a comprehensive reflection covering all dimensions above.

Output Format (JSON):
{{
    "reflection": "Overall assessment of this round",
    "expected_vs_actual": "Comparison of expected vs actual results",
    "what_worked": ["specific element 1", "specific element 2"],
    "what_failed": ["specific issue 1", "specific issue 2"],
    "key_learnings": ["learning 1", "learning 2"],
    "next_time": "What to do differently in the next attempt",
    "success_rating": 0.7,
    "dimension_analysis": {{
        "economic_logic": {{
            "consistent": true,
            "deviation": "description if any",
            "justified": true
        }},
        "trading_cost": {{
            "turnover_concern": "low/medium/high",
            "cost_impact": "description",
            "practical": true
        }},
        "overfitting_risk": {{
            "risk_level": "low/medium/high",
            "concerns": ["concern 1", "concern 2"],
            "generalizable": true
        }},
        "parameter_sensitivity": {{
            "robust": true,
            "sensitive_params": ["param 1", "param 2"],
            "recommendation": "use standard windows or adjust"
        }}
    }}
}}"""

    @staticmethod
    def reflection_with_full_data(
        current_code: str,
        current_metrics: Dict[str, Any],
        variants_data: List[Dict],
        improvement_history: List[Dict]
    ) -> List[Dict]:
        """
        使用完整回测数据进行反思，分析多个表达式变体的表现

        Args:
            current_code: 当前表达式代码
            current_metrics: 当前指标
            variants_data: 变体数据列表，每个包含 expression_id, expression_code, variant_index, expected_improvement, full_backtest_result
            improvement_history: 改进历史

        Returns:
            LLM 消息列表
        """
        # 构建变体数据字符串
        variants_str = ""
        for i, variant in enumerate(variants_data, 1):
            result = variant.get("full_backtest_result", {})
            is_stats = result.get("is_stats", {})

            def get_val(key):
                v = is_stats.get(key)
                return float(v.get("0", 0)) if isinstance(v, dict) else float(v) if v else 0.0

            sharpe = get_val("sharpe")
            returns = get_val("returns")
            drawdown = get_val("drawdown")
            turnover = get_val("turnover")
            fitness = get_val("fitness")
            margin = get_val("margin")

            variants_str += f"""
Variant {i} (ID: {variant.get('expression_id', 'unknown')}):
```
{variant.get('expression_code', '')}
```
Expected Improvement: {variant.get('expected_improvement', 'N/A')}
Results:
- Sharpe: {sharpe:.4f}
- Returns: {returns:.4f}
- Drawdown: {drawdown:.4f}
- Turnover: {turnover:.4f}
- Fitness: {fitness:.4f}
- Margin: {margin:.6f}
"""

        # 构建历史字符串
        history_str = ""
        if improvement_history:
            for h in improvement_history:
                round_num = h.get("round", 0)
                changes = h.get("changes_made", "")
                reflection = h.get("reflection", "")

                # 新格式：每轮可能有多个 alpha
                if "alphas" in h and h["alphas"]:
                    for alpha in h["alphas"]:
                        metrics = alpha.get("metrics", {})
                        history_str += f"""
Round {round_num}:
- Expression: {alpha.get('expression_code', '')[:100]}...
- Sharpe: {metrics.get('sharpe', 0):.4f}
- Changes: {changes}
- Reflection: {reflection[:200] if reflection else 'N/A'}
"""
                else:
                    # 旧格式兼容
                    metrics = h.get("metrics", {})
                    history_str += f"""
Round {round_num}:
- Sharpe: {metrics.get('sharpe', 0):.4f}
- Changes: {changes}
"""
        else:
            history_str = "No previous rounds."

        system_msg = """You are an expert quantitative researcher analyzing backtest results of multiple alpha factor variants.
Your task is to deeply analyze why some variants succeeded while others failed, and provide actionable insights for the next improvement round."""

        user_msg = f"""Please analyze the backtest results of multiple expression variants and provide comprehensive reflection.

Current Expression:
```
{current_code}
```

Current Metrics:
- Sharpe: {current_metrics.get('sharpe', 0):.4f}
- Returns: {current_metrics.get('returns', 0):.4f}
- Drawdown: {current_metrics.get('drawdown', 0):.4f}
- Turnover: {current_metrics.get('turnover', 0):.4f}

Tested Variants:
{variants_str}

Improvement History:
{history_str}

Your Task - Comprehensive Analysis:

**1. Variant Performance Comparison**
- Which variants performed best and why?
- What patterns distinguish successful from unsuccessful variants?
- Did the expected improvements match actual results?

**2. Structural Analysis**
- What structural differences exist between variants?
- Which operators/time windows contributed to success?
- Are there common failure patterns?

**3. Economic Logic Assessment**
- Do the successful variants maintain economic interpretability?
- Did any variant deviate too far from the original idea?
- Is the factor logic sound across variants?

**4. Trading Characteristics**
- Compare turnover, drawdown, and margin across variants
- Which variants are most practical for real trading?
- Are there risk-adjusted return differences?

**5. Overfitting Indicators**
- Do any variants show signs of overfitting?
- Are parameters robust or overly specific?
- Which variants would likely generalize best?

**6. Strategic Recommendations**
- What specific changes should be made next?
- Which direction shows most promise?
- What should be avoided in the next iteration?

Output Format (JSON):
{{
    "summary": "Overall assessment of variant testing",
    "best_variant": {{
        "index": 1,
        "reason": "Why this variant performed best"
    }},
    "performance_analysis": {{
        "successful_patterns": ["pattern1", "pattern2"],
        "failure_patterns": ["failure1", "failure2"],
        "unexpected_results": "description"
    }},
    "structural_insights": {{
        "key_differences": ["difference1", "difference2"],
        "effective_operators": ["operator1", "operator2"],
        "problematic_elements": ["element1", "element2"]
    }},
    "economic_logic": {{
        "maintained": true,
        "deviations": ["deviation1"],
        "interpretability": "high/medium/low"
    }},
    "trading_viability": {{
        "most_practical": "variant description",
        "cost_considerations": "analysis",
        "risk_assessment": "analysis"
    }},
    "overfitting_assessment": {{
        "risk_level": "low/medium/high",
        "robust_variants": [1, 2],
        "concerns": ["concern1", "concern2"]
    }},
    "next_round_strategy": {{
        "recommended_direction": "specific strategy",
        "operators_to_try": ["op1", "op2"],
        "operators_to_avoid": ["op3"],
        "parameter_adjustments": "suggestions",
        "expected_improvement": "what to expect"
    }},
    "confidence": 0.8
}}"""

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

    @staticmethod
    def alpha_overall_reflection(improvement_history: List[Dict],
                                  source_idea: str,
                                  initial_sharpe: float,
                                  final_sharpe: float) -> str:
        """
        在所有轮次结束后进行整体反思

        Args:
            improvement_history: 改进历史摘要
            source_idea: 原始factor idea
            initial_sharpe: 初始sharpe
            final_sharpe: 最终sharpe

        Returns:
            整体反思prompt
        """
        # 构建历史字符串
        history_str = ""
        for h in improvement_history:
            history_str += f"""
Round {h['round']}:
- Sharpe: {h['sharpe']:.4f}
- Return: {h['returns']:.4f}
- Drawdown: {h['drawdown']:.4f}
- Turnover: {h['turnover']:.4f}
- Fitness: {h['fitness']:.4f}
- Changes: {h['changes']}
- Expected: {h['expected']}
"""

        improvement_pct = ((final_sharpe - initial_sharpe) / abs(initial_sharpe) * 100) if initial_sharpe != 0 else 0
        trend = "improved" if final_sharpe > initial_sharpe else "declined" if final_sharpe < initial_sharpe else "unchanged"

        return f"""You are reviewing the complete improvement process of an alpha factor. Provide a comprehensive overall reflection.

Original Factor Idea:
{source_idea}

Complete Improvement History:
{history_str}

Summary Statistics:
- Initial Sharpe: {initial_sharpe:.4f}
- Final Sharpe: {final_sharpe:.4f}
- Improvement: {improvement_pct:.1f}%
- Overall Trend: {trend}
- Total Rounds: {len(improvement_history)}

Your Task - Provide Overall Reflection:

**1. Process Effectiveness**
- Did the iterative improvement process work well?
- Which rounds contributed most to the improvement?
- Were there any rounds that were counterproductive?

**2. Strategy Analysis**
- What types of modifications consistently worked?
- What strategies repeatedly failed?
- Is there a pattern in successful vs unsuccessful attempts?

**3. Economic Logic Preservation**
- Did the final expression maintain the original economic logic?
- Did we drift away from the core idea during iterations?
- Is the final expression economically interpretable?

**4. Practical Viability**
- Is the final alpha practically tradable?
- Are turnover and drawdown at acceptable levels?
- Would this survive transaction costs in real trading?

**5. Overfitting Assessment**
- Does the improvement trajectory suggest overfitting?
- Are the parameters robust or overly optimized?
- Would this generalize to out-of-sample periods?

**6. Key Insights & Learnings**
- What are the most important lessons from this process?
- What would you do differently if starting over?
- What does this teach us about improving this type of factor?

**7. Final Assessment**
- Is the final result satisfactory?
- Should we continue improving or is this good enough?
- What are the remaining weaknesses, if any?

Output Format (JSON):
{{
    "overall_assessment": "Summary of the entire improvement process",
    "process_effectiveness": {{
        "worked_well": true,
        "key_contributing_rounds": [1, 3],
        "counterproductive_rounds": [],
        "effectiveness_rating": 0.8
    }},
    "strategy_analysis": {{
        "successful_strategies": ["strategy1", "strategy2"],
        "failed_strategies": ["strategy3"],
        "pattern": "description of patterns observed"
    }},
    "economic_logic": {{
        "preserved": true,
        "drift": "description if any",
        "interpretable": true
    }},
    "practical_viability": {{
        "tradable": true,
        "turnover_acceptable": true,
        "drawdown_acceptable": true,
        "survives_costs": true
    }},
    "overfitting_assessment": {{
        "risk_level": "low/medium/high",
        "robust": true,
        "generalizable": true,
        "concerns": ["concern1", "concern2"]
    }},
    "key_insights": [
        "Key learning 1",
        "Key learning 2",
        "Key learning 3"
    ],
    "would_do_differently": "What to change if starting over",
    "final_assessment": {{
        "satisfactory": true,
        "continue_improving": false,
        "reason": "explanation",
        "remaining_weaknesses": ["weakness1", "weakness2"]
    }},
    "recommendation": "GOOD_ENOUGH / CONTINUE / ABANDON / RESTART"
}}"""

    @staticmethod
    def validation_error_correction(
        original_code: str,
        invalid_expression: str,
        invalid_settings: Dict,
        validation_errors: List[str],
        valid_fields: set,
        region: str,
        source_idea: str,
        target_sharpe: float = 1.25
    ) -> List[Dict]:
        """
        Build conversation for correcting validation errors

        Args:
            original_code: Original expression code
            invalid_expression: The expression that failed validation
            invalid_settings: The settings that failed validation
            validation_errors: List of validation error messages
            valid_fields: Allowed data fields (original fields + exempt fields)
            region: Market region
            source_idea: Original factor idea
            target_sharpe: Target sharpe ratio

        Returns:
            List of message dicts for LLM chat
        """
        errors_str = "\n".join([f"- {err}" for err in validation_errors])
        fields_str = ", ".join(sorted(valid_fields)) if valid_fields else "None"
        settings_str = json.dumps(invalid_settings, indent=2) if invalid_settings else "{}"

        # Get valid neutralization for region
        region = region.upper() if region else "USA"
        if region in ['ASI', 'EUR', 'GLB']:
            valid_neutral = "COUNTRY, MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST"
        elif region in ['CHN', 'KOR', 'IND']:
            valid_neutral = "MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, CROWDING, FAST, SLOW, SLOW_AND_FAST"
        elif region == 'MEA':
            valid_neutral = "COUNTRY, MARKET, INDUSTRY, SUBINDUSTRY, SECTOR"
        else:  # USA and others
            valid_neutral = "MARKET, INDUSTRY, SUBINDUSTRY, SECTOR, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, SLOW_AND_FAST"

        system_msg = """You are an expert quantitative researcher specializing in alpha factor optimization.
Your previous improvement attempt failed validation. Please correct ALL the errors and generate a valid expression.

Key Rules:
1. ONLY use data fields from the allowed list
2. ONLY use valid operators (ts_mean, ts_std, rank, group_neutralize, etc.)
3. Settings must be within valid ranges for the region
4. Maintain the original economic logic
5. Ensure proper syntax (balanced parentheses)
6. Special values (NaN) MUST use double quotes"""

        user_msg = f"""Your previous improvement failed validation. Please correct ALL the errors below.

Original Factor Idea:
{source_idea}

Original Expression:
```
{original_code}
```

Your Invalid Expression:
```
{invalid_expression}
```

Your Invalid Settings:
{settings_str}

VALIDATION ERRORS (Fix ALL of these):
{errors_str}

---

ALLOWED DATA FIELDS (You can ONLY use these):
{fields_str}

REGION: {region}

VALID NEUTRALIZATION OPTIONS for {region}:
{valid_neutral}

VALID SETTINGS RANGES:
- neutralization: See above options
- decay: 0 <= decay <= 512
- truncation: 0 < truncation < 1
- delay: 0 or 1
- maxTrade: ON or OFF

SPECIAL VALUES (MUST use double quotes):
- "NaN"

Your Task:
1. Fix ALL validation errors listed above
2. Generate a syntactically valid expression using ONLY allowed fields
3. Provide corrected settings within valid ranges
4. Maintain the original economic logic

Output Format (JSON):
{{
    "improved_expressions": [
        "your_corrected_expression_here",
        "alternative_expression_here (optional)",
        "... more variants (optional)"
    ],
    "improved_settings": {{
        "neutralization": "INDUSTRY",
        "decay": 5,
        "truncation": 0.08
    }},
    "changes_made": "Description of corrections made to fix validation errors",
    "expected_improvement": "Why this corrected version should work",
    "confidence": 0.8
}}

Notes:
- improved_expressions: Array of corrected expression variants, at least one required, no upper limit
- improved_settings: Single settings configuration (optional)
- All expressions will use the same settings configuration
- Generate multiple variants to increase chances of finding a valid expression"""

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
