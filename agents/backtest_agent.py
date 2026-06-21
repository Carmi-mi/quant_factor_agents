"""Backtest Agent
Performs backtesting on expressions and classifies results using LLM analysis
"""
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from core.agent_base import Agent
from models.backtest_result import BacktestResult, BacktestMetrics
from services.llm_service import LLMService
from infrastructure.prompt_templates import PromptTemplates



class BacktestAgent(Agent):
    """Backtest execution agent with LLM analysis"""
    
    def __init__(self, config: Dict[str, Any], llm_service: LLMService = None):
        """
        Initialize
        
        Args:
            config: Configuration dictionary
            llm_service: LLM service for result analysis
        """
        super().__init__("BacktestAgent", config)
        self.threshold = config.get("backtest", {}).get("threshold", {
            "good": {"min_sharpe": 1.0, "min_return": 0.05, "max_drawdown": 0.15},
            "bad": {"max_sharpe": 0.3},
            "improve": {"min_sharpe": 0.3, "max_sharpe": 1.0}
        })
        self.llm = llm_service
        self.use_llm = llm_service is not None
        
        # BRAIN API credentials
        self.brain_credentials = config.get("brain", {})
        self.email = self.brain_credentials.get("email")
        self.password = self.brain_credentials.get("password")
        
        # Session will be initialized on first use
        self.session = None
        
        # Backtest configuration
        self.backtest_config = config.get("backtest", {}).get("settings", {
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "test_period": "P0Y0M0D",
            "nan_handling": "ON",
            "max_trade": "OFF"
        })
        
        # Filter threshold from config
        self.filter_threshold = config.get("backtest", {}).get("filter_threshold", {})
        self.min_sharpe = self.filter_threshold.get("min_sharpe", 0.8)
    
    def run(self, idea: Dict[str, Any], file_info: Dict[str, Any] = None,
            settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run backtest on a single idea

        Args:
            idea: Single idea with idea_id, idea_content, expressions
                {"idea_id": str, "idea_content": str, "expressions": [...]}
            file_info: Optional file information
            settings: Optional partial settings to override default backtest config
                e.g., {"decay": 3, "neutralization": "SUBINDUSTRY"}

        Returns:
            Backtest result for the single idea with classified expressions
        """
        # Validate input
        if not idea or "expressions" not in idea:
            raise ValueError("Invalid idea: must contain 'expressions' field")

        idea_id = idea.get("idea_id", "unknown")
        idea_content = idea.get("idea_content", "")
        expressions = idea.get("expressions", [])

        print(f"[BacktestAgent] Processing idea '{idea_id}' with {len(expressions)} expressions")
        print(f"[BacktestAgent] Using LLM analysis: {self.use_llm}")

        # Merge partial settings with default config if provided
        if settings:
            merged_config = self.backtest_config.copy()
            merged_config.update(settings)
            print(f"[BacktestAgent] Using custom settings: {settings}")
        else:
            merged_config = self.backtest_config

        # Build expression objects for this idea
        idea_expressions = []
        for idx, expr_code in enumerate(expressions):
            expr_obj = {
                "code": expr_code,
                "id": f"{idea_id}_expr_{idx}",
                "idea_id": idea_id,
                "source_idea": idea_content
            }
            idea_expressions.append(expr_obj)

        # Backtest expressions using BRAIN API
        print(f"[BacktestAgent] Using BRAIN API batch backtest for idea '{idea_id}'")
        raw_results = self._batch_backtest(idea_expressions, merged_config)

        # Process raw results and extract structured data
        processed_result = self._process_raw_results(raw_results, idea)

        return processed_result
    
    def _init_session(self) -> bool:
        """Initialize BRAIN API session"""
        if self.session is not None:
            return True
        
        if not self.email or not self.password:
            print("[BacktestAgent] BRAIN credentials not configured")
            return False
        
        try:
            print(f"[BacktestAgent] Logging into BRAIN API with email: {self.email}")
            # Set credentials as environment variables for start_session()
            import os
            os.environ["BRAIN_CREDENTIAL_EMAIL"] = self.email
            os.environ["BRAIN_CREDENTIAL_PASSWORD"] = self.password
            self.session = start_session()
            print("[BacktestAgent] BRAIN API login successful")
            return True
        except Exception as e:
            print(f"[BacktestAgent] BRAIN API login failed: {e}")
            return False

    def _llm_classify(self, result: BacktestResult, expr_code: str = "") -> Dict[str, Any]:
        """Use LLM to classify and analyze backtest results"""
        try:
            # Generate prompt
            prompt = PromptTemplates.backtest_analysis(
                expr_code,
                {
                    "sharpe": result.metrics.sharpe,
                    "annual_return": result.metrics.return_pct,
                    "max_drawdown": result.metrics.max_drawdown,
                    "volatility": result.metrics.volatility,
                    "turnover": getattr(result.metrics, 'turnover', 0.5),
                    "fitness": getattr(result.metrics, 'fitness', result.metrics.sharpe)
                }
            )
            
            # Call LLM
            response = self.llm.client.complete(prompt)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return {
                    "classification": analysis.get("classification", "need_improve"),
                    "confidence": analysis.get("confidence", 0.5),
                    "analysis": analysis.get("analysis", {}),
                    "improvement_suggestions": analysis.get("improvement_suggestions", []),
                    "defect_reason": analysis.get("defect_reason", "")
                }
            else:
                raise ValueError("LLM response does not contain valid JSON")
                
        except Exception as e:
            print(f"[BacktestAgent] LLM classification failed: {e}")
            # Fallback to rule-based
            result.classify(self.threshold)
            return {
                "classification": result.status,
                "confidence": 0.5,
                "analysis": {},
                "improvement_suggestions": [],
                "defect_reason": result.defect_reason
            }
    
    def _process_raw_results(self, raw_results: Any, idea: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw backtest results from BRAIN API and extract structured data
        
        Args:
            raw_results: Raw results from simulate_alpha_list_multi
            idea: Original idea dict with idea_id and idea_content
            
        Returns:
            Processed result with extracted alphas data
        """
        idea_id = idea.get("idea_id", "unknown")
        idea_content = idea.get("idea_content", "")
        
        # Check for errors
        if isinstance(raw_results, dict) and raw_results.get("error"):
            print(f"[BacktestAgent] Idea '{idea_id}' has error: {raw_results.get('error_type', 'Unknown')}")
            return {
                "idea_id": idea_id,
                "idea_content": idea_content,
                "alphas": [],
                "total_count": 0,
                "error": raw_results.get("error"),
                "error_type": raw_results.get("error_type", "Unknown")
            }
        
        # Handle empty results
        if not raw_results:
            return {
                "idea_id": idea_id,
                "idea_content": idea_content,
                "alphas": [],
                "total_count": 0
            }
        
        # Process each result
        processed_alphas = []
        
        for result in raw_results:
            # Extract is_stats
            is_stats = result.get("is_stats", {})
            
            # Get sharpe value for filtering
            sharpe = self._extract_stat_value(is_stats, "sharpe")
            
            # Skip if sharpe is invalid
            if not isinstance(sharpe, (int, float)):
                continue
            
            # Get simulate_data for expression and settings
            simulate_data = result.get("simulate_data", {})
            settings = simulate_data.get("settings", {})
            
            # Process is_tests
            is_tests = result.get("is_tests", {})
            if is_tests is not None and not (hasattr(is_tests, 'empty') and is_tests.empty):
                # Handle DataFrame/Series
                if hasattr(is_tests, 'to_dict'):
                    is_tests_dict = is_tests.to_dict()
                else:
                    is_tests_dict = is_tests
                
                # Keep only necessary fields
                is_tests = {
                    "name": is_tests_dict.get("name") if isinstance(is_tests_dict, dict) else None,
                    "result": is_tests_dict.get("result") if isinstance(is_tests_dict, dict) else None,
                    "limit": is_tests_dict.get("limit") if isinstance(is_tests_dict, dict) else None,
                    "value": is_tests_dict.get("value") if isinstance(is_tests_dict, dict) else None,
                    "ratio": is_tests_dict.get("ratio") if isinstance(is_tests_dict, dict) else None,
                    "effective": is_tests_dict.get("effective") if isinstance(is_tests_dict, dict) else None
                }
            
            processed_alphas.append({
                "alpha_id": result.get("alpha_id"),
                "expression": simulate_data.get("regular"),
                "settings": settings,
                "is_stats": is_stats,
                "stats": result.get("stats"),
                "is_tests": is_tests,
                "sharpe": sharpe
            })
        
        # Filter qualified alphas (sharpe > threshold)
        qualified_alphas = [
            alpha for alpha in processed_alphas
            if alpha.get("sharpe", 0)**2 > self.min_sharpe**2 
        ]
        
        print(f"[BacktestAgent] Idea '{idea_id}' processed {len(processed_alphas)} alphas, "
              f"qualified {len(qualified_alphas)} (sharpe > {self.min_sharpe})")
        
        return {
            "idea_id": idea_id,
            "idea_content": idea_content,
            "alphas": qualified_alphas,  # 只保留筛选后的
            "total_count": len(processed_alphas),
            "qualified_count": len(qualified_alphas)
        }
    
    def _extract_stat_value(self, is_stats: Dict, key: str) -> Any:
        """Extract value from is_stats (supports dict and DataFrame/Series)"""
        if not hasattr(is_stats, 'get'):
            return None
        
        data = is_stats.get(key, {})
        if hasattr(data, 'iloc'):  # pandas Series
            return float(data.iloc[0]) if len(data) > 0 else None
        elif isinstance(data, dict):
            return data.get("0")
        return data
    
    def _convert_settings_to_snake_case(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert camelCase settings keys to snake_case for generate_alpha()
        Only includes keys that generate_alpha() accepts as parameters.
        
        Args:
            settings: Settings dict with camelCase keys (e.g., from backtest results)
            
        Returns:
            Settings dict with snake_case keys
        """
        # Mapping for keys that generate_alpha() accepts as parameters
        camel_to_snake = {
            "unitHandling": "unit_handling",
            "nanHandling": "nan_handling",
            "maxTrade": "max_trade",
            "testPeriod": "test_period",
            "selectionHandling": "selection_handling",
            "selectionLimit": "selection_limit",
        }
        
        # Keys that generate_alpha() accepts (already snake_case or no change needed)
        valid_keys = {
            "region", "universe", "delay", "decay", "neutralization",
            "truncation", "pasteurization", "test_period", "unit_handling",
            "nan_handling", "max_trade", "selection_handling", "selection_limit",
            "visualization"
        }
        
        result = {}
        for key, value in settings.items():
            # Convert camelCase to snake_case if mapping exists
            snake_key = camel_to_snake.get(key, key)
            # Only include if it's a valid key for generate_alpha()
            if snake_key in valid_keys:
                result[snake_key] = value
        
        return result
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, bool)):
            return obj
        elif isinstance(obj, float):
            # Convert NaN, Inf, -Inf to None
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, 'to_json'):  # Pandas DataFrame/Series - use to_json for better handling
            try:
                return json.loads(obj.to_json(date_format='iso'))
            except:
                # Fallback to to_dict if to_json fails
                return self._make_serializable(obj.to_dict())
        elif hasattr(obj, 'to_dict'):  # Pandas DataFrame/Series fallback
            return self._make_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__'):  # Custom objects
            return self._make_serializable(obj.__dict__)
        else:
            # Convert to string as fallback
            return str(obj)
    
    def _batch_backtest(self, expressions: List[Dict[str, Any]],
                        backtest_config: Dict[str, Any] = None) -> Any:
        """
        Batch backtest multiple expressions using BRAIN API

        Args:
            expressions: List of expression dicts with 'code' and 'id' keys
            backtest_config: Optional backtest configuration to use (defaults to self.backtest_config)

        Returns:
            Raw result from simulate_alpha_list_multi
        """
        # Initialize session if needed
        if not self._init_session():
            print("[BacktestAgent] Session not available, cannot perform batch backtest")
            return {"error": "BRAIN API session not available"}

        if not expressions:
            return []

        # Use provided config or default
        config = backtest_config if backtest_config else self.backtest_config

        # Convert camelCase keys to snake_case for generate_alpha()
        config = self._convert_settings_to_snake_case(config)

        print(f"[BacktestAgent] Starting batch backtest for {len(expressions)} expressions")

        # Generate alpha configurations for all expressions
        alpha_configs = []

        for idx, expr in enumerate(expressions):
            expr_code = expr.get("code", "")

            try:
                alpha_config = generate_alpha(
                    regular=expr_code,
                    alpha_type="REGULAR",
                    **config
                )
                alpha_configs.append(alpha_config)
            except Exception as e:
                print(f"[BacktestAgent] Failed to generate alpha config for expr_{idx}: {e}")
        
        if not alpha_configs:
            print("[BacktestAgent] No valid alpha configurations generated")
            return {"error": "Failed to generate alpha configuration"}
        
        # Retry mechanism for RuntimeError
        max_retries = 2
        retry_delay = 20 * 60  # 20 minutes in seconds
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Run batch backtest and return raw result directly
                results = simulate_alpha_list_multi(
                    self.session,
                    alpha_configs,
                    limit_of_concurrent_simulations=1,
                    limit_of_multi_simulations=5,
                    simulation_config=DEFAULT_CONFIG
                )
                return results
                
            except ValueError as e:
                # Expression error - don't retry
                print(f"[BacktestAgent] Batch expression error: {e}")
                return {"error": str(e), "error_type": "ValueError"}
                
            except RuntimeError as e:
                # RuntimeError - retry after 20 minutes
                retry_count += 1
                if retry_count < max_retries:
                    print(f"[BacktestAgent] Runtime error occurred: {e}")
                    print(f"[BacktestAgent] Retrying in 20 minutes... (attempt {retry_count}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    print(f"[BacktestAgent] Retry attempt {retry_count + 1}/{max_retries}")
                else:
                    print(f"[BacktestAgent] Max retries ({max_retries}) exceeded for RuntimeError")
                    return {"error": str(e), "error_type": "RuntimeError"}
                    
            except Exception as e:
                # Other exceptions - return error immediately
                import traceback
                print(f"[BacktestAgent] Batch backtest error: {e}")
                print(f"[BacktestAgent] Full traceback:")
                traceback.print_exc()  # 打印完整的堆栈跟踪
                return {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
        
        # Should not reach here
        return {"error": "Unexpected error in retry loop"}
