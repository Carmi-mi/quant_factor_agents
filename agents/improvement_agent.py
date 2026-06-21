"""
Improvement Agent
Multi-round alpha improvement using LLM conversation
Processes backtest results grouped by idea, improves each alpha under each idea
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from core.agent_base import Agent
from services.llm_service import LLMService
from infrastructure.prompt_templates import PromptTemplates
from agents.backtest_agent import BacktestAgent


class ImprovementAgent(Agent):
    """
    Expression improvement agent with internal multi-round loop

    Input format adaptation:
    - Supports backtest results grouped by idea (backtest_results.json format)
    - Each idea contains multiple alphas, improved group by group
    - Uses is_tests to analyze failure reasons for targeted improvements

    Usage:
        result = agent.run({
            "backtest_results": {...},  # Backtest results content
            "max_rounds": 3,            # Max improvement rounds per alpha
            "target_sharpe": 1.25       # Target sharpe ratio
        })
    """

    def __init__(self, config: Dict[str, Any], llm_service: LLMService = None):
        """Initialize with LLM service and BacktestAgent"""
        super().__init__("ImprovementAgent", config)
        self.llm = llm_service
        self.max_rounds = config.get("improvement", {}).get("max_rounds_per_expr", 3)
        self.target_sharpe = config.get("improvement", {}).get("target_sharpe", 1.25)

        # Enforce LLM usage
        if llm_service is None:
            raise ValueError("LLM service is required for ImprovementAgent.")
        self.use_llm = True

        # Pre-load valid operators for spell checking
        self.valid_operators = self._load_operators()
        self.valid_operator_params = self._get_operator_params()

        # Initialize BacktestAgent for running backtests
        self.backtest_agent = BacktestAgent(config, llm_service)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry: Process backtest results and improve alphas

        Args:
            input_data: {
                "backtest_results": {...},      # Backtest results content (required)
                "max_rounds": 3,                # Max improvement rounds per alpha (optional)
                "target_sharpe": 1.25,          # Target sharpe ratio (optional)
                "verbose": True                 # Whether to print detailed logs (optional)
            }

        Returns:
            {
                "status": "success|partial|failed",
                "improved_alphas": [...],
                "improvement_summary": {...},
                "failed_alphas": [...]
            }
        """
        self.validate_input(input_data, ["backtest_results"])

        backtest_results = input_data["backtest_results"]
        max_rounds = input_data.get("max_rounds", self.max_rounds)
        target_sharpe = input_data.get("target_sharpe", self.target_sharpe)
        verbose = input_data.get("verbose", True)

        file_info = backtest_results.get("file_info", {})
        final_alphas = backtest_results.get("final_alphas", [])

        if not final_alphas:
            return {"status": "failed", "error": "No alphas found", "improved_alphas": [], "improvement_summary": {}}

        if verbose:
            print(f"\n[ImprovementAgent] Starting improvement")
            print(f"[ImprovementAgent] Ideas: {len(final_alphas)}, Target Sharpe: {target_sharpe}")

        # Collect all alphas
        all_alphas = []
        for idea_group in final_alphas:
            idea_id = idea_group.get("idea_id", "unknown")
            idea_content = idea_group.get("idea_content", "")
            settings = idea_group.get("settings", {})
            for alpha in idea_group.get("alphas", []):
                alpha_info = self._extract_alpha_info(alpha, idea_id, idea_content, settings)
                if alpha_info:
                    all_alphas.append(alpha_info)

        if verbose:
            print(f"[ImprovementAgent] Total alphas: {len(all_alphas)}")

        # Improve alphas
        improved_results, failed_results = [], []
        for idx, alpha_info in enumerate(all_alphas, 1):
            if verbose:
                print(f"\n[{idx}/{len(all_alphas)}] Alpha: {alpha_info['alpha_id']}, Sharpe: {alpha_info['metrics'].get('sharpe', 0):.2f}")

            improvement_result = self._improve_alpha(
                alpha_info=alpha_info, max_rounds=max_rounds, target_sharpe=target_sharpe,
                verbose=verbose
            )

            if improvement_result["status"] in ["target_achieved", "success", "max_rounds"]:
                improved_results.append(improvement_result)
            else:
                failed_results.append(improvement_result)

        summary = self._generate_summary(improved_results, failed_results, all_alphas)

        if verbose:
            print(f"\n[ImprovementAgent] Complete! Success: {summary['success_count']}/{summary['total_count']}, Target: {summary['target_achieved_count']}")

        return {
            "status": "success" if improved_results else "failed",
            "improved_alphas": improved_results,
            "failed_alphas": failed_results,
            "improvement_summary": summary,
            "file_info": file_info
        }

    def _extract_alpha_info(self, alpha: Dict, idea_id: str, idea_content: str, settings: Dict) -> Optional[Dict]:
        """Extract key information from alpha data"""
        try:
            is_stats = alpha.get("is_stats", {})
            is_tests = alpha.get("is_tests", {})

            def get_val(key):
                v = is_stats.get(key)
                return float(v.get("0", 0)) if isinstance(v, dict) else float(v) if v else 0.0

            failures = []
            if is_tests:
                for idx, name in is_tests.get("name", {}).items():
                    if is_tests.get("result", {}).get(idx) == "FAIL":
                        failures.append({"test_name": name, "limit": is_tests.get("limit", {}).get(idx),
                                         "value": is_tests.get("value", {}).get(idx)})

            return {
                "alpha_id": alpha.get("alpha_id", "unknown"), "expression": alpha.get("expression", ""),
                "idea_id": idea_id, "idea_content": idea_content, "settings": settings,
                "metrics": {"sharpe": get_val("sharpe"), "returns": get_val("returns"),
                           "drawdown": get_val("drawdown"), "turnover": get_val("turnover"),
                           "fitness": get_val("fitness"), "margin": get_val("margin")},
                "test_failures": failures, "is_tests": is_tests
            }
        except Exception as e:
            print(f"[ImprovementAgent] Extract error: {e}")
            return None

    def _improve_alpha(self, alpha_info: Dict[str, Any], max_rounds: int,
                       target_sharpe: float, verbose: bool) -> Dict[str, Any]:
        """Improve a single alpha through multi-round LLM conversation with elite pool"""
        # Build defect reason from test failures
        failures = alpha_info["test_failures"]
        defect_reason = "Underperforming - needs optimization" if not failures else \
            "; ".join([f"{f['test_name']}: {f.get('value', 'N/A')} (limit: {f.get('limit', 'N/A')})"
                      for f in failures[:3]])

        alpha_id = alpha_info["alpha_id"]

        # Initialize elite pool
        elite_pool = []

        # Current expression for improvement (from elite pool)
        current_expr = {
            "id": alpha_id, "code": alpha_info["expression"],
            "source_idea": alpha_info["idea_content"], "defect_reason": defect_reason,
            "metrics": alpha_info["metrics"], "settings": alpha_info["settings"]
        }

        # Pre-compute allowed fields and exempt fields (avoid repeated extraction)
        allowed_fields = self._extract_fields(alpha_info["expression"])
        exempt_fields = self._get_exempt_fields()
        valid_fields = allowed_fields | exempt_fields

        improvement_history = []
        best_expression = current_expr.copy()
        best_sharpe = alpha_info["metrics"].get("sharpe", -999)
        initial_sharpe = best_sharpe

        # Loop recovery state
        loop_recovery_mode = False
        loop_recovery_count = 0
        max_loop_recovery = 3
        loop_recovery_history = []

        # Store last reflection for retry logic
        last_reflection_cache = None

        # Multi-round improvement loop with elite pool
        for round_num in range(max_rounds):
            if verbose:
                mode_str = " [LOOP RECOVERY]" if loop_recovery_mode else ""
                print(f"  [Round {round_num + 1}/{max_rounds}]{mode_str} ------------------------")

            # Build last reflection with loop warning if in recovery mode
            # Note: User should implement reflection saving after backtest
            last_reflection = last_reflection_cache
            last_reflection_cache = None

            if loop_recovery_mode and loop_recovery_history:
                last_reflection = last_reflection.copy() if last_reflection else {}
                last_reflection["loop_recovery_warning"] = (
                    f"CRITICAL: You are in loop recovery mode (attempt {loop_recovery_count}/{max_loop_recovery}). "
                    f"Previous attempts failed: {[h['error'] for h in loop_recovery_history]}. "
                    f"You MUST generate a COMPLETELY DIFFERENT approach. "
                    f"Do NOT use any expression similar to previous attempts."
                )

            improve_result = self._improve_round(
                expr=current_expr, round_num=round_num,
                improvement_history=improvement_history,
                source_idea=alpha_info["idea_content"],
                last_reflection=last_reflection,
                valid_fields=valid_fields,
                skip_history_update=loop_recovery_mode
            )

            if improve_result.get("status") != "success":
                # Check if it's a loop error
                if improve_result.get("status") == "loop_detected":
                    if loop_recovery_count >= max_loop_recovery:
                        if verbose:
                            print(f"  [Round {round_num + 1}] Max loop recovery ({max_loop_recovery}) exceeded, stopping")
                        self._save_loop_recovery_errors(alpha_info, loop_recovery_history)
                        break

                    loop_recovery_mode = True
                    loop_recovery_count += 1
                    loop_recovery_history.append({
                        "round": round_num + 1,
                        "attempt": loop_recovery_count,
                        "error": improve_result.get("reason", ""),
                        "expressions": improve_result.get("last_expressions", []),
                        "settings": improve_result.get("last_settings", {})
                    })

                    if verbose:
                        print(f"  [Round {round_num + 1}] Loop detected, entering recovery mode (attempt {loop_recovery_count})")

                    round_num -= 1
                    continue
                else:
                    # Other failures (validation_failed, etc.) - save to history before breaking
                    if verbose:
                        print(f"  [Round {round_num + 1}] Improvement failed: {improve_result.get('status')}")

                    # Save failed attempt to history with validation errors
                    validation_failed_alphas = []
                    errors = improve_result.get("errors", [])
                    # Get expressions from validation errors if available
                    if errors and len(errors) > 0:
                        last_error = errors[-1]
                        failed_exprs = last_error.get("expressions", [])
                        for idx, failed_code in enumerate(failed_exprs):
                            error_detail = ""
                            for err in errors:
                                for e in err.get("errors", "").split("; "):
                                    if f"Variant {idx + 1}" in e:
                                        error_detail = e
                                        break
                            validation_failed_alphas.append({
                                "id": f"{current_expr.get('id', 'unknown')}_r{round_num + 1}_v{idx + 1}_failed",
                                "code": failed_code,
                                "metrics": {"sharpe": 0, "returns": 0, "drawdown": 0, "turnover": 0, "fitness": 0, "margin": 0},
                                "changes_made": current_expr.get("changes_made", ""),
                                "expected_improvement": current_expr.get("expected_improvement", ""),
                                "confidence": current_expr.get("confidence", 0.5),
                                "validation_error": error_detail
                            })

                    if validation_failed_alphas:
                        self._save_round_to_history(
                            improvement_history, round_num + 1, validation_failed_alphas,
                            f"Validation failed: {improve_result.get('reason', 'Unknown error')}",
                            None
                        )
                    break

            # Success - clear recovery mode
            if loop_recovery_mode:
                if verbose:
                    print(f"  [Round {round_num + 1}] Loop recovery successful after {loop_recovery_count} attempts")
                loop_recovery_mode = False
                loop_recovery_count = 0
                loop_recovery_history = []

            # Get all valid expressions from this round
            expressions = improve_result["expressions"]
            if verbose:
                print(f"  [Round {round_num + 1}] Generated {improve_result['total_variants']} variants, {improve_result['valid_variants']} valid")

            # Run backtest using BacktestAgent
            try:
                # Build idea input for BacktestAgent with reviewed idea info and settings
                backtest_settings = current_expr.get("settings")

                idea_input = {
                    "idea_id": alpha_info["idea_id"],
                    "idea_content": alpha_info["idea_content"],
                    "expressions": [expr["code"] for expr in expressions]
                }

                if verbose:
                    print(f"  [Round {round_num + 1}] Starting backtest for {len(expressions)} expressions...")
                    if backtest_settings:
                        print(f"    Settings: decay={backtest_settings.get('decay', 0)}, "
                              f"neutralization={backtest_settings.get('neutralization', 'INDUSTRY')}, "
                              f"max_trade={backtest_settings.get('max_trade', 'OFF')}")

                # Run backtest with reviewed idea and settings
                backtest_result = self.backtest_agent.run(idea_input, settings=backtest_settings)

                # Extract full backtest results from alphas
                full_backtest_results = backtest_result.get("alphas", [])

                if not full_backtest_results:
                    if verbose:
                        print(f"  [Round {round_num + 1}] No valid backtest results (all expressions failed or below threshold)")
                    # Continue to next round instead of breaking
                    # Use original expressions for reflection to guide next iteration
                    reflection = self._reflect_with_full_data(expressions, [], current_expr, improvement_history)
                    if verbose and reflection:
                        print(f"  [Round {round_num + 1}] Reflection: {reflection.get('summary', 'N/A')[:100]}...")

                    # Save round to history even if no qualified alphas (record the attempt)
                    failed_alphas = []
                    for expr in expressions:
                        failed_alphas.append({
                            "id": expr.get("id", "unknown"),
                            "code": expr["code"],
                            "metrics": {"sharpe": 0, "returns": 0, "drawdown": 0, "turnover": 0, "fitness": 0, "margin": 0},
                            "changes_made": expr.get("changes_made", ""),
                            "expected_improvement": expr.get("expected_improvement", ""),
                            "confidence": expr.get("confidence", 0.5)
                        })
                    self._save_round_to_history(
                        improvement_history, round_num + 1, failed_alphas,
                        "All expressions failed or below threshold",
                        reflection
                    )

                    # Continue to next round
                    continue

                if verbose:
                    sharpe_list = []
                    for result in full_backtest_results:
                        is_stats = result.get("is_stats", {})
                        sharpe_val = is_stats.get("sharpe", {})
                        sharpe = float(sharpe_val.get("0", 0)) if isinstance(sharpe_val, dict) else float(sharpe_val) if sharpe_val else 0
                        sharpe_list.append(f"{sharpe:.2f}")
                    print(f"  [Round {round_num + 1}] Backtest completed: {len(full_backtest_results)} results (Sharpe: {', '.join(sharpe_list)})")

            except Exception as e:
                if verbose:
                    print(f"  [Round {round_num + 1}] Backtest failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Continue to next round instead of breaking
                continue

            # 1. Use full backtest data for reflection
            if verbose:
                print(f"  [Round {round_num + 1}] Analyzing backtest results with LLM...")
            reflection = self._reflect_with_full_data(expressions, full_backtest_results, current_expr, improvement_history)
            if verbose and reflection:
                print(f"  [Round {round_num + 1}] Reflection: {reflection.get('summary', 'N/A')[:100]}...")

            # 2. Select multiple best expressions
            if verbose:
                print(f"  [Round {round_num + 1}] Selecting best expressions (min_sharpe=0.5)...")
            top_alphas = self._select_best_expressions(expressions, full_backtest_results, top_n=3, current_round=round_num + 1)

            if not top_alphas:
                if verbose:
                    print(f"  [Round {round_num + 1}] No qualified alphas selected, continuing to next round...")
                # Save round to history even if no qualified alphas (record the attempt)
                low_sharpe_alphas = []
                for expr, result in zip(expressions, full_backtest_results):
                    is_stats = result.get("is_stats", {})
                    def get_val(key):
                        v = is_stats.get(key)
                        return float(v.get("0", 0)) if isinstance(v, dict) else float(v) if v else 0.0
                    low_sharpe_alphas.append({
                        "id": result.get("alpha_id", expr.get("id", "unknown")),
                        "code": expr["code"],
                        "metrics": {
                            "sharpe": get_val("sharpe"),
                            "returns": get_val("returns"),
                            "drawdown": get_val("drawdown"),
                            "turnover": get_val("turnover"),
                            "fitness": get_val("fitness"),
                            "margin": get_val("margin")
                        },
                        "changes_made": expr.get("changes_made", ""),
                        "expected_improvement": expr.get("expected_improvement", ""),
                        "confidence": expr.get("confidence", 0.5)
                    })
                self._save_round_to_history(
                    improvement_history, round_num + 1, low_sharpe_alphas,
                    "No alphas met selection criteria (min_sharpe=0.5)",
                    reflection
                )
                # Continue to next round instead of breaking
                continue

            if verbose:
                sharpe_list = [f"{alpha['metrics']['sharpe']:.2f}" for alpha in top_alphas]
                print(f"  [Round {round_num + 1}] Selected {len(top_alphas)} top alphas (Sharpe: {', '.join(sharpe_list)})")

            # 3. Update elite pool
            elite_pool = self._update_elite_pool(elite_pool, top_alphas)
            if verbose:
                print(f"  [Round {round_num + 1}] Elite pool updated: {len(elite_pool)} alphas (max 5)")

            # 4. Select next expression from elite pool for next round
            current_expr = self._select_from_elite_pool(elite_pool, round_num + 1, strategy="best")

            # 5. Save to improvement history (multiple alphas per round)
            self._save_round_to_history(improvement_history, round_num + 1, top_alphas, current_expr.get("changes_made", ""), reflection)

            # 6. Update best tracking
            if current_expr["metrics"]["sharpe"] > best_sharpe:
                best_sharpe = current_expr["metrics"]["sharpe"]
                best_expression = current_expr.copy()
                if verbose:
                    print(f"  [Round {round_num + 1}] New best Sharpe: {best_sharpe:.2f}")

            # 7. Check target
            if best_sharpe >= target_sharpe:
                if verbose:
                    print(f"  [Round {round_num + 1}] Target Sharpe {target_sharpe} achieved!")
                break

        final_status = "max_rounds" if len(improvement_history) >= max_rounds else "stopped"
        overall_reflection = self._overall_reflection(
            history=improvement_history,
            source_idea=alpha_info["idea_content"],
            initial_sharpe=initial_sharpe, final_sharpe=best_sharpe
        )
        # Save overall reflection to file
        self._save_overall_reflection(alpha_info, improvement_history, overall_reflection,
                                      initial_sharpe, best_sharpe, final_status)

        return {
            "status": final_status,
            "alpha_id": alpha_id,
            "original_expression": alpha_info["expression"],
            "final_expression": current_expr.get("code", "") if improvement_history else alpha_info["expression"],
            "improvement_history": improvement_history,
            "rounds_completed": len(improvement_history),
            "best_expression": best_expression,
            "final_metrics": current_expr.get("metrics", {}) if improvement_history else alpha_info["metrics"],
            "initial_sharpe": initial_sharpe,
            "final_sharpe": best_sharpe,
            "sharpe_improvement": best_sharpe - initial_sharpe,
            "idea_id": alpha_info["idea_id"],
            "overall_reflection": overall_reflection,
            "elite_pool": elite_pool  # Include final elite pool in results
        }

    def _improve_round(self, expr: Dict, round_num: int, improvement_history: List[Dict],
                        source_idea: str, last_reflection: Optional[Dict] = None,
                        valid_fields: Optional[set] = None,
                        max_validation_retries: int = 3,
                        skip_history_update: bool = False) -> Dict:
        """Execute single round improvement with validation retry (loop handling moved to main loop)
        
        LLM can generate multiple expression variants but only one settings configuration.
        All expressions will be validated and the best valid one will be returned.
        """
        validation_errors = []
        last_improved_codes = None
        last_improved_settings = None

        for retry in range(max_validation_retries):
            # Call LLM for improvement
            if retry == 0:
                # First attempt: normal improvement
                result = self._call_llm_for_improvement(
                    expr=expr, source_idea=source_idea, strategy="auto",
                    improvement_history=improvement_history or None, last_reflection=last_reflection
                )
            else:
                # Retry: use validation error correction prompt
                result = self._call_llm_for_validation_correction(
                    expr=expr, source_idea=source_idea,
                    validation_errors=validation_errors,
                    valid_fields=valid_fields or set(),
                    retry_count=retry
                )

            if not result:
                return {"status": "failed", "reason": "No improvement"}

            # Get expressions (support both single and multiple)
            expressions = result.get("improved_expressions", [])
            if not expressions and result.get("improved_expression"):
                # Backward compatibility: support old format
                expressions = [result["improved_expression"]]
            if not expressions:
                return {"status": "failed", "reason": "No improved expressions"}

            improved_settings = result.get("improved_settings", {})
            last_improved_codes = expressions
            last_improved_settings = improved_settings

            # Validate all expressions, collect valid ones
            valid_expressions = []
            all_errors = []

            for idx, expr_code in enumerate(expressions):
                validation = self._validate_improvement(
                    expr_code, improved_settings, expr, improvement_history, valid_fields
                )

                if validation["valid"]:
                    valid_expressions.append({
                        "code": expr_code,
                        "index": idx + 1
                    })
                else:
                    all_errors.append({
                        "expression_index": idx + 1,
                        "expression": expr_code,
                        "errors": validation["reason"]
                    })

            if valid_expressions:
                # Return all valid expressions for backtesting
                current_settings = expr.get("settings", {})
                new_settings = {**current_settings, **improved_settings} if improved_settings else current_settings

                expressions_list = []
                for i, ve in enumerate(valid_expressions):
                    expressions_list.append({
                        "id": f"{expr.get('id', 'unknown')}_r{round_num + 1}_v{i + 1}",
                        "code": ve["code"],
                        "source_idea": expr.get("source_idea", ""),
                        "changes_made": result.get("changes_made", ""),
                        "expected_improvement": result.get("expected_improvement", ""),
                        "confidence": result.get("confidence", 0.5),
                        "settings": new_settings,
                        "settings_changes": improved_settings,
                        "variant_index": ve["index"]
                    })

                return {
                    "status": "success",
                    "expressions": expressions_list,  # All valid expressions
                    "total_variants": len(expressions),
                    "valid_variants": len(valid_expressions)
                }

            # All expressions failed validation
            error_reason = f"All {len(expressions)} expression(s) failed validation: " + \
                          "; ".join([f"Variant {e['expression_index']}: {e['errors']}" for e in all_errors])

            # Check if it's a loop (return to main loop for recovery)
            if any("Loop detected" in e["errors"] for e in all_errors):
                return {
                    "status": "loop_detected",
                    "reason": error_reason,
                    "last_expressions": last_improved_codes,
                    "last_settings": last_improved_settings
                }

            # Regular validation error
            validation_errors.append({
                "retry": retry + 1,
                "expressions": expressions,
                "settings": improved_settings,
                "errors": error_reason
            })

        # Max retries exceeded
        self._save_validation_errors(expr, validation_errors, error_type="validation")
        return {"status": "validation_failed", "reason": f"Max validation retries ({max_validation_retries}) exceeded", "errors": validation_errors}

    def _validate_improvement(self, expr_code: str, settings: Dict, current_expr: Dict,
                               history: List[Dict], valid_fields: Optional[set] = None) -> Dict:
        """Validate LLM returned expression and settings, collect all errors"""
        errors = []

        # Check syntax
        if not self._validate_syntax(expr_code):
            errors.append("Invalid expression syntax")

        # Check if expression is empty
        if not expr_code or len(expr_code.strip()) == 0:
            errors.append("Empty expression")

        # Check if unchanged
        if expr_code == current_expr.get("code", "") and not settings:
            errors.append("No changes made")

        # Validate field usage - use pre-computed valid_fields if available
        if valid_fields is not None:
            used_fields = self._extract_fields(expr_code)
            invalid_fields = used_fields - valid_fields
            if invalid_fields:
                errors.append(f"Invalid fields: {', '.join(invalid_fields)}")
        else:
            # Fallback: compute on the fly (should not happen in normal flow)
            original_code = current_expr.get("code", "")
            field_check = self._validate_field_usage(expr_code, original_code)
            if not field_check["valid"]:
                errors.append(field_check["reason"])

        # Validate quoted fields must use double quotes
        quoted_check = self._validate_quoted_fields(expr_code)
        if not quoted_check["valid"]:
            errors.append(quoted_check["reason"])

        # Validate operator spelling
        spell_check = self._validate_operator_spelling(expr_code, valid_fields if valid_fields else set())
        if not spell_check["valid"]:
            errors.append(spell_check["reason"])

        # Validate settings with region
        region = current_expr.get("settings", {}).get("region", "USA")
        valid_settings = self._validate_settings(settings, region)
        if not valid_settings["valid"]:
            errors.append(f"Invalid settings: {valid_settings['reason']}")

        # Return all errors if any
        if errors:
            return {"valid": False, "reason": "; ".join(errors)}

        # Enhanced loop detection
        loop_check = self._detect_loop(expr_code, settings, current_expr, history)
        if loop_check["is_loop"]:
            return {"valid": False, "reason": f"Loop detected: {loop_check['reason']}"}

        return {"valid": True, "reason": ""}

    def _select_best_expression(self, expressions: List[Dict],
                                 backtest_results: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Select the best expression from backtest results.
        Currently selects by highest Sharpe ratio.

        Args:
            expressions: List of expression dicts with code, settings, etc.
            backtest_results: List of backtest metrics for each expression

        Returns:
            Tuple of (best_expression, best_metrics)
        """
        if not expressions or not backtest_results:
            return None, None

        if len(expressions) != len(backtest_results):
            # Mismatch, use first valid pair
            for expr, metrics in zip(expressions, backtest_results):
                if metrics:
                    return expr, metrics
            return None, None

        best_idx = 0
        best_sharpe = backtest_results[0].get("sharpe", -999) if backtest_results[0] else -999

        for i, metrics in enumerate(backtest_results[1:], 1):
            if metrics and metrics.get("sharpe", -999) > best_sharpe:
                best_sharpe = metrics.get("sharpe")
                best_idx = i

        return expressions[best_idx], backtest_results[best_idx]

    def _select_best_expressions(
        self,
        expressions: List[Dict],
        backtest_results: List[Dict],
        top_n: int = 3,
        min_sharpe: float = 0.5,
        current_round: int = 1
    ) -> List[Dict]:
        """
        从回测结果中选出多个优秀alpha，返回完整格式可直接加入精英池

        Args:
            expressions: 表达式列表
            backtest_results: 回测结果列表（完整原始数据）
            top_n: 最多返回几个
            min_sharpe: 入选门槛
            current_round: 当前轮次

        Returns:
            优秀alpha列表，每个包含完整格式（id, code, metrics, settings等）
        """
        from datetime import datetime

        candidates = []

        for expr, backtest_result in zip(expressions, backtest_results):
            if not backtest_result:
                continue

            # Extract metrics from BacktestAgent result format
            is_stats = backtest_result.get("is_stats", {})

            def get_stat_value(key):
                """Extract value from is_stats (handles dict format like {"0": value})"""
                v = is_stats.get(key)
                if isinstance(v, dict):
                    return float(v.get("0", 0))
                return float(v) if v else 0.0

            sharpe = get_stat_value("sharpe")
            if sharpe < min_sharpe:
                continue

            candidate = {
                "id": backtest_result.get("alpha_id", expr.get("id", f"unknown_r{current_round}_v{expr.get('variant_index', 1)}")),
                "code": backtest_result.get("expression", expr["code"]),
                "metrics": {
                    "sharpe": sharpe,
                    "returns": get_stat_value("returns"),
                    "drawdown": get_stat_value("drawdown"),
                    "turnover": get_stat_value("turnover"),
                    "fitness": get_stat_value("fitness"),
                    "margin": get_stat_value("margin")
                },
                "settings": backtest_result.get("settings", expr.get("settings", {})),
                "defect_reason": expr.get("defect_reason", "Underperforming"),
                "source_idea": expr.get("source_idea", ""),
                "round": current_round,
                "variant_index": expr.get("variant_index", 1),
                "changes_made": expr.get("changes_made", ""),
                "added_at": datetime.now().isoformat(),
                # Include full backtest data for reflection
                "full_backtest_data": backtest_result
            }
            candidates.append(candidate)

        # 按 Sharpe 排序，取前 N
        candidates.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)
        return candidates[:top_n]

    def _update_elite_pool(self, pool: List[Dict], new_alphas: List[Dict]) -> List[Dict]:
        """
        将新alpha加入精英池，维护池子大小，按Sharpe排序

        Args:
            pool: 当前精英池
            new_alphas: 新加入的alpha列表

        Returns:
            更新后的精英池
        """
        # 合并
        combined = pool + new_alphas

        # 按 Sharpe 排序
        combined.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)

        # 截断到最大容量
        max_size = self.config.get("improvement", {}).get("elite_pool_size", 5)

        return combined[:max_size]

    def _select_from_elite_pool(
        self,
        pool: List[Dict],
        current_round: int,
        strategy: str = "best"
    ) -> Dict:
        """
        从精英池中选择下一个要改进的alpha

        Args:
            pool: 精英池
            current_round: 当前轮次
            strategy: 选择策略 ("best"=选最好的, "round_robin"=轮流选)

        Returns:
            选中的alpha（完整格式，兼容_improve_round需要的expr格式）
        """
        if not pool:
            raise ValueError("Elite pool is empty")

        if strategy == "best":
            selected = pool[0]
        elif strategy == "round_robin":
            idx = current_round % len(pool)
            selected = pool[idx]
        else:
            selected = pool[0]

        # 返回完整格式，兼容 _improve_round
        return {
            "id": selected["id"],
            "code": selected["code"],
            "metrics": selected["metrics"],
            "settings": selected["settings"],
            "defect_reason": selected.get("defect_reason", "Underperforming"),
            "source_idea": selected.get("source_idea", ""),
            "changes_made": selected.get("changes_made", ""),
            "variant_index": selected.get("variant_index", 1)
        }

    def _reflect_with_full_data(
        self,
        expressions: List[Dict],
        full_backtest_results: List[Dict],
        current_expr: Dict,
        history: List[Dict]
    ) -> Optional[Dict]:
        """
        使用完整回测数据向 LLM 请求反思，一次性发送所有变体的完整原始数据

        Args:
            expressions: 表达式列表
            full_backtest_results: 完整回测结果列表
            current_expr: 当前表达式
            history: 改进历史

        Returns:
            反思结果字典
        """
        try:
            # 构建包含完整回测数据的列表
            full_data = []
            for expr, result in zip(expressions, full_backtest_results):
                full_data.append({
                    "expression_id": expr.get("id", "unknown"),
                    "expression_code": expr["code"],
                    "variant_index": expr.get("variant_index", 1),
                    "expected_improvement": expr.get("expected_improvement", ""),
                    "full_backtest_result": result  # 完整原始回测数据
                })

            # 调用 PromptTemplates 生成反思 prompt
            messages = PromptTemplates.reflection_with_full_data(
                current_code=current_expr["code"],
                current_metrics=current_expr["metrics"],
                variants_data=full_data,
                improvement_history=history
            )

            response = self.llm.client.chat(messages, )
            return self._parse_llm_response(response)

        except Exception as e:
            print(f"[ImprovementAgent] Full data reflection failed: {e}")
            return None

    def _save_round_to_history(
        self,
        history: List[Dict],
        round_num: int,
        top_alphas: List[Dict],
        changes_made: str,
        reflection: Optional[Dict]
    ):
        """
        保存本轮多个alpha到history，按轮分组，不保存settings

        Args:
            history: 历史列表
            round_num: 轮次
            top_alphas: 多个优秀alpha
            changes_made: 改进说明
            reflection: 反思结果
        """
        alphas_list = []
        for alpha in top_alphas:
            alphas_list.append({
                "alpha_id": alpha["id"],
                "expression_code": alpha["code"],
                "metrics": alpha["metrics"],
                "changes_made": alpha.get("changes_made", ""),
                "expected_improvement": alpha.get("expected_improvement", ""),
                "confidence": alpha.get("confidence", 0.5)
            })

        history.append({
            "round": round_num,
            "alphas": alphas_list,
            "changes_made": changes_made,
            "reflection": reflection.get("summary", "") if reflection else ""
        })

    def _build_llm_history(self, improvement_history: List[Dict]) -> List[Dict]:
        """
        从improvement_history构建给LLM看的历史，简化格式

        Args:
            improvement_history: 完整历史

        Returns:
            简化后的历史列表
        """
        llm_history = []
        for h in improvement_history:
            for alpha in h["alphas"]:
                llm_history.append({
                    "expression_code": alpha["expression_code"],
                    "metrics": alpha["metrics"],
                    "changes_made": h["changes_made"]
                })
        return llm_history

    def _detect_loop(self, expr_code: str, settings: Dict, current_expr: Dict,
                     history: List[Dict]) -> Dict:
        """Detect various loop patterns in improvement history

        Compatible with new history format where each item has 'alphas' list.
        """
        if not history:
            return {"is_loop": False, "type": None, "reason": ""}

        # Build new settings
        current_settings = current_expr.get("settings", {})
        new_settings = {**current_settings, **settings} if settings else current_settings

        # Detect duplicate by normalizing and comparing
        normalized_new = self._normalize_expression(expr_code)

        for i, h in enumerate(history):
            # Support new format: history item has 'alphas' list
            if "alphas" in h:
                # Check all alphas in this round
                for alpha in h["alphas"]:
                    hist_code = alpha.get("expression_code", "")
                    normalized_hist = self._normalize_expression(hist_code)

                    # For new format, we don't have settings per alpha in history
                    # So we only check expression code duplication
                    if normalized_new == normalized_hist:
                        is_exact = (hist_code == expr_code)
                        return {
                            "is_loop": True,
                            "type": "exact_duplicate" if is_exact else "semantic_duplicate",
                            "reason": f"Duplicate of round {i + 1}" + ("" if is_exact else " (different formatting)"),
                            "round": i + 1
                        }
            else:
                # Support old format for backward compatibility
                hist_code = h.get("expression_code", "")
                normalized_hist = self._normalize_expression(hist_code)
                if normalized_new == normalized_hist and new_settings == h.get("settings"):
                    is_exact = (hist_code == expr_code)
                    return {
                        "is_loop": True,
                        "type": "exact_duplicate" if is_exact else "semantic_duplicate",
                        "reason": f"Duplicate of round {i + 1}" + ("" if is_exact else " (different formatting)"),
                        "round": i + 1
                    }

        return {"is_loop": False, "type": None, "reason": ""}

    def _normalize_expression(self, expr: str) -> str:
        """Normalize expression for semantic comparison"""
        import re
        # Remove all whitespace
        normalized = re.sub(r'\s+', '', expr)
        # Convert to lowercase
        normalized = normalized.lower()
        return normalized

    def _load_operators(self) -> set:
        """Load operators from config file"""
        import json
        import os

        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "operators_desc.json")
        try:
            with open(config_path, 'r') as f:
                operators = json.load(f)
                return set(operators.keys())
        except Exception:
            # Fallback operators if file not found
            return {
                "abs", "add", "divide", "inverse", "log", "max", "min", "multiply",
                "power", "reverse", "sign", "sqrt", "subtract", "and", "or", "not",
                "if_else", "is_nan", "days_from_last_change", "ts_mean", "ts_std_dev",
                "ts_median", "ts_sum", "ts_product", "ts_rank", "ts_quantile", "ts_zscore",
                "ts_corr", "ts_covariance", "ts_delay", "ts_delta", "ts_decay_linear",
                "ts_decay_exp_window", "ts_scale", "ts_arg_max", "ts_arg_min", "ts_backfill",
                "ts_count_nans", "ts_regression", "ts_av_diff", "rank", "zscore",
                "normalize", "quantile", "scale", "winsorize", "truncate", "regression_neut",
                "group_mean", "group_median", "group_rank", "group_zscore", "group_scale",
                "group_neutralize", "group_normalize", "vec_avg", "vec_sum", "vec_count",
                "bucket", "left_tail", "right_tail", "trade_when", "hump"
            }

    def _get_operator_params(self) -> set:
        """Get common operator parameter names (not data fields)"""
        return {
            # Time series params
            "d", "window", "period", "lag", "factor", "constant", "dense",
            # Statistical params
            "std", "sigma", "useStd", "limit", "tolerance", "scale", "longscale", "shortscale",
            "hump", "maximum", "minimum", "driver", "rettype", "k",
            # Group params
            "group", "g1", "g2", "weight",
            # Other params
            "filter", "mode", "nlength", "target_tvr", "lambda_min", "lambda_max",
            "limit_volume", "maxPercent", "range", "buckets", "constantCheck"
        }

    def _load_operator_docs(self) -> Dict[str, str]:
        """Load detailed operator documentation from operators.json for LLM

        Returns:
            Dictionary of {operator_signature: documentation}
        """
        import json
        import os

        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "operators.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                operators = json.load(f)
                return operators
        except Exception as e:
            print(f"[ImprovementAgent] Failed to load operator docs: {e}")
            # Return empty dict as fallback
            return {}

    def _extract_fields(self, expr_code: str) -> set:
        """Extract data fields from expression by removing operators, params and numbers"""
        import re

        operators = self._load_operators()
        operator_params = self._get_operator_params()

        # Remove keyword arguments (e.g., std=4, driver="gaussian")
        # Pattern: word= (followed by value)
        cleaned_code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^,\)]+', '', expr_code)

        # Find all identifiers
        words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned_code))

        # Remove operators
        fields = words - operators

        # Remove operator parameters
        fields = fields - operator_params

        # Remove Python keywords and constants
        python_keywords = {"if", "else", "and", "or", "not", "in", "is", "None", "True", "False"}
        fields = fields - python_keywords

        # Remove pure numbers (like 20, 60, 0.5)
        fields = {f for f in fields if not f.replace('.', '').replace('-', '').isdigit()}

        return fields

    def _get_exempt_fields(self) -> set:
        """Get exempt fields that can be used without being in original expression"""
        return {
            # Grouping fields
            "country", "sector", "market", "industry", "subindustry",
            # Boolean values (without quotes)
            "false", "true", "f",
            # Special values and distributions (must be quoted in expressions)
            '"NaN"', '"gaussian"', '"uniform"', '"cauchy"'
        }

    def _get_quoted_exempt_fields(self) -> set:
        """Get fields that must appear with double quotes in expressions"""
        return {'"NaN"', '"gaussian"', '"uniform"', '"cauchy"'}

    def _validate_quoted_fields(self, expr_code: str) -> Dict:
        """Validate that fields requiring quotes use double quotes"""
        import re

        # Fields that must be quoted (without quotes for matching)
        must_quote = {"NaN", "gaussian", "uniform", "cauchy"}

        for field in must_quote:
            # Check if field appears without quotes (as a standalone word)
            # Pattern: word boundary + field + word boundary, not preceded by quote
            pattern = r'(?<!"|\')\b' + field + r'\b(?!"|\')'
            if re.search(pattern, expr_code, re.IGNORECASE):
                return {
                    "valid": False,
                    "reason": f'Field "{field}" must be enclosed in double quotes, e.g., "{field}"'
                }

        return {"valid": True, "reason": ""}

    def _validate_operator_spelling(self, expr_code: str, valid_fields: set) -> Dict:
        """Validate operator spelling by checking for potential typos"""
        import re

        # Extract all identifiers from expression
        words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr_code))

        # Remove known valid elements
        valid_elements = (
            self.valid_operators |
            self.valid_operator_params |
            valid_fields |
            self._get_exempt_fields() |
            {"if", "else", "and", "or", "not", "in", "is", "None", "True", "False"}
        )

        # Find unknown identifiers (potential misspelled operators)
        unknown = words - valid_elements

        # Filter out pure numbers
        unknown = {w for w in unknown if not w.replace('.', '').replace('-', '').isdigit()}

        if unknown:
            # Try to find close matches for suggestions
            suggestions = []
            for word in unknown:
                # Check similarity with valid operators
                for op in self.valid_operators:
                    if self._levenshtein_distance(word.lower(), op.lower()) <= 2:
                        suggestions.append(f"{word} -> {op}")
                        break

            reason = f"Unknown identifiers (possible typos): {', '.join(unknown)}"
            if suggestions:
                reason += f". Did you mean: {', '.join(suggestions)}?"
            return {"valid": False, "reason": reason}

        return {"valid": True, "reason": ""}

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _validate_field_usage(self, new_code: str, original_code: str) -> Dict:
        """Validate that new expression only uses fields from original expression"""
        # Extract allowed fields from original expression
        allowed_fields = self._extract_fields(original_code)

        # Add exempt fields
        allowed_fields = allowed_fields | self._get_exempt_fields()

        # Extract fields used in new expression
        used_fields = self._extract_fields(new_code)

        # Check for invalid fields
        invalid_fields = used_fields - allowed_fields

        if invalid_fields:
            return {
                "valid": False,
                "reason": f"Invalid fields: {', '.join(invalid_fields)}. Only allowed: {', '.join(allowed_fields)}"
            }

        return {"valid": True, "reason": ""}

    def _get_valid_neutralization(self, region: str) -> list:
        """Get valid neutralization options based on region"""
        region = region.upper() if region else "USA"

        if region in ['ASI', 'EUR', 'GLB']:
            return ['COUNTRY', "MARKET", "INDUSTRY", "SUBINDUSTRY", "SECTOR",
                    "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING",
                    "FAST", "SLOW", "SLOW_AND_FAST"]
        elif region in ['CHN', 'KOR', 'IND']:
            return ["MARKET", "INDUSTRY", "SUBINDUSTRY", "SECTOR",
                    "REVERSION_AND_MOMENTUM", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST"]
        elif region == 'MEA':
            return ['COUNTRY', "MARKET", "INDUSTRY", "SUBINDUSTRY", "SECTOR"]
        else:  # USA and others
            return ["MARKET", "INDUSTRY", "SUBINDUSTRY", "SECTOR",
                    "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING",
                    "FAST", "SLOW", "SLOW_AND_FAST"]

    def _validate_settings(self, settings: Dict, region: str = "USA") -> Dict:
        """Validate settings values based on region, collect all errors"""
        if not settings:
            return {"valid": True, "reason": ""}

        errors = []

        # Valid ranges
        valid_neutralization = self._get_valid_neutralization(region)
        valid_max_trade = ["ON", "OFF"]
        valid_delay = [0, 1]

        for key, value in settings.items():
            if key == "neutralization":
                if value not in valid_neutralization:
                    errors.append(f"Invalid neutralization: {value} for region {region}")
            elif key == "decay":
                if not isinstance(value, (int, float)) or value < 0 or value > 512:
                    errors.append(f"Decay out of range: {value} (must be 0 <= decay <= 512)")
            elif key == "truncation":
                if not isinstance(value, (int, float)) or value <= 0 or value >= 1:
                    errors.append(f"Truncation out of range: {value} (must be 0 < truncation < 1)")
            elif key == "delay":
                if value not in valid_delay:
                    errors.append(f"Invalid delay: {value} (must be 0 or 1)")
            elif key == "max_trade":
                if value not in ["ON", "OFF"]:
                    errors.append(f"Invalid max_trade: {value} (must be ON or OFF)")

        if errors:
            return {"valid": False, "reason": "; ".join(errors)}

        return {"valid": True, "reason": ""}

    def _call_llm_for_improvement(self, expr: Dict, source_idea: str, strategy: str,
                                   improvement_history: List[Dict] = None,
                                   last_reflection: Optional[Dict] = None) -> Optional[Dict]:
        """Call LLM for improvement (expression + settings) - unified call with is_first_round flag"""
        try:
            expr_code = expr.get("code", "")
            current_metrics = expr.get("metrics", {})
            current_settings = expr.get("settings", {})

            # Build backtest history for subsequent rounds using new format
            backtest_history = None
            if improvement_history:
                backtest_history = self._build_llm_history(improvement_history)

            # Load detailed operator documentation for LLM
            operator_docs = self._load_operator_docs()

            # Get region from expression settings
            region = expr.get("settings", {}).get("region", "USA")

            # Unified call - is_first_round determined by backtest_history inside the method
            messages = PromptTemplates.alpha_improvement_conversation(
                expr_code=expr_code, source_idea=source_idea,
                defect_reason=expr.get("defect_reason", ""), metrics=current_metrics,
                strategy=strategy, target_sharpe=self.target_sharpe,
                backtest_history=backtest_history,
                current_settings=current_settings,
                current_reflection=last_reflection,
                operator_docs=operator_docs,
                region=region
            )

            response = self.llm.client.chat(messages)
            return self._parse_llm_response(response)

        except Exception as e:
            print(f"[ImprovementAgent] LLM call failed: {e}")
            return None

    def _reflect_on_round(self, expr_code: str, backtest_result: Dict,
                          previous_changes: str, expected_improvement: str) -> Optional[Dict]:
        """Reflect on current round's result"""
        try:
            prompt = PromptTemplates.alpha_round_reflection(expr_code, backtest_result,
                                                            previous_changes, expected_improvement)
            messages = [{"role": "system", "content": "You are a quantitative analyst."},
                        {"role": "user", "content": prompt}]
            response = self.llm.client.chat(messages)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            return json.loads(match.group()) if match else None
        except Exception as e:
            print(f"[ImprovementAgent] Reflection failed: {e}")
            return None

    def _overall_reflection(self, history: List[Dict], source_idea: str,
                            initial_sharpe: float, final_sharpe: float) -> Optional[Dict]:
        """Perform overall reflection after all rounds"""
        try:
            if not history:
                return None

            # Build history summary compatible with new format
            history_summary = []
            for h in history:
                round_num = h.get('round', 0)
                changes_made = h.get('changes_made', '')

                # New format: get metrics from first alpha in the round
                if "alphas" in h and h["alphas"]:
                    first_alpha = h["alphas"][0]
                    metrics = first_alpha.get("metrics", {})
                    sharpe = metrics.get("sharpe", 0)
                    returns = metrics.get("returns", 0)
                    drawdown = metrics.get("drawdown", 0)
                    turnover = metrics.get("turnover", 0)
                    fitness = metrics.get("fitness", 0)
                    expected = first_alpha.get("expected_improvement", '')
                else:
                    # Old format fallback
                    sharpe = h.get("metrics", {}).get("sharpe", 0)
                    returns = h.get("metrics", {}).get("returns", 0)
                    drawdown = h.get("metrics", {}).get("drawdown", 0)
                    turnover = h.get("metrics", {}).get("turnover", 0)
                    fitness = h.get("metrics", {}).get("fitness", 0)
                    expected = h.get("expected_improvement", '')

                history_summary.append({
                    "round": round_num,
                    "sharpe": sharpe,
                    "returns": returns,
                    "drawdown": drawdown,
                    "turnover": turnover,
                    "fitness": fitness,
                    "changes": changes_made,
                    "expected": expected
                })

            prompt = PromptTemplates.alpha_overall_reflection(history_summary, source_idea,
                                                              initial_sharpe, final_sharpe)
            messages = [{"role": "system", "content": "You are a senior quant director."},
                        {"role": "user", "content": prompt}]
            response = self.llm.client.chat(messages)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            return json.loads(match.group()) if match else None
        except Exception as e:
            print(f"[ImprovementAgent] Overall reflection failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            print(f"[ImprovementAgent] Parse error: {e}")
            return None

    def _generate_summary(self, improved: List[Dict], failed: List[Dict], all_alphas: List[Dict]) -> Dict:
        """Generate improvement summary"""
        total = len(all_alphas)
        success = len(improved)
        target = len([r for r in improved if r["status"] == "target_achieved"])
        avg_improvement = sum(r["sharpe_improvement"] for r in improved) / success if success else 0
        best_sharpe = max((r["final_sharpe"] for r in improved), default=0)

        return {
            "total_count": total, "success_count": success, "failed_count": len(failed),
            "target_achieved_count": target, "success_rate": success / total if total else 0,
            "avg_sharpe_improvement": avg_improvement, "best_final_sharpe": best_sharpe
        }

    def _validate_syntax(self, code: str) -> bool:
        """Validate expression syntax"""
        return bool(code and isinstance(code, str) and code.count("(") == code.count(")"))

    def _call_llm_for_validation_correction(self, expr: Dict, source_idea: str,
                                             validation_errors: List[Dict],
                                             valid_fields: set,
                                             retry_count: int) -> Optional[Dict]:
        """Call LLM to fix validation errors"""
        try:
            # Import at top level to avoid relative import issues
            from quant_factor_agents.infrastructure.prompt_templates import PromptTemplates

            expr_code = expr.get("code", "")
            region = expr.get("settings", {}).get("region", "USA")

            # Get last failed attempt
            last_error = validation_errors[-1]
            invalid_expr = last_error["expression"]
            invalid_settings = last_error["settings"]
            error_messages = [e["errors"] for e in validation_errors]

            messages = PromptTemplates.validation_error_correction(
                original_code=expr_code,
                invalid_expression=invalid_expr,
                invalid_settings=invalid_settings,
                validation_errors=error_messages,
                valid_fields=valid_fields,
                region=region,
                source_idea=source_idea,
                target_sharpe=self.target_sharpe
            )

            print(f"[ImprovementAgent] Validation retry {retry_count}, calling LLM for correction...")
            response = self.llm.client.chat(messages)
            return self._parse_llm_response(response)

        except Exception as e:
            print(f"[ImprovementAgent] Validation correction failed: {e}")
            return None

    def _save_validation_errors(self, expr: Dict, validation_errors: List[Dict],
                                 error_type: str = "validation"):
        """Save validation errors to output folder"""
        try:
            import os
            from datetime import datetime

            output_dir = "d:\\HuaweiMoveData\\Users\\10557\\Desktop\\test\\quant_factor_agents\\output"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{error_type}_errors_{expr.get('id', 'unknown')}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            error_data = {
                "timestamp": datetime.now().isoformat(),
                "alpha_id": expr.get("id", "unknown"),
                "original_expression": expr.get("code", ""),
                "original_settings": expr.get("settings", {}),
                "error_type": error_type,
                "validation_attempts": validation_errors
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

            print(f"[ImprovementAgent] Saved {error_type} errors to {filepath}")

        except Exception as e:
            print(f"[ImprovementAgent] Failed to save validation errors: {e}")

    def _save_loop_recovery_errors(self, alpha_info: Dict, loop_recovery_history: List[Dict]):
        """Save loop recovery errors to output folder"""
        try:
            import os
            from datetime import datetime

            output_dir = "d:\\HuaweiMoveData\\Users\\10557\\Desktop\\test\\quant_factor_agents\\output"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alpha_id = alpha_info.get("alpha_id", "unknown")
            filename = f"loop_recovery_errors_{alpha_id}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            error_data = {
                "timestamp": datetime.now().isoformat(),
                "alpha_id": alpha_id,
                "original_expression": alpha_info.get("expression", ""),
                "original_settings": alpha_info.get("settings", {}),
                "error_type": "loop_recovery_failed",
                "loop_recovery_attempts": loop_recovery_history,
                "total_attempts": len(loop_recovery_history)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

            print(f"[ImprovementAgent] Saved loop recovery errors to {filepath}")

        except Exception as e:
            print(f"[ImprovementAgent] Failed to save loop recovery errors: {e}")

    def _save_overall_reflection(self, alpha_info: Dict, improvement_history: List[Dict],
                                  overall_reflection: Optional[Dict], initial_sharpe: float,
                                  final_sharpe: float, final_status: str):
        """Save overall reflection to output folder"""
        try:
            import os
            from datetime import datetime

            output_dir = "d:\\HuaweiMoveData\\Users\\10557\\Desktop\\test\\quant_factor_agents\\output"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alpha_id = alpha_info.get("alpha_id", "unknown")
            filename = f"overall_reflection_{alpha_id}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            # Build improvement summary compatible with new history format
            improvement_summary = []
            for h in improvement_history:
                round_num = h.get("round", 0)
                changes = h.get("changes_made", "")

                # New format: each round has multiple alphas
                if "alphas" in h and h["alphas"]:
                    # Use the first (best) alpha for summary
                    best_alpha = h["alphas"][0]
                    # Collect all alpha IDs in this round
                    alpha_ids = [alpha.get("alpha_id", "") for alpha in h["alphas"]]
                    improvement_summary.append({
                        "round": round_num,
                        "expression": best_alpha.get("expression_code", "")[:100],
                        "sharpe": best_alpha.get("metrics", {}).get("sharpe", 0),
                        "changes": changes,
                        "alpha_count": len(h["alphas"]),
                        "alpha_ids": alpha_ids
                    })
                else:
                    # Old format fallback
                    improvement_summary.append({
                        "round": round_num,
                        "expression": h.get("expression_code", "")[:100],
                        "sharpe": h.get("metrics", {}).get("sharpe", 0),
                        "changes": changes,
                        "alpha_count": 1,
                        "alpha_ids": [h.get("alpha_id", "")]
                    })

            reflection_data = {
                "timestamp": datetime.now().isoformat(),
                "alpha_id": alpha_id,
                "source_idea": alpha_info.get("idea_content", ""),
                "original_expression": alpha_info.get("expression", ""),
                "final_status": final_status,
                "initial_sharpe": initial_sharpe,
                "final_sharpe": final_sharpe,
                "sharpe_improvement": final_sharpe - initial_sharpe,
                "total_rounds": len(improvement_history),
                "improvement_summary": improvement_summary,
                "overall_reflection": overall_reflection
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(reflection_data, f, indent=2, ensure_ascii=False)

            print(f"[ImprovementAgent] Saved overall reflection to {filepath}")

        except Exception as e:
            print(f"[ImprovementAgent] Failed to save overall reflection: {e}")
