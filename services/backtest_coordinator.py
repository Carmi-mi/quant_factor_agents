"""
Backtest Coordinator Service
Manages parallel backtesting using thread pool
Each thread has its own BRAIN API session
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import time

from agents.backtest_agent import BacktestAgent
from services.llm_service import LLMService
from infrastructure.ace_lib import start_session


class BacktestCoordinator:
    """
    Coordinator for parallel backtesting
    
    Features:
    - Thread pool for parallel execution
    - Each thread has independent BRAIN API session
    - Configurable max workers (max 4)
    - Immediate callback to downstream agent
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_service: Optional[LLMService] = None,
        max_workers: int = 4
    ):
        """
        Initialize coordinator
        
        Args:
            config: Configuration dictionary with brain credentials and backtest settings
            llm_service: Optional LLM service for result analysis
            max_workers: Maximum number of parallel workers (1-4)
        """
        self.config = config
        self.llm_service = llm_service
        
        # Get max_workers from config if available, otherwise use parameter
        config_max_workers = config.get("backtest", {}).get("max_workers", max_workers)
        
        # Validate and correct max_workers (must be 1-4)
        if config_max_workers < 1:
            print(f"[BacktestCoordinator] max_workers ({config_max_workers}) < 1, correcting to 1")
            self.max_workers = 1
        elif config_max_workers > 4:
            print(f"[BacktestCoordinator] max_workers ({config_max_workers}) > 4, correcting to 4")
            self.max_workers = 4
        else:
            self.max_workers = config_max_workers
        
        # Downstream callback
        self.downstream_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # BRAIN API session for correlation check
        self.correlation_session = None
        
        # Statistics
        self.stats = {
            "total_ideas": 0,
            "completed": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Thread lock for stats update
        self.stats_lock = threading.Lock()
    
    def _init_correlation_session(self):
        """Initialize BRAIN API session for correlation check"""
        if self.correlation_session is None:
            self.correlation_session = start_session()
        return self.correlation_session is not None
    
    def set_downstream_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for downstream agent
        
        Args:
            callback: Function to call with each idea's result
                     callback(idea_result: dict) -> None
        """
        self.downstream_callback = callback
    
    def run_parallel_backtest(
        self,
        ideas: List[Dict[str, Any]],
        file_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run parallel backtest on multiple ideas
        
        Args:
            ideas: List of ideas, each with idea_id, idea_content, expressions
            file_info: Optional file information
            
        Returns:
            Combined results from all ideas
        """
        if not ideas:
            return {
                "status": "success",
                "file_info": file_info or {},
                "final_alphas": []
            }
        
        self.stats["total_ideas"] = len(ideas)
        self.stats["completed"] = 0
        self.stats["failed"] = 0
        self.stats["start_time"] = time.time()
        
        print(f"[BacktestCoordinator] Starting parallel backtest")
        print(f"[BacktestCoordinator] Total ideas: {len(ideas)}")
        print(f"[BacktestCoordinator] Max workers: {self.max_workers}")
        
        # Results collection
        idea_results = []
        
        # Use thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idea = {
                executor.submit(self._process_single_idea, idea, file_info): idea
                for idea in ideas
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_idea):
                idea = future_to_idea[future]
                idea_id = idea.get("idea_id", "unknown")
                
                try:
                    result = future.result()
                    
                    if result:
                        idea_results.append(result)
                        
                        # Update stats
                        with self.stats_lock:
                            self.stats["completed"] += 1
                        
                        print(f"[BacktestCoordinator] Idea '{idea_id}' completed "
                              f"({self.stats['completed']}/{self.stats['total_ideas']})")
                    else:
                        with self.stats_lock:
                            self.stats["failed"] += 1
                        print(f"[BacktestCoordinator] Idea '{idea_id}' failed")
                        
                except Exception as e:
                    with self.stats_lock:
                        self.stats["failed"] += 1
                    print(f"[BacktestCoordinator] Idea '{idea_id}' error: {e}")
        
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        print(f"[BacktestCoordinator] All ideas completed in {duration:.2f}s")
        print(f"[BacktestCoordinator] Success: {self.stats['completed']}, Failed: {self.stats['failed']}")
        
        # Collect qualified results (already filtered by BacktestAgent)
        qualified_results = self._collect_qualified_results(idea_results)
        
        # Sort alphas within each idea for correlation detection
        for idea_result in qualified_results:
            idea_result["alphas"] = self._sort_alphas_for_correlation(
                idea_result["alphas"]
            )
            # Extract sorted alpha_ids for correlation detection
            idea_result["sorted_alpha_ids"] = [
                a["alpha_id"] for a in idea_result["alphas"]
            ]
            
            # Run correlation check
            if len(idea_result["sorted_alpha_ids"]) >= 2:
                # Initialize session if needed
                if self._init_correlation_session():
                    print(f"[BacktestCoordinator] Running correlation check for idea '{idea_result['idea_id']}'")
                    from services.correlation_checker import run as check_correlation
                    kept_alphas = check_correlation(
                        idea_result["sorted_alpha_ids"],
                        session=self.correlation_session,
                        years_of_data=4,
                        correlation_threshold=0.7
                    )
                    idea_result["kept_alpha_ids"] = kept_alphas
                    print(f"[BacktestCoordinator] Correlation check: {len(kept_alphas)}/{len(idea_result['sorted_alpha_ids'])} alphas kept")
                else:
                    print(f"[BacktestCoordinator] Skipping correlation check for idea '{idea_result['idea_id']}' (no session)")
        
        # Build final result - only kept alphas after correlation check
        final_alphas = []
        for idea_result in qualified_results:
            # Get kept alpha IDs (after correlation check)
            kept_ids = idea_result.get("kept_alpha_ids", idea_result.get("sorted_alpha_ids", []))
            
            # Filter alphas to only include kept ones
            kept_alphas = [
                alpha for alpha in idea_result["alphas"]
                if alpha["alpha_id"] in kept_ids
            ]
            
            if kept_alphas:
                # 从第一个 alpha 获取 settings（所有 alpha 的 settings 相同）
                settings = kept_alphas[0].get("settings", {}) if kept_alphas else {}
                
                # 移除每个 alpha 中的 settings，减少冗余
                alphas_without_settings = []
                for alpha in kept_alphas:
                    alpha_copy = {k: v for k, v in alpha.items() if k != "settings"}
                    alphas_without_settings.append(alpha_copy)
                
                final_alphas.append({
                    "idea_id": idea_result["idea_id"],
                    "idea_content": idea_result["idea_content"],
                    "kept_count": len(kept_alphas),
                    "settings": settings,
                    "alphas": alphas_without_settings
                })
        
        final_result = {
              "status": "success",
              "file_info": file_info or {},
              "final_alphas": final_alphas
          }
        
        # Save results to file
        self._save_results(final_result, file_info)
        
        return final_result
    
    def _collect_qualified_results(self, idea_results: List[Dict]) -> List[Dict]:
        """
        收集 BacktestAgent 已筛选的 qualified alphas
        筛选已在 BacktestAgent 中完成，这里只需收集和聚合
        
        Args:
            idea_results: 所有 idea 的回测结果（已由 BacktestAgent 处理和筛选）
            
        Returns:
            按 idea 分组的符合条件的 alpha 列表
        """
        qualified_ideas = []
        total_qualified = 0
        
        for idea_result in idea_results:
            idea_id = idea_result.get("idea_id", "unknown")
            idea_content = idea_result.get("idea_content", "")
            
            # 检查是否有错误，有错误则跳过该 idea
            if idea_result.get("error"):
                print(f"[BacktestCoordinator] Idea '{idea_id}' has error: {idea_result.get('error_type', 'Unknown')}, skipping")
                continue
            
            # 直接使用 BacktestAgent 已筛选的 alphas
            alphas = idea_result.get("alphas", [])
            
            # 如果当前 idea 有符合条件的 alphas，添加到结果中
            if alphas:
                qualified_ideas.append({
                    "idea_id": idea_id,
                    "idea_content": idea_content,
                    "qualified_count": len(alphas),
                    "alphas": alphas
                })
                total_qualified += len(alphas)
        
        print(f"[BacktestCoordinator] Collected {total_qualified} qualified alphas from {len(qualified_ideas)} ideas (filtered by Agent)")
        return qualified_ideas
    
    def _sort_alphas_for_correlation(self, alphas: List[Dict]) -> List[Dict]:
        """
        对 alpha 进行排序，用于相关性检测
        
        排序规则：
        1. sharpe > 0 的排在前面，按 sharpe 降序（大的在前）
        2. sharpe <= 0 的排在后面，按 sharpe 升序（最负的在前）
        
        Args:
            alphas: alpha 列表
            
        Returns:
            排序后的 alpha 列表
        """
        if not alphas:
            return []
        
        # 分离正负 sharpe
        positive = [a for a in alphas if a.get("sharpe", 0) > 0]
        negative = [a for a in alphas if a.get("sharpe", 0) <= 0]
        
        # 正 sharpe：降序（大的在前）
        positive_sorted = sorted(positive, key=lambda x: x.get("sharpe", 0), reverse=True)
        
        # 负 sharpe：升序（最负的在前）
        negative_sorted = sorted(negative, key=lambda x: x.get("sharpe", 0))
        
        # 合并：正的在前面，负的在后面
        return positive_sorted + negative_sorted
    
    def _save_results(self, results: Dict[str, Any], file_info: Optional[Dict[str, Any]]):
        """
        Save backtest results to output directory
        
        Args:
            results: Final results dictionary
            file_info: File information for naming
        """
        try:
            # Build output directory path
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to extract region/universe/dataset from file_info
            region = file_info.get("region", "UNKNOWN") if file_info else "UNKNOWN"
            universe = file_info.get("universe", "UNKNOWN") if file_info else "UNKNOWN"
            dataset = file_info.get("dataset", "UNKNOWN") if file_info else "UNKNOWN"
            data_type = file_info.get("type", "MATRIX") if file_info else "MATRIX"
            
            filename = f"backtest_results_{region}_{universe}_{dataset}_{data_type}.json"
            output_path = output_dir / filename
            
            # Convert results to JSON-serializable format
            serializable_results = self._make_serializable(results)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"[BacktestCoordinator] Results saved to: {output_path}")
            
        except Exception as e:
            print(f"[BacktestCoordinator] Warning: Failed to save results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        import math
        
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
    
    def _process_single_idea(self, idea: Dict[str, Any], file_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Process a single idea in a worker thread
        Each thread has its own BRAIN API session
        
        Args:
            idea: Single idea with idea_id, idea_content, expressions
            file_info: Optional file information
            
        Returns:
            Raw idea result from BacktestAgent.run() or None if failed
        """
        idea_id = idea.get("idea_id", "unknown")
        thread_name = threading.current_thread().name
        
        print(f"[BacktestCoordinator][{thread_name}] Processing idea '{idea_id}'")
        
        try:
            # Create agent config with credentials
            agent_config = self.config.copy()
            
            # Create BacktestAgent (will login with its own session)
            agent = BacktestAgent(agent_config, self.llm_service)
            
            # Run backtest on single idea
            idea_result = agent.run(idea, file_info=file_info)
            
            if idea_result and idea_result.get("idea_id"):
                # Immediately send to downstream via callback
                if self.downstream_callback:
                    try:
                        print(f"[BacktestCoordinator][{thread_name}] "
                              f"Sending idea '{idea_id}' to downstream agent")
                        self.downstream_callback(idea_result)
                    except Exception as e:
                        print(f"[BacktestCoordinator][{thread_name}] "
                              f"Downstream callback error: {e}")
                
                # Return raw result directly
                return idea_result
            else:
                print(f"[BacktestCoordinator][{thread_name}] Idea '{idea_id}' backtest failed")
                return None
                
        except Exception as e:
            print(f"[BacktestCoordinator][{thread_name}] Error processing idea '{idea_id}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        if stats["start_time"] and not stats["end_time"]:
            stats["elapsed_seconds"] = time.time() - stats["start_time"]
        return stats


# Convenience function for simple usage
def run_parallel_backtest(
    ideas: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_service: Optional[LLMService] = None,
    max_workers: int = 4,
    downstream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    file_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run parallel backtest
    
    Args:
        ideas: List of ideas to backtest
        config: Configuration with brain credentials
        llm_service: Optional LLM service
        max_workers: Maximum parallel workers (1-4)
        downstream_callback: Optional callback for downstream agent
        file_info: Optional file information
        
    Returns:
        Combined backtest results
    """
    coordinator = BacktestCoordinator(config, llm_service, max_workers)
    
    if downstream_callback:
        coordinator.set_downstream_callback(downstream_callback)
    
    return coordinator.run_parallel_backtest(ideas, file_info)
