"""
Logger Agent
Records execution results and statistics
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from core.agent_base import Agent


class LoggerAgent(Agent):
    """Logging agent for recording execution results"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("Logger", config)
        self.log_dir = config.get("logging", {}).get("log_dir", "logs")
        self.level = config.get("logging", {}).get("level", "INFO")
        self.save_good = config.get("logging", {}).get("save_good", True)
        self.save_bad = config.get("logging", {}).get("save_bad", True)
        self.save_rounds = config.get("logging", {}).get("save_rounds", True)
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log files
        self.good_factors_file = os.path.join(self.log_dir, "good_factors.jsonl")
        self.bad_factors_file = os.path.join(self.log_dir, "bad_factors.jsonl")
        self.rounds_file = os.path.join(self.log_dir, "rounds.jsonl")
        self.system_log_file = os.path.join(self.log_dir, "system.log")
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log execution results
        
        Args:
            input_data: {
                "type": "good|bad|round|system",
                "data": {...}
            }
            
        Returns:
            Logging status
        """
        log_type = input_data.get("type", "system")
        data = input_data.get("data", {})
        
        # Add timestamp
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": log_type,
            **data
        }
        
        if log_type == "good" and self.save_good:
            self._append_to_file(self.good_factors_file, log_entry)
            print(f"[Logger] Saved good factor: {data.get('expression_id', '')}")
            
        elif log_type == "bad" and self.save_bad:
            self._append_to_file(self.bad_factors_file, log_entry)
            print(f"[Logger] Saved bad factor: {data.get('expression_id', '')}")
            
        elif log_type == "round" and self.save_rounds:
            self._append_to_file(self.rounds_file, log_entry)
            print(f"[Logger] Saved round {data.get('round', 0)} statistics")
            
        elif log_type == "system":
            self._write_system_log(data.get("message", ""), data.get("level", "INFO"))
        
        return {
            "status": "success",
            "logged": True,
            "type": log_type
        }
    
    def _append_to_file(self, filepath: str, data: Dict[str, Any]):
        """Append JSON line to file"""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def _write_system_log(self, message: str, level: str = "INFO"):
        """Write system log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.system_log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
    
    def log_factor(self, factor: Dict[str, Any], category: str):
        """Convenience method to log a factor"""
        return self.run({
            "type": category,
            "data": factor
        })
    
    def log_round(self, round_num: int, stats: Dict[str, Any]):
        """Convenience method to log round statistics"""
        return self.run({
            "type": "round",
            "data": {
                "round": round_num,
                **stats
            }
        })
    
    def log_system(self, message: str, level: str = "INFO"):
        """Convenience method to log system message"""
        return self.run({
            "type": "system",
            "data": {
                "message": message,
                "level": level
            }
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "good_factors": self._count_lines(self.good_factors_file),
            "bad_factors": self._count_lines(self.bad_factors_file),
            "rounds_logged": self._count_lines(self.rounds_file),
            "log_dir": self.log_dir
        }
        return stats
    
    def _count_lines(self, filepath: str) -> int:
        """Count lines in file"""
        if not os.path.exists(filepath):
            return 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
