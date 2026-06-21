"""
大模型客户端
支持多种LLM提供商：OpenAI、Azure、本地模型等
"""
import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话接口
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数（temperature, max_tokens等）
            
        Returns:
            模型回复文本
        """
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        补全接口
        
        Args:
            prompt: 提示词
            **kwargs: 额外参数
            
        Returns:
            补全文本
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4", temperature: float = 0.7):
        """
        初始化
        
        Args:
            api_key: API密钥，默认从环境变量OPENAI_API_KEY读取
            base_url: 基础URL，用于兼容其他OpenAI格式API
            model: 模型名称
            temperature: 温度参数，从配置文件读取
        """
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        self.base_url = base_url or "https://api.deepseek.com"
        self.model = model
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")
        
        # 延迟导入，避免强制依赖
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("请安装openai包: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话接口"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            # Only set max_tokens if explicitly provided
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[OpenAIClient] 调用失败: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全接口"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI客户端"""
    
    def __init__(self, api_key: str = None, endpoint: str = None, 
                 deployment_name: str = None, api_version: str = "2024-02-01",
                 temperature: float = 0.7):
        """
        初始化
        
        Args:
            api_key: API密钥
            endpoint: Azure端点
            deployment_name: 部署名称
            api_version: API版本
            temperature: 温度参数，从配置文件读取
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version
        self.temperature = temperature
        
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError("Azure OpenAI配置不完整")
        
        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version
            )
        except ImportError:
            raise ImportError("请安装openai包: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话接口"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 2000)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[AzureOpenAIClient] 调用失败: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全接口"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class LocalLLMClient(BaseLLMClient):
    """本地模型客户端（支持vLLM、TGI等）"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None, temperature: float = 0.7):
        """
        初始化
        
        Args:
            base_url: 本地模型服务地址
            model: 模型名称（可选）
            temperature: 温度参数，从配置文件读取
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="not-needed")
        except ImportError:
            raise ImportError("请安装openai包: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话接口"""
        try:
            params = {
                "model": self.model or "local-model",
                "messages": messages,
                "temperature": self.temperature
            }
            # Only set max_tokens if explicitly provided
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LocalLLMClient] 调用失败: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全接口"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端（用于测试）"""
    
    def __init__(self, response_template: str = "模拟回复: {prompt}"):
        """
        初始化
        
        Args:
            response_template: 回复模板
        """
        self.response_template = response_template
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """模拟对话"""
        last_message = messages[-1]["content"] if messages else ""
        return self.response_template.format(prompt=last_message[:50])
    
    def complete(self, prompt: str, **kwargs) -> str:
        """模拟补全"""
        return self.response_template.format(prompt=prompt[:50])


class LLMClientFactory:
    """LLM客户端工厂"""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseLLMClient:
        """
        创建LLM客户端
        
        Args:
            config: 配置字典
                {
                    "provider": "openai" | "azure" | "local" | "mock",
                    "api_key": "...",
                    "model": "gpt-4",
                    ...
                }
        
        Returns:
            LLM客户端实例
        """
        provider = config.get("provider", "mock").lower()
        
        if provider == "openai":
            return OpenAIClient(
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                model=config.get("model", "gpt-4"),
                temperature=config.get("temperature", 0.7)
            )
        elif provider == "azure":
            return AzureOpenAIClient(
                api_key=config.get("api_key"),
                endpoint=config.get("endpoint"),
                deployment_name=config.get("deployment_name"),
                api_version=config.get("api_version", "2024-02-01"),
                temperature=config.get("temperature", 0.7)
            )
        elif provider == "local":
            return LocalLLMClient(
                base_url=config.get("base_url", "http://localhost:8000/v1"),
                model=config.get("model"),
                temperature=config.get("temperature", 0.7)
            )
        elif provider == "mock":
            return MockLLMClient(config.get("response_template"))
        else:
            raise ValueError(f"未知的LLM提供商: {provider}")
