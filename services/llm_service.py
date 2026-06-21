"""
LLM服务层
为Agent提供高阶LLM功能，包括Prompt模板、结果解析等
"""
from typing import Dict, Any, List, Optional
from infrastructure.llm_client import BaseLLMClient, LLMClientFactory


class LLMService:
    """LLM服务"""
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        初始化
        
        Args:
            llm_client: LLM客户端实例
        """
        self.client = llm_client
    
    # ==================== Idea生成相关 ====================
    
    def generate_ideas(self, class_list: List[str], op_list: List[str], 
                       num: int = 4, history: List[str] = None) -> List[Dict[str, Any]]:
        """
        生成因子想法
        
        Args:
            class_list: 数据集分类列表
            op_list: 操作符列表
            num: 生成数量
            history: 历史想法（用于避免重复）
            
        Returns:
            想法列表
        """
        history_str = "\n".join([f"- {h}" for h in (history or [])[-10:]])  # 最近10个
        
        prompt = f"""你是一位量化投资专家，请基于以下信息生成{num}个创新的量化因子想法。

可用数据集分类：
{', '.join(class_list)}

可用操作符：
{', '.join(op_list)}

{'历史想法（请避免重复）：' if history_str else ''}
{history_str}

要求：
1. 每个想法要有明确的经济学逻辑
2. 说明使用的数据集分类和操作符
3. 描述预期的市场规律
4. 想法要多样化，不要重复

请以JSON格式输出：
{{
    "ideas": [
        {{
            "content": "想法描述",
            "use_class": ["使用的分类"],
            "use_op": ["使用的操作符"],
            "logic": "经济学逻辑解释"
        }}
    ]
}}"""
        
        try:
            response = self.client.complete(prompt, max_tokens=2000)
            return self._parse_ideas_response(response, num)
        except Exception as e:
            print(f"[LLMService] 生成想法失败: {e}")
            # 返回默认想法
            return self._generate_default_ideas(class_list, op_list, num)
    
    def _parse_ideas_response(self, response: str, expected_num: int) -> List[Dict[str, Any]]:
        """解析想法响应"""
        import json
        import re
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                ideas = data.get("ideas", [])
                # 确保格式正确
                for idea in ideas:
                    idea.setdefault("use_class", [])
                    idea.setdefault("use_op", [])
                    idea.setdefault("logic", "")
                return ideas[:expected_num]
        except Exception as e:
            print(f"[LLMService] 解析想法失败: {e}")
        
        return []
    
    def _generate_default_ideas(self, class_list: List[str], op_list: List[str], num: int) -> List[Dict[str, Any]]:
        """生成默认想法（LLM失败时使用）"""
        import random
        
        ideas = []
        templates = [
            "基于{class1}的{op1}计算动量因子",
            "使用{op1}和{op2}构建{class1}和{class2}的价差策略",
            "通过{op1}识别{class1}的均值回归机会",
            "结合{class1}和{class2}的{op1}复合因子",
        ]
        
        for i in range(num):
            template = templates[i % len(templates)]
            class1 = random.choice(class_list)
            class2 = random.choice(class_list)
            op1 = random.choice(op_list)
            op2 = random.choice(op_list)
            
            content = template.format(class1=class1, class2=class2, op1=op1, op2=op2)
            
            ideas.append({
                "content": content,
                "use_class": list(set([class1, class2])),
                "use_op": list(set([op1, op2])),
                "logic": "基于历史数据的技术分析"
            })
        
        return ideas
    
    # ==================== 表达式生成相关 ====================
    
    def generate_expression(self, idea: Dict[str, Any], field_mapping: Dict[str, List[str]],
                           op_config: Dict[str, Any]) -> Optional[str]:
        """
        根据想法生成表达式代码
        
        Args:
            idea: 想法字典
            field_mapping: 字段映射
            op_config: 操作符配置
            
        Returns:
            表达式代码
        """
        # 获取可用的字段和操作符
        available_fields = []
        for cls in idea.get("use_class", []):
            available_fields.extend(field_mapping.get(cls, []))
        
        available_ops = idea.get("use_op", [])
        
        prompt = f"""请根据以下想法生成量化因子表达式代码。

想法描述：
{idea.get('content', '')}

经济学逻辑：
{idea.get('logic', '')}

可用字段：
{', '.join(available_fields)}

可用操作符：
{', '.join(available_ops)}

操作符说明：
- ts_mean(field, window): 时间序列均值
- ts_std(field, window): 时间序列标准差
- ts_zscore(field, window): Z-score标准化
- ts_rank(field, window): 排名
- ts_corr(field1, field2, window): 相关性

要求：
1. 使用FASTEXPR语法
2. 表达式要简洁有效
3. 只输出表达式代码，不要解释

表达式："""
        
        try:
            response = self.client.complete(prompt, max_tokens=500)
            # 清理响应
            expr = response.strip()
            # 移除可能的代码块标记
            if expr.startswith("```"):
                expr = expr[3:]
            if expr.endswith("```"):
                expr = expr[:-3]
            return expr.strip()
        except Exception as e:
            print(f"[LLMService] 生成表达式失败: {e}")
            return None
    
    # ==================== 改进相关 ====================
    
    def improve_expression(self, expr_code: str, defect_reason: str, 
                          strategy: str = "auto") -> Optional[str]:
        """
        改进表达式
        
        Args:
            expr_code: 原表达式
            defect_reason: 缺陷原因
            strategy: 改进策略
            
        Returns:
            改进后的表达式
        """
        strategy_prompts = {
            "replace_op": "尝试替换操作符，如将ts_mean换成ts_median",
            "adjust_param": "调整参数，如改变窗口大小",
            "simplify": "简化表达式结构，减少嵌套",
            "auto": "根据缺陷原因选择合适的改进策略"
        }
        
        prompt = f"""请改进以下量化因子表达式。

原表达式：
{expr_code}

缺陷原因：
{defect_reason}

改进策略：
{strategy_prompts.get(strategy, strategy_prompts['auto'])}

要求：
1. 保持经济学逻辑不变
2. 解决上述缺陷
3. 只输出改进后的表达式代码

改进后的表达式："""
        
        try:
            response = self.client.complete(prompt, max_tokens=500)
            expr = response.strip()
            if expr.startswith("```"):
                expr = expr[3:]
            if expr.endswith("```"):
                expr = expr[:-3]
            return expr.strip()
        except Exception as e:
            print(f"[LLMService] 改进表达式失败: {e}")
            return None

    # ==================== 分类相关 ====================
    
    def classify_expression(self, expr_code: str) -> Dict[str, Any]:
        """
        分析表达式特征
        
        Args:
            expr_code: 表达式代码
            
        Returns:
            分类结果
        """
        prompt = f"""请分析以下量化因子表达式的特征。

表达式：
{expr_code}

请分析：
1. 因子类型（动量/均值回归/波动率/价值等）
2. 复杂度（简单/中等/复杂）
3. 可能的风险（过拟合/相关性高等）

以JSON格式输出：
{{
    "factor_type": "因子类型",
    "complexity": "简单/中等/复杂",
    "risks": ["风险1", "风险2"],
    "description": "简要描述"
}}"""
        
        try:
            response = self.client.complete(prompt, max_tokens=800)
            import json
            import re

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[LLMService] 分类表达式失败: {e}")

        return {
            "factor_type": "未知",
            "complexity": "中等",
            "risks": [],
            "description": "无法分析"
        }


class LLMServiceFactory:
    """LLM服务工厂"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LLMService:
        """
        从配置创建LLM服务
        
        Args:
            config: 配置字典，包含llm配置
            
        Returns:
            LLM服务实例
        """
        llm_config = config.get("llm", {"provider": "mock"})
        client = LLMClientFactory.create(llm_config)
        return LLMService(client)
