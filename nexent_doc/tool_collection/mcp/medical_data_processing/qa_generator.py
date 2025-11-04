"""
医疗Q&A数据集生成模块
基于医疗数据使用大模型生成高质量的问答对.
"""

from typing import Dict, List, Any, Optional
import re
import json
from datetime import datetime
from nexent.core.models import OpenAIModel
from nexent.core.utils.observer import MessageObserver
from utils.config_utils import tenant_config_manager, get_model_name_from_config
from consts.const import MODEL_CONFIG_MAPPING
import logging

logger = logging.getLogger(__name__)

class MedicalQAGenerator:
    """医疗Q&A数据集生成器"""
    
    def __init__(self, tenant_id: str = None):
        self.tenant_id = tenant_id
        self.observer = MessageObserver()
        self.llm_model = None
    
    def _init_llm_model(self):
        """初始化大语言模型"""
        if self.llm_model is None:
            try:
                logger.info(f"开始初始化LLM模型，tenant_id: {self.tenant_id}")
                
                # 获取 LLM 模型配置
                llm_config = tenant_config_manager.get_model_config(
                    MODEL_CONFIG_MAPPING["llm"], 
                    tenant_id=self.tenant_id
                )
                
                if not llm_config:
                    logger.error(f"未找到租户 {self.tenant_id} 的 LLM 模型配置")
                    raise ValueError(f"未找到租户 {self.tenant_id} 的 LLM 模型配置")
                
                # 使用 get_model_name_from_config 函数正确组合模型名称
                model_name = get_model_name_from_config(llm_config)
                api_key = llm_config.get("api_key")
                base_url = llm_config.get("base_url")
                
                if not model_name:
                    raise ValueError("LLM 模型配置中缺少 model_name")
                if not api_key:
                    raise ValueError("LLM 模型配置中缺少 api_key")
                if not base_url:
                    raise ValueError("LLM 模型配置中缺少 base_url")
                
                # 初始化 LLM 模型
                self.llm_model = OpenAIModel(
                    observer=self.observer,
                    model_id=model_name,
                    api_base=base_url,
                    api_key=api_key,
                    temperature=0.7,
                    top_p=0.9,
                    frequency_penalty=0.3,
                    max_tokens=4096  # 修改为4096以符合API限制
                )
                logger.info(f"LLM模型初始化成功: {model_name}")
                
            except Exception as e:
                logger.error(f"LLM模型初始化失败: {e}")
                raise

    def generate_qa_from_content(self, content: str, qa_count: int = 10) -> Dict[str, Any]:
        """从医疗内容生成Q&A数据集，简化算法
        
        规则：
        - 3000个字生成3个高质量问答
        - 不足3000字生成1个问答
        - 不考虑qa_count参数
        
        Args:
            content: 医疗文本内容
            qa_count: 忽略此参数
        
        Returns:
            包含Q&A数据集的字典
        """
        try:
            logger.info(f"开始从内容生成Q&A，内容长度: {len(content)}")
            
            # 初始化LLM模型
            self._init_llm_model()
            
            # 简化的分块处理
            all_qa_pairs = []
            chunk_size = 3000
            
            # 按3000字符分块
            chunks = []
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                chunks.append(chunk)
            
            # 处理每个块
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                
                # 根据块长度决定生成的Q&A数量
                if chunk_length >= 3000:
                    qa_count_for_chunk = 3  # 3000字或以上生成3个Q&A
                else:
                    qa_count_for_chunk = 1  # 不足3000字生成1个Q&A
                
                logger.info(f"处理第 {i+1}/{len(chunks)} 个块，长度: {chunk_length}，生成 {qa_count_for_chunk} 个Q&A")
                
                # 为当前块生成Q&A
                qa_pairs = self._generate_qa_for_chunk(chunk, qa_count_for_chunk)
                all_qa_pairs.extend(qa_pairs)
            
            # 生成统计信息
            stats = self._generate_dataset_stats(all_qa_pairs)
            quality_metrics = self._calculate_quality_metrics(all_qa_pairs)
            
            result = {
                'success': True,
                'qa_pairs': all_qa_pairs,
                'statistics': stats,
                'quality_metrics': quality_metrics,
                'dataset_info': {
                    'source': 'content_generation',
                    'content_length': len(content),
                    'total_chunks': len(chunks),
                    'generated_count': len(all_qa_pairs),
                    'generation_method': 'simplified_chunked',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Q&A生成成功，共处理 {len(chunks)} 个块，生成 {len(all_qa_pairs)} 个Q&A对")
            return result
            
        except Exception as e:
            logger.error(f"从内容生成Q&A失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'qa_pairs': [],
                'statistics': {},
                'quality_metrics': {}
            }

    def _generate_qa_for_chunk(self, chunk: str, qa_count: int) -> List[Dict[str, Any]]:
        """为单个内容块生成Q&A"""
        try:
            # 构建针对块的提示词
            prompt = self._build_chunk_qa_generation_prompt(chunk, qa_count)
            
            messages = [
                {"role": "system", "content": "你是一位专业的医疗教育专家，擅长从医疗文本中生成高质量的问答对。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_model(messages)
            ai_response = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"AI响应长度: {len(ai_response)}")
            
            # 解析AI生成的Q&A结果
            parsed_result = self._parse_ai_qa_result(ai_response)
            
            # 验证Q&A对
            qa_pairs = parsed_result.get('qa_pairs', [])
            validated_pairs = []
            
            for pair in qa_pairs:
                if pair.get('question') and pair.get('answer'):
                    validated_pair = {
                        'id': pair.get('id', f"qa_{len(validated_pairs) + 1}"),
                        'question': pair.get('question', '').strip(),
                        'answer': pair.get('answer', '').strip(),
                        'question_type': pair.get('question_type', 'general'),
                        'difficulty': pair.get('difficulty', 'medium'),
                        'keywords': pair.get('keywords', []),
                        'quality_score': pair.get('quality_score', 0.8)
                    }
                    validated_pairs.append(validated_pair)
            
            return validated_pairs
            
        except Exception as e:
            logger.error(f"块Q&A生成失败: {e}")
            return []

    def _build_chunk_qa_generation_prompt(self, content: str, qa_count: int) -> str:
        """构建针对内容块的Q&A生成提示词"""
        prompt = f"""请基于以下医疗文本内容，生成 {qa_count} 个高质量的医疗问答对。

医疗文本内容：
{content}

要求：
1. 问题应该涵盖文本中的关键医疗信息
2. 答案必须基于提供的文本内容，准确且详细
3. 问题类型应该多样化，包括：定义类、症状类、诊断类、治疗类、病理类
4. 确保医学术语的准确性和专业性
5. 必须生成完整的 {qa_count} 个Q&A对

**严格的JSON格式要求：**

你必须返回一个完全有效的JSON对象，格式如下：

{{
    "qa_pairs": [
        {{
            "id": "qa_1",
            "question": "具体问题内容",
            "answer": "详细专业答案",
            "question_type": "定义类",
            "difficulty": "中等",
            "keywords": ["关键词1", "关键词2"],
            "quality_score": 0.85
        }}
    ]
}}

**格式要求：**
- 所有键名和字符串值都必须用双引号
- 不允许尾随逗号
- 不允许单引号
- 数字不用引号
- 必须返回完整的JSON，不要添加任何解释文字

现在请生成JSON格式的医疗问答对："""
        
        return prompt

    def _parse_ai_qa_result(self, ai_response: str) -> Dict[str, Any]:
        """解析AI生成的Q&A结果"""
        try:
            # 打印AI返回的原始内容
            logger.info("=" * 50)
            logger.info("AI返回的原始JSON内容:")
            logger.info(ai_response)
            logger.info("=" * 50)
            
            # 清理响应内容
            cleaned_response = self._clean_ai_response(ai_response)
            
            # 打印清理后的内容
            logger.info("清理后的JSON内容:")
            logger.info(cleaned_response)
            logger.info("=" * 30)
            
            # 首先尝试直接解析JSON
            if cleaned_response.strip().startswith('{'):
                return json.loads(cleaned_response)
            
            # 如果不是直接的JSON，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.info(f"提取的JSON字符串: {json_str}")
                
                # 尝试修复常见的JSON格式问题
                fixed_json = self._fix_json_format(json_str)
                logger.info(f"修复后的JSON: {fixed_json}")
                
                return json.loads(fixed_json)
            
            # 如果无法解析，尝试从文本中提取结构化信息
            logger.warning("无法解析AI返回的Q&A JSON结果，尝试文本解析")
            return self._extract_qa_from_text(cleaned_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Q&A JSON解析失败: {e}")
            logger.error(f"错误位置: line {e.lineno}, column {e.colno}")
            logger.error(f"错误的JSON内容片段: {e.doc[max(0, e.pos-50):e.pos+50] if hasattr(e, 'doc') and e.doc else 'N/A'}")
            
            # 尝试修复JSON格式后再次解析
            try:
                if 'json_str' in locals():
                    fixed_json = self._fix_json_format(json_str)
                    logger.info(f"尝试修复后的JSON: {fixed_json[:200]}...")
                    return json.loads(fixed_json)
                elif 'cleaned_response' in locals():
                    # 尝试修复清理后的响应
                    fixed_json = self._fix_json_format(cleaned_response)
                    logger.info(f"尝试修复清理后的JSON: {fixed_json[:200]}...")
                    return json.loads(fixed_json)
            except Exception as fix_error:
                logger.error(f"JSON修复也失败: {fix_error}")
            
            # 最后尝试从文本中提取
            logger.info("使用文本提取方法作为备用方案")
            return self._extract_qa_from_text(ai_response)
        except Exception as e:
            logger.error(f"解析AI响应时发生未知错误: {e}")
            return self._extract_qa_from_text(ai_response)

    def _clean_ai_response(self, response: str) -> str:
        """清理AI响应内容"""
        # 移除markdown代码块标记
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # 移除可能的前后解释文字
        lines = response.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # 找到JSON开始位置
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                start_idx = i
                break
        
        # 找到JSON结束位置
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().endswith('}'):
                end_idx = i + 1
                break
        
        return '\n'.join(lines[start_idx:end_idx])

    def _fix_json_format(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        # 移除BOM和特殊字符
        json_str = json_str.encode('utf-8').decode('utf-8-sig')
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        # 修复无引号的键名（包括中文键名）
        json_str = re.sub(r'(\s*)([a-zA-Z_\u4e00-\u9fff][a-zA-Z0-9_\u4e00-\u9fff]*)\s*:', r'\1"\2":', json_str)
        
        # 修复单引号为双引号
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # 修复数组中没有引号的中文字符串值
        # 匹配数组中的中文字符串（不在引号内的）
        def fix_unquoted_chinese_in_array(match):
            content = match.group(1)
            # 查找没有引号的中文字符串
            content = re.sub(r'(?<=[,\[])\s*([^\s",\[\]{}]+[\u4e00-\u9fff][^\s",\[\]{}]*)\s*(?=[,\]])', r' "\1"', content)
            content = re.sub(r'(?<=[,\[])\s*([\u4e00-\u9fff][^\s",\[\]{}]*)\s*(?=[,\]])', r' "\1"', content)
            return '[' + content + ']'
        
        json_str = re.sub(r'\[([^\[\]]*)\]', fix_unquoted_chinese_in_array, json_str)
        
        # 修复对象中没有引号的值（特别是中文）
        # 匹配冒号后面没有引号的中文或字母数字值
        json_str = re.sub(r':\s*([^\s",\[\]{}]+[\u4e00-\u9fff][^\s",\[\]{}]*)\s*(?=[,}\]])', r': "\1"', json_str)
        json_str = re.sub(r':\s*([\u4e00-\u9fff][^\s",\[\]{}]*)\s*(?=[,}\]])', r': "\1"', json_str)
        
        # 移除尾随逗号
        json_str = re.sub(r',(\s*[\]\}])', r'\1', json_str)
        
        # 处理不完整的JSON（如果以逗号结尾但没有闭合括号）
        json_str = json_str.strip()
        if json_str.endswith(','):
            json_str = json_str[:-1]
        
        # 确保JSON结构完整
        if json_str.startswith('{') and not json_str.endswith('}'):
            # 尝试找到最后一个完整的对象
            last_complete = json_str.rfind('}}')
            if last_complete != -1:
                json_str = json_str[:last_complete + 2] + ']}'
            else:
                json_str += ']}'
        
        return json_str

    def _extract_qa_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取Q&A对作为备用方案"""
        try:
            qa_pairs = []
            
            # 使用正则表达式提取问答对
            # 匹配问题模式
            question_patterns = [
                r'问题?\s*[:：]\s*(.+?)(?=答案?[:：]|$)',
                r'Q\s*[:：]\s*(.+?)(?=A[:：]|$)',
                r'(\d+)\.\s*(.+?)(?=答案?[:：]|$)'
            ]
            
            # 匹配答案模式
            answer_patterns = [
                r'答案?\s*[:：]\s*(.+?)(?=问题?[:：]|Q[:：]|\d+\.|$)',
                r'A\s*[:：]\s*(.+?)(?=问题?[:：]|Q[:：]|\d+\.|$)'
            ]
            
            # 尝试提取结构化的问答对
            for i, q_pattern in enumerate(question_patterns):
                questions = re.findall(q_pattern, text, re.DOTALL | re.IGNORECASE)
                if questions:
                    for j, a_pattern in enumerate(answer_patterns):
                        answers = re.findall(a_pattern, text, re.DOTALL | re.IGNORECASE)
                        if answers and len(answers) >= len(questions):
                            for k, question in enumerate(questions):
                                if k < len(answers):
                                    qa_pair = {
                                        'id': f'qa_{len(qa_pairs) + 1}',
                                        'question': question.strip(),
                                        'answer': answers[k].strip(),
                                        'question_type': 'extracted',
                                        'difficulty': 'medium',
                                        'keywords': [],
                                        'quality_score': 0.6
                                    }
                                    qa_pairs.append(qa_pair)
                            break
                    if qa_pairs:
                        break
            
            # 如果没有找到结构化的问答对，生成基本的问答对
            if not qa_pairs:
                sentences = [s.strip() for s in text.split('。') if len(s.strip()) > 20][:5]
                for i, sentence in enumerate(sentences):
                    qa_pair = {
                        'id': f'qa_{i + 1}',
                        'question': f'请解释以下内容：{sentence[:30]}...',
                        'answer': sentence,
                        'question_type': 'general',
                        'difficulty': 'medium',
                        'keywords': [],
                        'quality_score': 0.5
                    }
                    qa_pairs.append(qa_pair)
            
            return {'qa_pairs': qa_pairs}
            
        except Exception as e:
            logger.error(f"文本提取Q&A失败: {e}")
            return {'qa_pairs': []}

    def _generate_dataset_stats(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据集统计"""
        if not qa_pairs:
            return {}
        
        # 难度分布
        difficulty_dist = {}
        for pair in qa_pairs:
            diff = pair.get('difficulty', 'medium')
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        # 问题类型分布
        type_dist = {}
        for pair in qa_pairs:
            q_type = pair.get('question_type', 'general')
            type_dist[q_type] = type_dist.get(q_type, 0) + 1
        
        # 质量分布
        quality_scores = [pair.get('quality_score', 0) for pair in qa_pairs]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'difficulty_distribution': difficulty_dist,
            'question_type_distribution': type_dist,
            'average_quality_score': round(avg_quality, 3),
            'total_keywords': len(set(kw for pair in qa_pairs for kw in pair.get('keywords', []))),
            'average_question_length': sum(len(pair.get('question', '')) for pair in qa_pairs) / len(qa_pairs),
            'average_answer_length': sum(len(pair.get('answer', '')) for pair in qa_pairs) / len(qa_pairs)
        }
    
    def _calculate_quality_metrics(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算质量指标"""
        if not qa_pairs:
            return {}
        
        # 完整性指标
        complete_pairs = sum(1 for pair in qa_pairs if pair.get('question') and pair.get('answer'))
        completeness = complete_pairs / len(qa_pairs)
        
        # 多样性指标
        unique_questions = len(set(pair.get('question', '') for pair in qa_pairs))
        diversity = unique_questions / len(qa_pairs)
        
        # 专业性指标
        medical_terms_count = sum(len(pair.get('keywords', [])) for pair in qa_pairs)
        professionalism = min(medical_terms_count / (len(qa_pairs) * 5), 1.0)
        
        return {
            'completeness': round(completeness, 3),
            'diversity': round(diversity, 3),
            'professionalism': round(professionalism, 3),
            'overall_quality': round((completeness + diversity + professionalism) / 3, 3)
        }
