"""
医疗知识库集成模块123
用于将Q&A数据集存储到Elasticsearch知识库
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
import sys
import os
import re

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from nexent.vector_database.elasticsearch_core import ElasticSearchCore
from services.elasticsearch_service import get_embedding_model  # 添加这个导入

import os
from consts.const import ES_HOST, ES_API_KEY, ES_PASSWORD  # 添加导入

# 参考case_analyzer.py的导入模式
from utils.config_utils import tenant_config_manager, get_model_name_from_config
from consts.const import MODEL_CONFIG_MAPPING

logger = logging.getLogger(__name__)

class MedicalKnowledgeBaseIntegrator:
    """医疗知识库集成器"""
    
    def __init__(self, es_host: str = None, es_api_key: str = None, tenant_id: str = None):
        """
        初始化医疗知识库集成器
        
        Args:
            es_host: Elasticsearch主机地址
            es_api_key: Elasticsearch API密钥
            tenant_id: 租户ID，用于获取已注册的向量模型
        """
        try:
            # 使用项目的 Elasticsearch 配置
            es_host = es_host or ES_HOST or "http://localhost:9210"
            es_api_key = es_api_key or ES_API_KEY
            
            logger.info(f"[DEBUG] Elasticsearch 连接配置: host={es_host}, api_key={'***' if es_api_key else 'None'}")
            
            # 初始化Elasticsearch核心
            self.es_core = ElasticSearchCore(
                host=es_host,
                api_key=es_api_key
            )
            
            # 设置分块参数
            self.max_chunk_tokens = 1000
            
            # 初始化嵌入模型相关属性
            self.embedding_model = None
            self.embedding_dim = None
            self.tenant_id = tenant_id
            
            if tenant_id:
                logger.info(f"[DEBUG] 开始为租户 {tenant_id} 配置嵌入模型...")
                self._init_embedding_model()
            else:
                logger.warning("[WARNING] 未提供tenant_id，无法获取已注册的向量模型")
            
            logger.info(f"医疗知识库集成器初始化成功，连接到: {es_host}")
            
        except Exception as e:
            logger.error(f"医疗知识库集成器初始化失败: {e}")
            raise
    
    def _init_embedding_model(self):
        """初始化嵌入模型 - 参考case_analyzer.py的模式"""
        if self.embedding_model is None:
            try:
                # 添加调试日志 - 检查 tenant_id
                logger.info(f"[DEBUG] 开始初始化嵌入模型，tenant_id: {self.tenant_id}")
                
                # 获取嵌入模型配置
                embedding_config = tenant_config_manager.get_model_config(
                    MODEL_CONFIG_MAPPING["embedding"], 
                    tenant_id=self.tenant_id
                )
                
                # 添加调试日志 - 检查配置获取结果
                logger.info(f"[DEBUG] 获取到的嵌入模型配置: {embedding_config}")
                if embedding_config:
                    logger.info(f"[DEBUG] 配置内容: {list(embedding_config.keys()) if isinstance(embedding_config, dict) else 'NOT_DICT'}")
                
                # 使用 get_embedding_model 函数获取嵌入模型实例
                embedding_info = get_embedding_model(self.tenant_id)
                
                # 修复：检查正确的属性名称
                if embedding_info and (hasattr(embedding_info, 'model') or hasattr(embedding_info, 'model_name')):
                    # 创建一个包装器来添加 embedding_model_name 属性
                    class EmbeddingModelWrapper:
                        def __init__(self, original_model):
                            self._original = original_model
                            # 复制所有原始属性
                            for attr in dir(original_model):
                                if not attr.startswith('_'):
                                    setattr(self, attr, getattr(original_model, attr))
                            
                            # 添加 embedding_model_name 属性以兼容 elasticsearch_core
                            if hasattr(original_model, 'model'):
                                self.embedding_model_name = original_model.model
                            elif hasattr(original_model, 'model_name'):
                                self.embedding_model_name = original_model.model_name
                            else:
                                self.embedding_model_name = 'unknown'
                        
                        def __getattr__(self, name):
                            # 如果属性不存在，尝试从原始对象获取
                            return getattr(self._original, name)
                    
                    self.embedding_model = EmbeddingModelWrapper(embedding_info)
                    self.embedding_dim = getattr(embedding_info, 'embedding_dim', 1536)
                    
                    # 获取模型名称（兼容不同的属性名）
                    model_name = getattr(embedding_info, 'model', None) or getattr(embedding_info, 'model_name', 'Unknown')
                    
                    logger.info(f"[DEBUG] 嵌入模型初始化成功: {model_name}, 维度: {self.embedding_dim}")
                    logger.info(f"[DEBUG] 嵌入模型对象类型: {type(embedding_info).__name__}")
                    logger.info(f"[DEBUG] 包装器添加了 embedding_model_name 属性: {self.embedding_model.embedding_model_name}")
                    print(f"[SUCCESS] 使用已注册的向量模型: {model_name}, 维度: {self.embedding_dim}")
                else:
                    logger.error(f"[DEBUG] get_embedding_model 返回无效结果: {embedding_info}")
                    logger.error(f"[WARNING] 无法初始化嵌入模型，使用默认配置")
                
            except Exception as e:
                logger.error(f"[DEBUG] 嵌入模型初始化失败: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数量
        """
        # 简单估算：中文字符约1.5个token，英文单词约1个token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - len(''.join(re.findall(r'[a-zA-Z]+', text)))
        
        return int(chinese_chars * 1.5 + english_words + other_chars * 0.5)
    
    def _chunk_text(self, text: str, max_tokens: int = 1000, enable_chunking: bool = True, overlap_tokens: int = 0) -> List[str]:
        """
        将长文本分块处理
        
        Args:
            text: 输入文本
            max_tokens: 每个分块的最大token数
            enable_chunking: 是否启用分块
            overlap_tokens: 分块之间的重叠token数
            
        Returns:
            文本分块列表
        """
        if not text or not text.strip():
            return []
        
        # 简单估算token数量（中文约1字符=1token，英文约4字符=1token）
        estimated_tokens = len(text)
        total_tokens = int(estimated_tokens * 0.8)  # 保守估计
        
        if not enable_chunking or total_tokens <= max_tokens:
            return [text]
        
        print(f"开始文本分块：总长度={len(text)}字符，估算token={total_tokens}，最大分块={max_tokens}token")
        
        chunks = []
        sentences = re.split(r'[。！？；\n]', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence = sentence.strip() + "。"
            sentence_tokens = int(len(sentence) * 0.8)
            
            # 如果单个句子就超过最大token，直接作为一个分块
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                chunks.append(sentence)
                continue
            
            # 如果加上当前句子会超过最大token，先保存当前分块
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # 添加最后一个分块
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 如果没有生成任何分块，返回原文本
        if not chunks:
            chunks = [text]
        
        print(f"文本分块完成：原文{total_tokens}token -> {len(chunks)}个分块")
        return chunks
    
    def create_medical_index(self, index_name: str, embedding_dim: int = None) -> Dict[str, Any]:
        """创建医疗知识库索引"""
        try:
            # 确保索引名称符合医疗知识库命名规范
            if not index_name.startswith('medical_'):
                original_name = index_name
                index_name = f"medical_{index_name}"
                print(f"自动为索引添加医疗前缀: '{original_name}' -> '{index_name}'")
            
            # 创建向量索引
            success = self.es_core.create_vector_index(index_name, embedding_dim=self.embedding_dim)
            
            if success:
                result = {
                    'success': True,
                    'index_name': index_name,
                    'message': f'医疗知识库索引 {index_name} 创建成功',
                    'created_at': datetime.now().isoformat(),
                    'has_medical_prefix': True
                }
                logger.info(f"医疗知识库索引创建成功: {index_name}")
            else:
                result = {
                    'success': False,
                    'error': f'索引 {index_name} 创建失败',
                    'index_name': index_name
                }
                logger.error(f"医疗知识库索引创建失败: {index_name}")
            
            return result
            
        except Exception as e:
            print(f"医疗知识库索引创建失败: {index_name}")
            print(f"创建医疗知识库索引失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name if 'index_name' in locals() else 'unknown'
            }
    
    def _normalize_index_name(self, index_name: str) -> str:
        """
        规范化索引名称，确保符合Elasticsearch要求
        
        Args:
            index_name: 原始索引名称
            
        Returns:
            规范化后的索引名称
        """
        import re
        import urllib.parse
        
        # URL解码（处理%E8%82%BA%E7%99%8C这样的编码）
        try:
            decoded_name = urllib.parse.unquote(index_name)
        except:
            decoded_name = index_name
        
        # 转换为小写
        normalized = decoded_name.lower()
        
        # 替换特殊字符为下划线或删除
        # Elasticsearch索引名称规则：只能包含小写字母、数字、-、_、+
        normalized = re.sub(r'[^a-z0-9\-_+]', '_', normalized)
        
        # 移除连续的下划线
        normalized = re.sub(r'_+', '_', normalized)
        
        # 移除开头和结尾的下划线
        normalized = normalized.strip('_')
        
        # 确保不为空
        if not normalized:
            normalized = "medical_knowledge_base"
        
        # 添加医疗前缀（如果没有的话）
        if not normalized.startswith('medical_'):
            normalized = f"medical_{normalized}"
        
        print(f"[DEBUG] 索引名称规范化: '{index_name}' -> '{normalized}'")
        return normalized

    def integrate_qa_dataset(self, qa_dataset: Any, index_name: str, 
                           enable_chunking: bool = True, chunk_size: int = 500) -> Dict[str, Any]:
        """集成Q&A数据集到医疗知识库"""
        try:
            # 规范化索引名称
            normalized_index_name = self._normalize_index_name(index_name)
            print(f"开始集成医疗Q&A数据到索引: {normalized_index_name}")
            
            # 确保嵌入模型已初始化
            if self.embedding_model is None:
                print("[DEBUG] 嵌入模型未初始化，尝试重新初始化...")
                self._init_embedding_model()
            
            # 动态检测嵌入模型的实际维度
            print(f"[DEBUG] 开始检测嵌入模型实际维度...")
            try:
                test_text = "测试嵌入维度"
                test_embedding = self.embedding_model.get_embeddings(test_text)
                if test_embedding and len(test_embedding) > 0:
                    actual_dim = len(test_embedding[0])
                    print(f"[DEBUG] 检测到实际嵌入维度: {actual_dim}")
                    
                    # 如果实际维度与配置的维度不一致，更新配置
                    if actual_dim != self.embedding_dim:
                        print(f"[WARNING] 维度不匹配！配置维度: {self.embedding_dim}, 实际维度: {actual_dim}")
                        print(f"[INFO] 自动更新嵌入维度为: {actual_dim}")
                        self.embedding_dim = actual_dim
                    else:
                        print(f"[SUCCESS] 维度匹配确认: {actual_dim}")
                else:
                    print(f"[ERROR] 嵌入模型测试失败，无法获取有效的嵌入向量")
                    return {
                        'success': False,
                        'error': '嵌入模型测试失败，无法获取有效的嵌入向量',
                        'indexed_count': 0
                    }
            except Exception as embed_error:
                print(f"[ERROR] 嵌入模型维度检测失败: {embed_error}")
                return {
                    'success': False,
                    'error': f'嵌入模型维度检测失败: {str(embed_error)}',
                    'indexed_count': 0
                }
            
            # 添加嵌入模型状态检查
            print(f"[DEBUG] 嵌入模型状态: {type(self.embedding_model).__name__}")
            print(f"[DEBUG] 最终使用的嵌入维度: {self.embedding_dim}")
            
            # 创建向量索引（使用检测到的正确维度）
            print(f"[DEBUG] 使用维度 {self.embedding_dim} 创建索引 {normalized_index_name}")
            index_created = self.es_core.create_vector_index(normalized_index_name, embedding_dim=self.embedding_dim)
            print(f"[DEBUG] 索引创建结果: {index_created}")
            
            # 如果索引创建失败，检查是否是因为索引已存在
            if not index_created:
                print(f"[WARNING] 索引创建失败，检查索引是否已存在...")
                # 可以选择删除现有索引并重新创建，或者跳过创建步骤
                try:
                    # 检查索引是否存在
                    index_exists = self.es_core.check_index_exists(normalized_index_name)
                    if index_exists:
                        print(f"[INFO] 索引 {normalized_index_name} 已存在，将直接使用现有索引")
                        # 可以选择验证现有索引的维度设置
                    else:
                        print(f"[ERROR] 索引不存在且创建失败")
                        return {
                            'success': False,
                            'error': f'索引 {normalized_index_name} 创建失败',
                            'indexed_count': 0
                        }
                except Exception as check_error:
                    print(f"[ERROR] 检查索引存在性失败: {check_error}")
            
            # 准备文档数据
            print(f"[DEBUG] 开始处理输入数据，类型: {type(qa_dataset)}")
            documents = self._process_qa_dataset(qa_dataset, enable_chunking=True)
            
            if not documents:
                print("[ERROR] 没有有效的文档数据可以索引")
                print(f"[DEBUG] 输入数据详情: {qa_dataset}")
                return {
                    'success': False,
                    'error': '没有有效的文档数据可以索引',
                    'indexed_count': 0,
                    'input_data_type': str(type(qa_dataset)),
                    'input_data_sample': str(qa_dataset)[:500] if qa_dataset else 'None'
                }
            
            print(f"[DEBUG] 准备索引 {len(documents)} 个文档到 {normalized_index_name}")
            print(f"[DEBUG] 文档示例: {documents[0] if documents else 'None'}")
            
            # 测试嵌入模型是否正常工作
            try:
                test_text = "测试文本"
                test_embedding = self.embedding_model.get_embeddings(test_text)
                print(f"[DEBUG] 嵌入模型测试成功，返回维度: {len(test_embedding[0]) if test_embedding and len(test_embedding) > 0 else 'None'}")
            except Exception as embed_error:
                print(f"[ERROR] 嵌入模型测试失败: {embed_error}")
                return {
                    'success': False,
                    'error': f'嵌入模型测试失败: {str(embed_error)}',
                    'indexed_count': 0
                }
            
            # 使用正确的参数顺序调用 index_documents
            try:
                indexed_count = self.es_core.index_documents(
                    index_name=normalized_index_name,
                    embedding_model=self.embedding_model,
                    documents=documents
                )
                
                print(f"[DEBUG] elasticsearch_core 返回的 indexed_count: {indexed_count}")
                
                if indexed_count == 0:
                    # 提供更详细的错误信息
                    print("[ERROR] 文档索引失败，indexed_count为0")
                    print(f"[DEBUG] 检查文档格式: {documents[0] if documents else 'None'}")
                    print(f"[DEBUG] 检查索引是否存在")
                    
                    # 检查索引是否真的存在
                    try:
                        indices = self.es_core.get_user_indices(f"{normalized_index_name}*")
                        print(f"[DEBUG] 匹配的索引: {indices}")
                    except Exception as idx_error:
                        print(f"[DEBUG] 检查索引时出错: {idx_error}")
                    
                    raise Exception("文档索引失败，indexed_count为0。可能原因：1) 文档格式不正确 2) Elasticsearch连接问题 3) 嵌入模型API调用失败")
                
            except Exception as index_error:
                print(f"[ERROR] 调用 index_documents 时出错: {index_error}")
                return {
                    'success': False,
                    'error': f'索引文档时出错: {str(index_error)}',
                    'indexed_count': 0
                }
            
            return {
                'success': True,
                'message': f'成功将 {indexed_count} 个Q&A文档集成到医疗知识库 {normalized_index_name}',
                'index_name': normalized_index_name,
                'indexed_count': indexed_count,
                'documents_processed': len(documents)
            }
            
        except Exception as e:
            print(f"医疗Q&A数据集成失败: {e}")
            import traceback
            print(f"[DEBUG] 完整错误堆栈: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f'医疗Q&A数据集成失败: {str(e)}',
                'index_name': normalized_index_name if 'normalized_index_name' in locals() else index_name
            }
    
    def _process_qa_dataset(self, qa_dataset: Any, enable_chunking: bool = True, 
                      chunk_size: int = 500, max_chunk_tokens: int = None) -> List[Dict[str, Any]]:
        """处理Q&A数据集，转换为可索引的文档格式"""
        # 如果没有传入 max_chunk_tokens，使用实例属性
        if max_chunk_tokens is None:
            max_chunk_tokens = getattr(self, 'max_chunk_tokens', 1000)
        documents = []
        
        try:
            print(f"输入数据类型: {type(qa_dataset)}")
            print(f"输入数据长度: {len(qa_dataset) if isinstance(qa_dataset, (list, dict, str)) else 'N/A'}")
            
            # 如果qa_dataset是字符串，尝试解析为JSON
            if isinstance(qa_dataset, str):
                try:
                    qa_dataset = json.loads(qa_dataset)
                    print(f"成功解析JSON字符串，解析后类型: {type(qa_dataset)}")
                except json.JSONDecodeError as e:
                    print(f"无法解析JSON字符串: {e}")
                    # 如果无法解析JSON，检查是否是简单的字符串内容
                    if qa_dataset.strip():
                        print("将字符串内容作为单个文档处理")
                        doc = {
                            'content': qa_dataset,
                            'metadata': {
                                'source': 'text_content',
                                'created_at': datetime.now().isoformat()
                            }
                        }
                        documents.append(doc)
                    return documents
            
            # 如果qa_dataset是字典，需要进一步处理
            if isinstance(qa_dataset, dict):
                # 检查是否是错误响应
                if 'success' in qa_dataset and qa_dataset.get('success') is False:
                    print(f"输入数据是错误响应: {qa_dataset.get('error', '未知错误')}")
                    print(f"建议: {qa_dataset.get('suggestion', '无建议')}")
                    return documents
                
                # 尝试提取Q&A数据的多种路径
                if 'qa_pairs' in qa_dataset:
                    qa_dataset = qa_dataset['qa_pairs']
                    print(f"提取qa_pairs字段，类型: {type(qa_dataset)}, 长度: {len(qa_dataset) if isinstance(qa_dataset, (list, dict)) else 'N/A'}")
                elif 'final_output' in qa_dataset and isinstance(qa_dataset['final_output'], dict):
                    final_output = qa_dataset['final_output']
                    if 'qa_dataset' in final_output:
                        qa_dataset = final_output['qa_dataset']
                        print(f"从final_output.qa_dataset提取数据，类型: {type(qa_dataset)}, 长度: {len(qa_dataset) if isinstance(qa_dataset, (list, dict)) else 'N/A'}")
                    elif 'qa_pairs' in final_output:
                        qa_dataset = final_output['qa_pairs']
                        print(f"从final_output.qa_pairs提取数据，类型: {type(qa_dataset)}, 长度: {len(qa_dataset) if isinstance(qa_dataset, (list, dict)) else 'N/A'}")
                    else:
                        print(f"final_output中缺少qa_dataset或qa_pairs字段。可用字段: {list(final_output.keys())}")
                        return documents
                elif 'pipeline_steps' in qa_dataset:
                    # 这是医疗数据工程流水线的输出，尝试从pipeline_steps中提取Q&A数据
                    print("检测到医疗数据工程流水线输出，尝试从pipeline_steps中提取Q&A数据")
                    qa_result = None
                    pipeline_steps = qa_dataset.get('pipeline_steps', [])
                    
                    # 确保pipeline_steps是列表
                    if not isinstance(pipeline_steps, list):
                        print(f"pipeline_steps不是列表类型: {type(pipeline_steps)}")
                        return documents
                    
                    for step in pipeline_steps:
                        if isinstance(step, dict) and step.get('step') == 'qa_generation' and step.get('status') == 'completed':
                            qa_result = step.get('result', {})
                            break
                    
                    if qa_result and isinstance(qa_result, dict) and 'qa_pairs' in qa_result:
                        qa_dataset = qa_result['qa_pairs']
                        print(f"从pipeline_steps.qa_generation.result.qa_pairs提取数据，类型: {type(qa_dataset)}, 长度: {len(qa_dataset) if isinstance(qa_dataset, (list, dict)) else 'N/A'}")
                    else:
                        print("无法从pipeline_steps中找到有效的Q&A数据")
                        return documents
                else:
                    print(f"字典格式不正确，缺少qa_pairs、final_output或pipeline_steps字段。可用字段: {list(qa_dataset.keys())}")
                    return documents
            
            # 确保qa_dataset是列表
            if not isinstance(qa_dataset, list):
                print(f"Q&A数据集必须是列表类型，当前类型: {type(qa_dataset)}")
                # 如果是字符串，尝试作为单个内容处理
                if isinstance(qa_dataset, str) and qa_dataset.strip():
                    doc = {
                        'content': qa_dataset,
                        'metadata': {
                            'source': 'text_content',
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    documents.append(doc)
                return documents
            
            if len(qa_dataset) == 0:
                print("Q&A数据集为空列表")
                return documents
            
            print(f"开始处理 {len(qa_dataset)} 个Q&A对，分块设置: {enable_chunking}")
            
            # 处理每个Q&A对
            for i, qa_pair in enumerate(qa_dataset):
                try:
                    # 如果qa_pair是字符串，尝试解析为JSON
                    if isinstance(qa_pair, str):
                        try:
                            qa_pair = json.loads(qa_pair)
                        except json.JSONDecodeError as e:
                            print(f"无法解析第 {i+1} 个Q&A对的JSON: {e}")
                            # 将字符串作为内容处理
                            if qa_pair.strip():
                                doc = {
                                    'content': qa_pair,
                                    'metadata': {
                                        'qa_pair_id': i + 1,
                                        'source': 'text_content',
                                        'created_at': datetime.now().isoformat()
                                    }
                                }
                                documents.append(doc)
                            continue
                    
                    # 确保qa_pair是字典
                    if not isinstance(qa_pair, dict):
                        print(f"第 {i+1} 个Q&A对不是字典格式: {type(qa_pair)}")
                        continue
                    
                    # 提取问题和答案
                    question = qa_pair.get('question', '').strip()
                    answer = qa_pair.get('answer', '').strip()
                    
                    if not question or not answer:
                        print(f"第 {i+1} 个Q&A对缺少问题或答案: question='{question}', answer='{answer}'")
                        continue
                    
                    # 检查答案长度，决定是否需要分块
                    answer_tokens = self._estimate_tokens(answer)
                    if enable_chunking and answer_tokens > max_chunk_tokens:
                        print(f"第 {i+1} 个Q&A对的答案过长({answer_tokens} tokens)，进行分块处理")
                        
                        # 对答案进行分块
                        chunks = self._chunk_text(answer, max_tokens=max_chunk_tokens, overlap_tokens=50)
                        
                        # 为每个分块创建文档
                        for j, chunk in enumerate(chunks):
                            doc = {
                                'content': f"问题: {question}\n答案: {chunk}",
                                'metadata': {
                                    'question': question,
                                    'answer_chunk': chunk,
                                    'full_answer': answer,
                                    'qa_pair_id': i + 1,
                                    'chunk_id': j + 1,
                                    'total_chunks': len(chunks),
                                    'is_chunked': True,
                                    'source': 'medical_qa_dataset',
                                    'created_at': datetime.now().isoformat()
                                }
                            }
                            documents.append(doc)
                        
                        print(f"第 {i+1} 个Q&A对分块完成：{len(chunks)}个分块")
                    else:
                        # 不需要分块，直接创建文档
                        doc = {
                            'content': f"问题: {question}\n答案: {answer}",
                            'metadata': {
                                'question': question,
                                'answer': answer,
                                'qa_pair_id': i + 1,
                                'is_chunked': False,
                                'source': 'medical_qa_dataset',
                                'created_at': datetime.now().isoformat()
                            }
                        }
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"处理第 {i+1} 个Q&A对时出错: {e}")
                    continue
        
        except Exception as e:
            print(f"处理Q&A数据集时发生错误: {e}")
            return documents
        
        print(f"Q&A数据集处理完成，生成 {len(documents)} 个文档")
        return documents

    def get_medical_indices(self) -> Dict[str, Any]:
        """获取所有医疗知识库索引列表"""
        try:
            # 获取所有用户索引
            all_indices = self.es_core.get_user_indices()
            
            # 过滤出医疗相关的索引（以medical_开头或包含医疗相关关键词）
            medical_indices = []
            medical_keywords = ['medical_', '医疗', 'health', 'clinical', 'pathology', 'diagnosis']
            
            for index in all_indices:
                # 检查索引名是否包含医疗相关关键词
                if any(keyword in index.lower() for keyword in medical_keywords):
                    medical_indices.append(index)
            
            return {
                'success': True,
                'indices': medical_indices,
                'total_count': len(medical_indices),
                'all_indices_count': len(all_indices),
                'message': f'找到 {len(medical_indices)} 个医疗知识库索引'
            }
            
        except Exception as e:
            print(f"获取医疗知识库索引列表失败: {e}")
            return {
                'success': False,
                'error': f'获取医疗知识库索引列表失败: {str(e)}',
                'indices': [],
                'total_count': 0
            }
    
    def search_medical_knowledge(self, index_name: str, query: str, 
                               search_type: str = "hybrid", top_k: int = 10) -> Dict[str, Any]:
        """搜索医疗知识库"""
        try:
            # 规范化索引名称
            normalized_index_name = self._normalize_index_name(index_name)
            print(f"搜索医疗知识库: {normalized_index_name}, 查询: {query}")
            
            # 确保嵌入模型已初始化
            if self.embedding_model is None:
                self._init_embedding_model()
            
            if not self.embedding_model and search_type in ["semantic", "hybrid"]:
                return {
                    'success': False,
                    'error': '语义搜索需要嵌入模型，但模型未初始化',
                    'results': []
                }
            
            # 根据搜索类型执行搜索
            if search_type == "accurate":
                results = self.es_core.accurate_search(
                    index_names=[normalized_index_name],
                    query_text=query,
                    top_k=top_k
                )
            elif search_type == "semantic":
                results = self.es_core.semantic_search(
                    index_names=[normalized_index_name],
                    query_text=query,
                    embedding_model=self.embedding_model,
                    top_k=top_k
                )
            elif search_type == "hybrid":
                results = self.es_core.hybrid_search(
                    index_names=[normalized_index_name],
                    query_text=query,
                    embedding_model=self.embedding_model,
                    top_k=top_k
                )
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")
            
            return {
                'success': True,
                'results': results,
                'query': query,
                'search_type': search_type,
                'index_name': normalized_index_name,
                'total_results': len(results)
            }
            
        except Exception as e:
            print(f"医疗知识库搜索失败: {e}")
            return {
                'success': False,
                'error': f'医疗知识库搜索失败: {str(e)}',
                'results': [],
                'query': query,
                'search_type': search_type
            }
    

        """删除医疗知识库索引"""
        try:
            # 规范化索引名称
            normalized_index_name = self._normalize_index_name(index_name)
            print(f"删除医疗知识库索引: {normalized_index_name}")
            
            # 删除索引
            success = self.es_core.delete_index(normalized_index_name)
            
            if success:
                return {
                    'success': True,
                    'message': f'医疗知识库索引 {normalized_index_name} 删除成功',
                    'index_name': normalized_index_name
                }
            else:
                return {
                    'success': False,
                    'error': f'索引 {normalized_index_name} 删除失败',
                    'index_name': normalized_index_name
                }
                
        except Exception as e:
            print(f"删除医疗知识库索引失败: {e}")
            return {
                'success': False,
                'error': f'删除医疗知识库索引失败: {str(e)}',
                'index_name': normalized_index_name if 'normalized_index_name' in locals() else index_name
            }