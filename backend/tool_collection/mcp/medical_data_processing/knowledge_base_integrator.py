"""
医疗知识库集成模块
将处理后的医疗Q&A数据存储到Elasticsearch知识库
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from nexent.vector_database.elasticsearch_core import ElasticSearchCore

logger = logging.getLogger(__name__)

class MedicalKnowledgeBaseIntegrator:
    """医疗知识库集成器"""
    
    def __init__(self, es_host: str = None, es_api_key: str = None):
        """初始化知识库集成器
        
        Args:
            es_host: Elasticsearch主机地址
            es_api_key: Elasticsearch API密钥
        """
        try:
            # 从环境变量或参数获取Elasticsearch配置
            host = es_host or os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200')
            # 优先使用ELASTICSEARCH_API_KEY，然后是ELASTIC_PASSWORD
            api_key = es_api_key or os.getenv('ELASTICSEARCH_API_KEY') or os.getenv('ELASTIC_PASSWORD')
            
            # 验证必要参数
            if not host:
                raise ValueError("Elasticsearch主机地址未配置。请提供es_host参数或设置ELASTICSEARCH_HOST环境变量。")
            
            if not api_key:
                raise ValueError("Elasticsearch API密钥未配置。请提供es_api_key参数或设置ELASTICSEARCH_API_KEY/ELASTIC_PASSWORD环境变量。")
            
            # 打印调试信息（生产环境中应该移除）
            logger.info(f"使用Elasticsearch配置: host={host}, api_key={'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'None'}")
            
            # 初始化Elasticsearch核心服务
            self.es_core = ElasticSearchCore(
                host=host,
                api_key=api_key,
                verify_certs=False,
                ssl_show_warn=False
            )
            
            # 医疗知识库索引前缀
            self.medical_index_prefix = "medical_pathology_kb"
            
            logger.info(f"医疗知识库集成器初始化成功，连接到: {host}")
            
        except Exception as e:
            logger.error(f"医疗知识库集成器初始化失败: {e}")
            raise
    
    def create_medical_knowledge_index(self, index_name: str = None) -> Dict[str, Any]:
        """创建医疗知识库索引
        
        Args:
            index_name: 索引名称，默认使用时间戳
            
        Returns:
            创建结果
        """
        try:
            if not index_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                index_name = f"{self.medical_index_prefix}_{timestamp}"
            
            # 创建向量索引
            success = self.es_core.create_vector_index(index_name)
            
            if success:
                result = {
                    'success': True,
                    'index_name': index_name,
                    'message': f'医疗知识库索引 {index_name} 创建成功',
                    'created_at': datetime.now().isoformat()
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
            logger.error(f"创建医疗知识库索引失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name or 'unknown'
            }
    
    def integrate_qa_dataset(self, qa_dataset: List[Dict[str, Any]], index_name: str, batch_size: int = 100) -> Dict[str, Any]:
        """集成医疗Q&A数据集到知识库
        
        Args:
            qa_dataset: Q&A数据集
            index_name: 索引名称
            batch_size: 批处理大小
            
        Returns:
            集成结果
        """
        try:
            # 确保索引名称为小写
            index_name = index_name.lower()
            logger.info(f"开始集成医疗Q&A数据集到索引: {index_name}")
            
            if not qa_dataset:
                return {
                    'success': False,
                    'error': 'Q&A数据集为空',
                    'total_processed': 0
                }
            
            # 转换Q&A数据为Elasticsearch文档格式
            documents = self._convert_qa_to_documents(qa_dataset)
            
            # 直接使用bulk索引，不使用嵌入模型
            return self._direct_bulk_index(documents, index_name)
            
        except Exception as e:
            logger.error(f"医疗Q&A数据集集成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_processed': 0,
                'index_name': index_name
            }
    
    def _direct_bulk_index(self, documents: List[Dict[str, Any]], index_name: str) -> Dict[str, Any]:
        """直接使用Elasticsearch bulk API索引文档（不使用嵌入）
        
        Args:
            documents: 要索引的文档列表
            index_name: 索引名称
            
        Returns:
            索引结果
        """
        try:
            # 确保索引名称为小写
            index_name = index_name.lower()
            logger.info(f"使用小写索引名称: {index_name}")
            
            # 检查文档列表是否为空
            if not documents:
                logger.warning("文档列表为空，跳过索引操作")
                return {
                    'success': False,
                    'error': '没有有效的文档可以索引',
                    'total_processed': 0,
                    'index_name': index_name
                }
            
            # 准备bulk操作
            operations = []
            for doc in documents:
                operations.append({"index": {"_index": index_name}})
                # 移除embedding字段，因为我们不生成嵌入
                doc_copy = doc.copy()
                if "embedding" in doc_copy:
                    del doc_copy["embedding"]
                operations.append(doc_copy)
            
            # 检查operations是否为空
            if not operations:
                logger.warning("Bulk操作列表为空")
                return {
                    'success': False,
                    'error': 'Bulk操作列表为空',
                    'total_processed': 0,
                    'index_name': index_name
                }
            
            # 执行bulk索引
            response = self.es_core.client.bulk(
                index=index_name,
                operations=operations,
                refresh='wait_for'
            )
            
            # 检查错误
            if response.get('errors'):
                errors = []
                for item in response.get('items', []):
                    if 'index' in item and 'error' in item['index']:
                        errors.append(item['index']['error'])
                if errors:
                    logger.error(f"Bulk索引部分失败: {errors[:5]}")  # 只显示前5个错误
            
            total_indexed = len([item for item in response.get('items', []) 
                               if 'index' in item and item['index'].get('status') in [200, 201]])
            
            logger.info(f"直接bulk索引完成: {total_indexed} 个文档已索引")
            
            return {
                'success': True,
                'total_processed': len(documents) // 2,  # 因为每个Q&A对生成2个文档
                'total_documents_created': len(documents),
                'total_indexed': total_indexed,
                'index_name': index_name,
                'integration_time': datetime.now().isoformat(),
                'method': 'direct_bulk_index'
            }
            
        except Exception as e:
            logger.error(f"直接bulk索引失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_processed': 0,
                'index_name': index_name
            }
    
    def _convert_qa_to_documents(self, qa_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将Q&A数据转换为Elasticsearch文档格式
        
        Args:
            qa_dataset: Q&A数据集
            
        Returns:
            Elasticsearch文档列表
        """
        documents = []
        
        for i, qa_pair in enumerate(qa_dataset):
            try:
                # 检查qa_pair是否为字符串，如果是则尝试解析为JSON
                if isinstance(qa_pair, str):
                    try:
                        qa_pair = json.loads(qa_pair)
                    except json.JSONDecodeError:
                        logger.error(f"Q&A对 {i} 是无效的JSON字符串: {qa_pair[:100]}...")
                        continue
                
                # 确保qa_pair是字典类型
                if not isinstance(qa_pair, dict):
                    logger.error(f"Q&A对 {i} 不是字典类型: {type(qa_pair)}")
                    continue
                
                # 验证必要字段
                if 'question' not in qa_pair or 'answer' not in qa_pair:
                    logger.error(f"Q&A对 {i} 缺少必要字段 'question' 或 'answer': {qa_pair.keys()}")
                    continue
                
                # 为每个Q&A对创建两个文档：问题文档和答案文档
                
                # 问题文档
                question_doc = {
                    'id': f"qa_{i}_question",
                    'title': f"医疗问题: {qa_pair.get('question', '')[:50]}...",
                    'content': qa_pair.get('question', ''),
                    'document_type': 'medical_question',
                    'qa_pair_id': i,
                    'difficulty_level': qa_pair.get('difficulty_level', 'medium'),
                    'question_type': qa_pair.get('question_type', 'general'),
                    'medical_domain': qa_pair.get('medical_domain', 'pathology'),
                    'source': 'medical_data_engineering',
                    'create_time': datetime.now().isoformat(),
                    'metadata': {
                        'is_question': True,
                        'related_answer_id': f"qa_{i}_answer",
                        'quality_score': qa_pair.get('quality_score', 0.5),
                        'keywords': qa_pair.get('keywords', [])
                    }
                }
                
                # 答案文档
                answer_doc = {
                    'id': f"qa_{i}_answer",
                    'title': f"医疗答案: {qa_pair.get('question', '')[:50]}...",
                    'content': qa_pair.get('answer', ''),
                    'document_type': 'medical_answer',
                    'qa_pair_id': i,
                    'difficulty_level': qa_pair.get('difficulty_level', 'medium'),
                    'question_type': qa_pair.get('question_type', 'general'),
                    'medical_domain': qa_pair.get('medical_domain', 'pathology'),
                    'source': 'medical_data_engineering',
                    'create_time': datetime.now().isoformat(),
                    'metadata': {
                        'is_question': False,
                        'related_question_id': f"qa_{i}_question",
                        'quality_score': qa_pair.get('quality_score', 0.5),
                        'keywords': qa_pair.get('keywords', []),
                        'answer_length': len(qa_pair.get('answer', '')),
                        'medical_terms': qa_pair.get('medical_terms', [])
                    }
                }
                
                documents.extend([question_doc, answer_doc])
                
            except Exception as e:
                logger.error(f"转换Q&A对 {i} 失败: {e}")
                continue
        
        logger.info(f"成功转换 {len(documents)} 个文档")
        return documents
    
    def search_medical_knowledge(self, index_name: str, query: str, 
                               search_type: str = "hybrid", 
                               top_k: int = 10) -> Dict[str, Any]:
        """搜索医疗知识库
        
        Args:
            index_name: 索引名称
            query: 搜索查询
            search_type: 搜索类型 (accurate, semantic, hybrid)
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        try:
            if search_type == "accurate":
                results = self.es_core.accurate_search(index_name, query, top_k)
            elif search_type == "semantic":
                results = self.es_core.semantic_search(index_name, query, top_k)
            elif search_type == "hybrid":
                results = self.es_core.hybrid_search(index_name, query, top_k, weight_accurate=0.3)
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")
            
            # 格式化搜索结果
            formatted_results = []
            for result in results:
                formatted_result = {
                    'score': result.get('score', 0),
                    'document_type': result.get('document', {}).get('document_type', 'unknown'),
                    'title': result.get('document', {}).get('title', ''),
                    'content': result.get('document', {}).get('content', ''),
                    'qa_pair_id': result.get('document', {}).get('qa_pair_id', -1),
                    'difficulty_level': result.get('document', {}).get('difficulty_level', 'unknown'),
                    'medical_domain': result.get('document', {}).get('medical_domain', 'unknown'),
                    'metadata': result.get('document', {}).get('metadata', {})
                }
                formatted_results.append(formatted_result)
            
            return {
                'success': True,
                'query': query,
                'search_type': search_type,
                'total_results': len(formatted_results),
                'results': formatted_results,
                'search_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"搜索医疗知识库失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'search_type': search_type,
                'results': []
            }
    
    def get_medical_indices(self) -> Dict[str, Any]:
        """获取所有医疗知识库索引
        
        Returns:
            医疗索引列表
        """
        try:
            all_indices = self.es_core.get_user_indices()
            medical_indices = [idx for idx in all_indices if idx.startswith(self.medical_index_prefix)]
            
            # 获取每个医疗索引的详细信息
            indices_info = {}
            for index_name in medical_indices:
                try:
                    stats = self.es_core.get_index_stats(index_name)
                    indices_info[index_name] = stats
                except Exception as e:
                    logger.error(f"获取索引 {index_name} 统计信息失败: {e}")
                    indices_info[index_name] = {'error': str(e)}
            
            return {
                'success': True,
                'total_medical_indices': len(medical_indices),
                'medical_indices': medical_indices,
                'indices_info': indices_info,
                'query_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取医疗知识库索引失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'medical_indices': []
            }
    
    def delete_medical_index(self, index_name: str) -> Dict[str, Any]:
        """删除医疗知识库索引
        
        Args:
            index_name: 要删除的索引名称
            
        Returns:
            删除结果
        """
        try:
            if not index_name.startswith(self.medical_index_prefix):
                return {
                    'success': False,
                    'error': f'索引 {index_name} 不是医疗知识库索引',
                    'index_name': index_name
                }
            
            success = self.es_core.delete_index(index_name)
            
            if success:
                result = {
                    'success': True,
                    'message': f'医疗知识库索引 {index_name} 删除成功',
                    'index_name': index_name,
                    'deleted_at': datetime.now().isoformat()
                }
                logger.info(f"医疗知识库索引删除成功: {index_name}")
            else:
                result = {
                    'success': False,
                    'error': f'索引 {index_name} 删除失败',
                    'index_name': index_name
                }
                logger.error(f"医疗知识库索引删除失败: {index_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"删除医疗知识库索引失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name
            }