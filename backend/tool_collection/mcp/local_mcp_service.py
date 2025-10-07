from fastmcp import FastMCP
from .medical_image_analysis import MedicalCaseAnalyzer
from .medical_data_processing.data_cleaner import MedicalDataCleaner
from .medical_data_processing.professional_annotator import MedicalProfessionalAnnotator
from .medical_data_processing.qa_generator import MedicalQAGenerator
from .medical_data_processing.knowledge_base_integrator import MedicalKnowledgeBaseIntegrator
import json
import os
from typing import List
from database.client import get_db_session
from database.db_models import ConversationMessage, ConversationRecord
from database.attachment_db import get_file_stream
from sqlalchemy import select, desc
from consts.const import DEFAULT_TENANT_ID

# Create MCP server 
local_mcp_service = FastMCP("local")  

@local_mcp_service.tool(name="breast_histology_analyzer", 
                        description="专业的乳腺组织学显微镜图像分析工具，支持乳腺病理切片的智能分析和诊断辅助")
async def breast_histology_analyzer(image_path: str, 
                                  magnification: str = "未知",
                                  staining_method: str = "HE",
                                  clinical_info: str = "",
                                  custom_requirements: str = "") -> str:
    """
    乳腺组织学显微镜图像分析工具
    
    Args:
        image_path: 乳腺组织学显微镜图像文件的完整路径
        magnification: 显微镜放大倍数（如：40x, 100x, 200x, 400x等）
        staining_method: 染色方法（默认：HE染色，也可以是免疫组化等）
        clinical_info: 相关临床信息（年龄、症状、影像学发现等）
        custom_requirements: 自定义分析要求或特别关注点
    
    Returns:
        JSON格式的病理学分析结果
    """
    
    print("breast_histology_analyzer 乳腺组织学分析工具启动成功")
    print(f"原始文件名: {image_path}")
    
    try:
        # 获取 MinIO 对象名
        minio_object_name = find_minio_object_name_by_ui_filename(image_path)
        print(f"MinIO对象名: {minio_object_name}")
        
        if not minio_object_name or minio_object_name == image_path:
            return json.dumps({
                "success": False,
                "error": f"无法找到文件 {image_path} 对应的MinIO对象",
                "analysis": None
            }, ensure_ascii=False, indent=2)
        
        # 从 MinIO 获取文件流
        file_stream = get_file_stream(minio_object_name)
        
        if not file_stream:
            return json.dumps({
                "success": False,
                "error": f"无法从MinIO获取文件流: {minio_object_name}",
                "analysis": None
            }, ensure_ascii=False, indent=2)
        
        print("MinIO文件流获取成功")
        
        # 尝试从数据库中获取最近的 tenant_id
        tenant_id = get_recent_tenant_id_from_conversation()
        print(f"[DEBUG] 获取到的租户ID: {tenant_id}")
        print(f"[DEBUG] 租户ID类型: {type(tenant_id)}")
        print(f"[DEBUG] 租户ID是否为空: {tenant_id is None}")
        
        # 初始化医疗病例分析器
        print(f"[DEBUG] 开始初始化 MedicalCaseAnalyzer，传入 tenant_id: {tenant_id}")
        analyzer = MedicalCaseAnalyzer(tenant_id=tenant_id)
        print(f"[DEBUG] MedicalCaseAnalyzer 初始化完成")
        
        # 构建自定义提示词，包含所有分析参数
        custom_prompt = f"""
你是一位专业的乳腺病理学专家，请对这张乳腺组织学显微镜图像进行详细分析。

分析参数：
- 放大倍数：{magnification}
- 染色方法：{staining_method}
- 临床信息：{clinical_info if clinical_info else "无"}
- 特殊要求：{custom_requirements if custom_requirements else "无"}

请从以下几个方面进行分析：
1. 组织结构特征
2. 细胞形态学特征
3. 病理学诊断意见
4. 分级评估（如适用）
5. 临床意义和建议

请提供专业、准确、详细的分析结果。
"""
        
        # 分析图像 - 直接使用图像流
        result = analyzer.analyze_medical_image_from_stream(
        image_stream=file_stream,
        analysis_type="general_analysis",
        custom_prompt=custom_prompt
        )
        
        # 记录元数据
        metadata = {
            "tool_name": "breast_histology_analyzer",
            "image_path": image_path,
            "minio_object_name": minio_object_name,
            "magnification": magnification,
            "staining_method": staining_method,
            "clinical_info": clinical_info,
            "custom_requirements": custom_requirements,
            "tenant_id": tenant_id,
            "timestamp": result.get("timestamp", "")
        }
        
        print(f"乳腺组织学分析完成，元数据: {metadata}")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"乳腺组织学分析工具执行失败: {str(e)}"
        print(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "analysis": None
        }, ensure_ascii=False, indent=2)


def get_recent_tenant_id_from_conversation() -> str:
    """
    从最近的对话记录中获取 tenant_id
    这是一个临时解决方案，用于在 MCP 工具中获取上下文信息
    由于ConversationMessage表没有tenant_id字段，直接返回默认值
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 添加调试日志
    logger.info(f"[DEBUG] get_recent_tenant_id_from_conversation 被调用")
    logger.info(f"[DEBUG] DEFAULT_TENANT_ID: {DEFAULT_TENANT_ID}")
    logger.info(f"[DEBUG] DEFAULT_TENANT_ID 类型: {type(DEFAULT_TENANT_ID)}")
    
    # 尝试从数据库获取真实的 tenant_id
    try:
        with get_db_session() as session:
            # 查询最近的对话记录，尝试获取 tenant_id
            stmt = select(ConversationRecord).order_by(desc(ConversationRecord.created_at)).limit(1)
            result = session.execute(stmt).first()
            
            if result and hasattr(result[0], 'tenant_id'):
                actual_tenant_id = result[0].tenant_id
                logger.info(f"[DEBUG] 从数据库获取到的 tenant_id: {actual_tenant_id}")
                return actual_tenant_id
            else:
                logger.warning(f"[DEBUG] 无法从数据库获取 tenant_id，使用默认值: {DEFAULT_TENANT_ID}")
                
    except Exception as e:
        logger.error(f"[DEBUG] 从数据库获取 tenant_id 时出错: {e}")
        logger.error(f"[DEBUG] 使用默认值: {DEFAULT_TENANT_ID}")
    
    return DEFAULT_TENANT_ID


@local_mcp_service.tool(name="medical_data_cleaner",
                        description="医疗数据清洗工具，专门用于清洗病理教科书等医疗专业素材，去除噪声并提取医学术语")
async def medical_data_cleaner(text_content: str = "",
                             file_path: str = "",
                             batch_file_paths: str = "") -> str:
    """
    医疗数据清洗工具
    
    Args:
        text_content: 直接输入的医疗文本内容
        file_path: 单个文件路径
        batch_file_paths: 批量文件路径，用逗号分隔
    
    Returns:
        JSON格式的清洗结果
    """
    try:
        cleaner = MedicalDataCleaner()
        
        if batch_file_paths:
            # 批量处理文件
            file_paths = [path.strip() for path in batch_file_paths.split(',') if path.strip()]
            result = cleaner.batch_clean_files(file_paths)
        elif file_path:
            # 处理单个文件
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = cleaner.clean_medical_text(content)
            result['source_file'] = file_path
        elif text_content:
            # 处理直接输入的文本
            result = cleaner.clean_medical_text(text_content)
            result['source_type'] = 'direct_input'
        else:
            raise ValueError("必须提供text_content、file_path或batch_file_paths中的一个参数")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"医疗数据清洗失败: {str(e)}",
            "suggestion": "请检查输入参数是否正确"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@local_mcp_service.tool(name="medical_professional_annotator",
                        description="医疗专业标注工具，对清洗后的医疗数据进行病理学术语、疾病诊断、组织学特征等专业标注")
async def medical_professional_annotator(content: str,
                                       content_type: str = "general") -> str:
    """
    医疗专业标注工具
    
    Args:
        content: 需要标注的医疗文本内容
        content_type: 内容类型，可选值：
                     - general: 通用医疗内容
                     - pathology: 病理学内容
                     - diagnosis: 诊断相关内容
                     - treatment: 治疗相关内容
    
    Returns:
        JSON格式的标注结果
    """
    try:
        annotator = MedicalProfessionalAnnotator()
        result = annotator.annotate_medical_content(content, content_type)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"医疗专业标注失败: {str(e)}",
            "suggestion": "请检查输入内容是否为有效的医疗文本"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@local_mcp_service.tool(name="medical_qa_generator",
                        description="医疗Q&A数据集生成工具，基于标注后的医疗数据生成高质量的病理学问答对数据集")
async def medical_qa_generator(annotated_content: str,
                             qa_count: int = 10) -> str:
    """
    医疗Q&A数据集生成工具
    
    Args:
        annotated_content: 标注后的医疗内容（JSON字符串格式）
        qa_count: 生成的Q&A对数量，默认10个
    
    Returns:
        JSON格式的Q&A数据集
    """
    try:
        # 解析输入的标注内容
        if isinstance(annotated_content, str):
            annotated_data = json.loads(annotated_content)
        else:
            annotated_data = annotated_content
        
        generator = MedicalQAGenerator()
        result = generator.generate_qa_dataset(annotated_data, qa_count)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"JSON解析失败: {str(e)}",
            "suggestion": "请确保annotated_content是有效的JSON格式"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Q&A数据集生成失败: {str(e)}",
            "suggestion": "请检查输入的标注内容是否完整和正确"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@local_mcp_service.tool(name="medical_data_pipeline",
                        description="医疗数据工程流水线工具，完整执行数据清洗→专业标注→Q&A生成的全流程处理")
async def medical_data_pipeline(input_content: str = "",
                              input_file_path: str = "",
                              qa_count: int = 10,
                              content_type: str = "general") -> str:
    """
    医疗数据工程流水线工具
    
    Args:
        input_content: 输入的原始医疗文本
        input_file_path: 输入文件路径
        qa_count: 生成的Q&A对数量
        content_type: 内容类型
    
    Returns:
        JSON格式的完整处理结果
    """
    try:
        pipeline_result = {
            "success": True,
            "pipeline_steps": [],
            "final_output": {},
            "processing_stats": {}
        }
        
        # 步骤1: 数据清洗
        print("开始数据清洗...")
        cleaner = MedicalDataCleaner()
        
        if input_file_path:
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"文件不存在: {input_file_path}")
            with open(input_file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        elif input_content:
            raw_content = input_content
        else:
            raise ValueError("必须提供input_content或input_file_path")
        
        cleaning_result = cleaner.clean_medical_text(raw_content)
        pipeline_result["pipeline_steps"].append({
            "step": "data_cleaning",
            "status": "completed" if cleaning_result.get("success") else "failed",
            "result": cleaning_result
        })
        
        if not cleaning_result.get("success"):
            raise Exception("数据清洗失败")
        
        # 步骤2: 专业标注
        print("开始专业标注...")
        annotator = MedicalProfessionalAnnotator()
        cleaned_content = cleaning_result.get("cleaned_text", "")
        
        annotation_result = annotator.annotate_medical_content(cleaned_content, content_type)
        pipeline_result["pipeline_steps"].append({
            "step": "professional_annotation",
            "status": "completed" if annotation_result.get("success") else "failed",
            "result": annotation_result
        })
        
        if not annotation_result.get("success"):
            raise Exception("专业标注失败")
        
        # 步骤3: Q&A生成
        print("开始Q&A生成...")
        generator = MedicalQAGenerator()
        
        qa_result = generator.generate_qa_dataset(annotation_result, qa_count)
        pipeline_result["pipeline_steps"].append({
            "step": "qa_generation",
            "status": "completed" if qa_result.get("success") else "failed",
            "result": qa_result
        })
        
        if not qa_result.get("success"):
            raise Exception("Q&A生成失败")
        
        # 汇总最终结果
        pipeline_result["final_output"] = {
            "original_content_length": len(raw_content),
            "cleaned_content_length": len(cleaned_content),
            "annotation_count": annotation_result.get("annotation_stats", {}).get("total_annotations", 0),
            "qa_pairs_count": len(qa_result.get("qa_pairs", [])),
            "quality_metrics": qa_result.get("quality_metrics", {}),
            "qa_dataset": qa_result.get("qa_pairs", [])
        }
        
        # 处理统计
        pipeline_result["processing_stats"] = {
            "total_steps": 3,
            "completed_steps": 3,
            "success_rate": 1.0,
            "processing_time": "completed",
            "data_quality_score": cleaning_result.get("quality_score", 0),
            "annotation_quality": annotation_result.get("structured_annotation", {}).get("annotation_quality", 0),
            "qa_quality": qa_result.get("quality_metrics", {}).get("overall_quality", 0)
        }
        
        print("医疗数据工程流水线处理完成！")
        return json.dumps(pipeline_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"医疗数据工程流水线处理失败: {str(e)}",
            "pipeline_steps": pipeline_result.get("pipeline_steps", []),
            "suggestion": "请检查输入数据和参数设置"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@local_mcp_service.tool(name="medical_knowledge_base_integrator",
                        description="医疗知识库集成工具，将Q&A数据集存储到Elasticsearch知识库，支持创建索引、数据存储和搜索")
async def medical_knowledge_base_integrator(action: str,
                                          index_name: str = "",
                                          qa_dataset: str = "",
                                          query: str = "",
                                          search_type: str = "hybrid",
                                          top_k: int = 10) -> str:
    """
    医疗知识库集成工具
    
    Args:
        action: 操作类型 (create_index, integrate_data, store, search, list_indices, delete_index)
        index_name: 索引名称
        qa_dataset: Q&A数据集 (JSON字符串格式)
        query: 搜索查询
        search_type: 搜索类型 (accurate, semantic, hybrid)
        top_k: 搜索结果数量
    
    Returns:
        JSON格式的操作结果
    """
    try:
        integrator = MedicalKnowledgeBaseIntegrator()
        
        if action == "create_index":
            # 创建医疗知识库索引
            result = integrator.create_medical_knowledge_index(index_name if index_name else None)
            
        elif action == "integrate_data" or action == "store":
            # 集成Q&A数据到知识库 (store是integrate_data的别名)
            if not index_name:
                raise ValueError("集成数据需要指定index_name")
            if not qa_dataset:
                raise ValueError("集成数据需要提供qa_dataset")
            
            # 解析Q&A数据集
            try:
                qa_data = json.loads(qa_dataset)
                if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
                    qa_data = qa_data['qa_pairs']
            except json.JSONDecodeError:
                raise ValueError("qa_dataset必须是有效的JSON格式")
            
            result = integrator.integrate_qa_dataset(qa_data, index_name)
            
        elif action == "search":
            # 搜索医疗知识库
            if not index_name:
                raise ValueError("搜索需要指定index_name")
            if not query:
                raise ValueError("搜索需要提供query")
            
            result = integrator.search_medical_knowledge(index_name, query, search_type, top_k)
            
        elif action == "list_indices":
            # 列出所有医疗知识库索引
            result = integrator.get_medical_indices()
            
        elif action == "delete_index":
            # 删除医疗知识库索引
            if not index_name:
                raise ValueError("删除索引需要指定index_name")
            
            result = integrator.delete_medical_index(index_name)
            
        else:
            raise ValueError(f"不支持的操作类型: {action}。支持的操作: create_index, integrate_data, store, search, list_indices, delete_index")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"医疗知识库集成操作失败: {str(e)}",
            "action": action,
            "suggestion": "请检查参数设置和Elasticsearch连接状态"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)




def find_minio_object_name_by_ui_filename(ui_filename: str) -> str:
    """
    通过UI文件名查找MinIO中的真实对象名
    
    Args:
        ui_filename: UI上传的原始文件名 (如: aug_0_1018.jpg)
    
    Returns:
        str: MinIO中的对象名 (如: attachments/20241201123456_abc123def.jpg)
    """
    with get_db_session() as session:
        # 查询最近的包含该文件名的消息记录
        query = select(ConversationMessage.minio_files).where(
            ConversationMessage.minio_files.isnot(None),
            ConversationMessage.delete_flag == 'N'
        ).order_by(desc(ConversationMessage.create_time))
        
        results = session.execute(query).scalars().all()
        
        for minio_files_str in results:
            try:
                # 解析JSON字符串
                if isinstance(minio_files_str, str):
                    minio_files = json.loads(minio_files_str)
                else:
                    minio_files = minio_files_str
                
                # 在文件列表中查找匹配的文件名
                if isinstance(minio_files, list):
                    for file_info in minio_files:
                        if isinstance(file_info, dict):
                            # 检查不同可能的文件名字段
                            file_name = file_info.get('name') or file_info.get('file_name')
                            if file_name == ui_filename:
                                return file_info.get('object_name', '')
                                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                continue
    
    # 如果没找到，返回原文件名（作为fallback）
    return ui_filename

