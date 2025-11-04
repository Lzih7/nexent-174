from fastmcp import FastMCP
from .medical_image_analysis import MedicalCaseAnalyzer
from .medical_data_processing.qa_generator import MedicalQAGenerator
from .medical_data_processing.knowledge_base_integrator import MedicalKnowledgeBaseIntegrator
import json
import os
import io
from typing import List, Optional, BinaryIO
from database.client import get_db_session
from database.db_models import ConversationMessage, ConversationRecord
from database.attachment_db import get_file_stream
from sqlalchemy import select, desc
from consts.const import DEFAULT_TENANT_ID
import logging

from utils.logging_utils import configure_logging
configure_logging(logging.INFO)

local_mcp_service = FastMCP("local")
logger = logging.getLogger(__name__)


async def _read_pdf_from_stream(file_stream, max_chars: int = None) -> str:
    """从文件流读取PDF内容，使用pypdf库
    
    Args:
        file_stream: PDF文件流
        max_chars: 最大字符数限制，None表示不限制
    
    Returns:
        PDF内容（完整或截取后的）
    """
    try:
        import pypdf
        
        # 将文件流内容读取到内存
        pdf_bytes = file_stream.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        
        reader = pypdf.PdfReader(pdf_stream)
        content = ""
        total_pages = len(reader.pages)
        processed_pages = 0
        
        logger.info(f"使用 pypdf 开始处理PDF文件流，总页数: {total_pages}")
        if max_chars:
            logger.info(f"字符数限制: {max_chars}")
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                
                # 如果设置了字符限制，检查是否会超出
                if max_chars and len(content) + len(page_text) > max_chars:
                    remaining_chars = max_chars - len(content)
                    if remaining_chars > 0:
                        content += page_text[:remaining_chars]
                    logger.info(f"达到字符限制 {max_chars}，停止处理。已处理页数: {processed_pages + 1}/{total_pages}")
                    break
                
                content += page_text + "\n"
                processed_pages += 1
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"已处理 {page_num + 1}/{total_pages} 页，当前内容长度: {len(content)}")
                    
            except Exception as e:
                logger.warning(f"处理第 {page_num + 1} 页时出错: {e}")
                continue
        
        logger.info(f"pypdf 处理完成，总页数: {total_pages}，处理页数: {processed_pages}，最终内容长度: {len(content)}")
        return content.strip()
        
    except ImportError:
        logger.error("pypdf 库未安装，请确保在 pyproject.toml 中已添加 pypdf>=3.6.0 依赖")
        raise Exception("pypdf 库未安装，无法处理PDF文件")
    except Exception as e:
        logger.error(f"使用 pypdf 读取PDF文件流失败: {e}")
        raise


def find_minio_object_name_by_ui_filename(ui_filename: str) -> Optional[str]:
    """根据UI文件名查找MinIO对象名"""
    try:
        with get_db_session() as session:
            # 查询最近的对话记录
            stmt = select(ConversationMessage).order_by(desc(ConversationMessage.created_at)).limit(100)
            messages = session.execute(stmt).scalars().all()
            
            for message in messages:
                if message.attachments:
                    for attachment in message.attachments:
                        if attachment.get('ui_filename') == ui_filename:
                            return attachment.get('minio_object_name')
            
            logger.warning(f"未找到UI文件名 '{ui_filename}' 对应的MinIO对象")
            return None
            
    except Exception as e:
        logger.error(f"查找MinIO对象名失败: {e}")
        return None


def get_recent_tenant_id_from_conversation() -> str:
    """从最近的对话记录中获取租户ID"""
    try:
        with get_db_session() as session:
            # 查询最近的对话记录
            stmt = select(ConversationRecord).order_by(desc(ConversationRecord.created_at)).limit(1)
            recent_conversation = session.execute(stmt).scalar_one_or_none()
            
            if recent_conversation and recent_conversation.tenant_id:
                logger.info(f"从对话记录获取到租户ID: {recent_conversation.tenant_id}")
                return recent_conversation.tenant_id
            else:
                logger.warning(f"未找到最近的对话记录或租户ID为空")
                
    except Exception as e:
        logger.error(f"获取租户ID失败: {e}")
        
    # 如果无法获取，使用默认值
    logger.error(f"使用默认值: {DEFAULT_TENANT_ID}")
    
    return DEFAULT_TENANT_ID


@local_mcp_service.tool(name="medical_data_pipeline", 
                        description="简化的医疗数据处理管道")
async def medical_data_pipeline(content: str = "", 
                              file_path: str = "", 
                              max_chars: int = None) -> str:
    """简化版医疗数据处理管道"""
    
    try:
        # 步骤1: 获取内容（简化日志）
        if file_path:
            minio_object_name = find_minio_object_name_by_ui_filename(file_path)
            if not minio_object_name:
                return json.dumps({"status": "error", "message": "文件未找到"}, ensure_ascii=False)
            
            content = await _read_pdf_from_stream(get_file_stream(minio_object_name), max_chars)
        
        if not content:
            return json.dumps({"status": "error", "message": "内容为空"}, ensure_ascii=False)
        
        # 步骤2: 生成Q&A
        tenant_id = get_recent_tenant_id_from_conversation()
        generator = MedicalQAGenerator(tenant_id=tenant_id)
        qa_result = generator.generate_qa_from_content(content)
        
        if not qa_result.get('success'):
            return json.dumps({"status": "error", "message": "Q&A生成失败"}, ensure_ascii=False)
        
        # 步骤3: 存储到向量数据库
        integrator = MedicalKnowledgeBaseIntegrator(tenant_id=tenant_id)
        integration_result = integrator.integrate_qa_dataset(qa_result, "medical_q_a")
        
        return json.dumps({
            "status": "success",
            "indexed_count": integration_result.get('indexed_count', 0),
            "qa_pairs_count": len(qa_result.get("qa_pairs", []))
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


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
    
    logger.info("breast_histology_analyzer 乳腺组织学分析工具启动成功")
    logger.info(f"原始文件名: {image_path}")
    
    try:
        # 获取 MinIO 对象名
        minio_object_name = find_minio_object_name_by_ui_filename(image_path)
        logger.info(f"MinIO对象名: {minio_object_name}")
        
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
        
        logger.info("MinIO文件流获取成功")
        
        # 尝试从数据库中获取最近的 tenant_id
        tenant_id = get_recent_tenant_id_from_conversation()
        logger.info(f"获取到的租户ID: {tenant_id}")
        # logger.info(f"租户ID类型: {type(tenant_id)}")
        logger.info(f"租户ID是否为空: {tenant_id is None}")
        
        # 初始化医疗病例分析器
        logger.info(f"开始初始化 MedicalCaseAnalyzer，传入 tenant_id: {tenant_id}")
        analyzer = MedicalCaseAnalyzer(tenant_id=tenant_id)
        logger.info("MedicalCaseAnalyzer 初始化完成")
        
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
        
        logger.info(f"乳腺组织学分析完成，元数据: {metadata}")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"乳腺组织学分析工具执行失败: {str(e)}"
        logger.error(error_msg)
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
    
    # 添加调试日志
    logger.debug("get_recent_tenant_id_from_conversation 被调用")
    logger.debug(f"DEFAULT_TENANT_ID: {DEFAULT_TENANT_ID}")
    logger.debug(f"DEFAULT_TENANT_ID 类型: {type(DEFAULT_TENANT_ID)}")
    
    # 尝试从数据库获取真实的 tenant_id
    try:
        with get_db_session() as session:
            # 查询最近的对话记录，使用 create_time 而不是 created_at
            stmt = select(ConversationRecord).order_by(desc(ConversationRecord.create_time)).limit(1)
            result = session.execute(stmt).first()
            
            if result and hasattr(result[0], 'tenant_id'):
                actual_tenant_id = result[0].tenant_id
                logger.debug(f"从数据库获取到的 tenant_id: {actual_tenant_id}")
                return actual_tenant_id
            else:
                logger.warning(f"无法从数据库获取 tenant_id，使用默认值: {DEFAULT_TENANT_ID}")
                
    except Exception as e:
        logger.error(f"从数据库获取 tenant_id 时出错: {e}")
        logger.error(f"使用默认值: {DEFAULT_TENANT_ID}")
    
    return DEFAULT_TENANT_ID



@local_mcp_service.tool(name="medical_data_pipeline",
                        description="医疗数据处理流水线工具，从MinIO获取PDF数据，使用LLM生成高质量Q&A，并存储到向量数据库")
async def medical_data_pipeline(input_file_path: str = "",
                              qa_count: int = 10) -> str:
    """
    简化的医疗数据处理流水线工具
    
    Args:
        input_file_path: 输入PDF文件路径
        qa_count: 生成的Q&A对数量
    
    Returns:
        JSON格式的处理结果
    """
    try:
        logger.info(f"🔄 开始医疗数据流水线处理")
        logger.info(f"参数: input_file_path='{input_file_path}', qa_count={qa_count}")
        
        # 获取租户ID
        tenant_id = get_recent_tenant_id_from_conversation()
        logger.info(f"获取到租户ID: {tenant_id}")
        
        # 步骤1: 从MinIO读取PDF内容
        logger.info(f"📄 步骤1: 从MinIO读取PDF文件")
        if not input_file_path.lower().endswith('.pdf'):
            return json.dumps({
                "status": "error",
                "message": "仅支持PDF文件格式"
            }, ensure_ascii=False, indent=2)
        
        # 统一使用 basename 进行 MinIO 匹配，避免携带路径导致匹配失败
        ui_basename = os.path.basename(input_file_path)
        minio_object_name = find_minio_object_name_by_ui_filename(ui_basename)
        
        file_stream = None
        if minio_object_name and minio_object_name != ui_basename:
            logger.info(f"从MinIO读取PDF: {minio_object_name}")
            file_stream = get_file_stream(minio_object_name)
            if not file_stream:
                logger.warning(f"无法从MinIO获取文件流: {minio_object_name}，准备尝试本地回退读取")
        else:
            logger.warning(f"未找到对应的MinIO对象或对象名与UI名相同: {ui_basename}，准备尝试本地回退读取")
        
        # 本地文件回退读取
        content = None
        if file_stream:
            content = await _read_pdf_from_stream(file_stream)
        else:
            if os.path.exists(input_file_path):
                logger.info(f"使用本地文件回退读取: {input_file_path}")
                with open(input_file_path, 'rb') as f:
                    content = await _read_pdf_from_stream(f)
            else:
                # 再尝试当前工作目录下的 basename
                local_basename_path = os.path.join(os.getcwd(), ui_basename)
                if os.path.exists(local_basename_path):
                    logger.info(f"使用当前目录下的文件回退读取: {local_basename_path}")
                    with open(local_basename_path, 'rb') as f:
                        content = await _read_pdf_from_stream(f)
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"无法从MinIO或本地读取文件，请确认：1) 通过UI上传后传入原始文件名'{ui_basename}'；或 2) 提供可访问的本地绝对路径/当前目录文件"
                    }, ensure_ascii=False, indent=2)
        
        if not content:
            return json.dumps({
                "status": "error",
                "message": "PDF文件内容为空"
            }, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ PDF内容读取完成，长度: {len(content)}")
        
        # 步骤2: 使用LLM直接生成Q&A
        logger.info(f"🤖 步骤2: 使用LLM生成Q&A")
        qa_generator = MedicalQAGenerator(tenant_id=tenant_id)
        qa_result = qa_generator.generate_qa_from_content(content, qa_count)
        
        if not qa_result.get('success'):
            return json.dumps({
                "status": "error",
                "message": f"Q&A生成失败: {qa_result.get('error', '未知错误')}"
            }, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Q&A生成完成，生成数量: {len(qa_result.get('qa_pairs', []))}")
        
        # 步骤3: 存储到向量数据库
        logger.info(f"📚 步骤3: 存储Q&A到向量数据库")
        integrator = MedicalKnowledgeBaseIntegrator(tenant_id=tenant_id)
        
        # 创建或确保索引存在
        index_name = "medical_q_a"
        create_result = integrator.create_medical_index(index_name)
        logger.info(f"索引创建结果: {create_result.get('message', '未知')}")
        
        # 存储Q&A数据
        integration_result = integrator.integrate_qa_dataset(qa_result, index_name)
        
        if not integration_result.get('success'):
            return json.dumps({
                "status": "error",
                "message": f"向量数据库存储失败: {integration_result.get('error', '未知错误')}"
            }, ensure_ascii=False, indent=2)
        
        indexed_count = integration_result.get('indexed_count', 0)
        logger.info(f"✅ 向量数据库存储完成，存储数量: {indexed_count}")
        
        logger.info(f"🎉 医疗数据流水线处理完成")
        
        return json.dumps({
            "status": "success",
            "message": "医疗数据处理流水线执行成功",
            "results": {
                "pdf_content_length": len(content),
                "qa_pairs_generated": len(qa_result.get('qa_pairs', [])),
                "qa_pairs_stored": indexed_count,
                "index_name": index_name,
                "qa_pairs": qa_result.get('qa_pairs', [])
            },
            "statistics": qa_result.get('statistics', {}),
            "quality_metrics": qa_result.get('quality_metrics', {})
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"❌ 医疗数据流水线处理失败: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"流水线处理失败: {str(e)}"
        }, ensure_ascii=False, indent=2)



# @local_mcp_service.tool(name="medical_knowledge_base_integrator",
#                         description="医疗知识库集成工具，将Q&A数据集存储到Elasticsearch知识库，支持创建索引、数据存储和搜索")
# async def medical_knowledge_base_integrator(action: str,
#                                           index_name: str = "",
#                                           qa_dataset: str = "",
#                                           query: str = "",
#                                           search_type: str = "hybrid",
#                                           top_k: int = 10) -> str:
#     """
#     医疗知识库集成工具
    
#     Args:
#         action: 操作类型 (create_index, integrate_data, store, search, list_indices, delete_index)
#         index_name: 索引名称
#         qa_dataset: Q&A数据集 (JSON字符串格式)
#         query: 搜索查询
#         search_type: 搜索类型 (accurate, semantic, hybrid)
#         top_k: 搜索结果数量
    
#     Returns:
#         JSON格式的操作结果
#     """
#     try:
#         # 获取 tenant_id
#         tenant_id = get_recent_tenant_id_from_conversation()
#         logger.debug(f"医疗知识库集成器使用的租户ID: {tenant_id}")
        
#         integrator = MedicalKnowledgeBaseIntegrator(tenant_id=tenant_id)
        
#         if action == "create_index":
#             # 创建医疗知识库索引
#             result = integrator.create_medical_knowledge_index(index_name if index_name else None)
            
#         elif action == "integrate_data" or action == "store":
#             # 集成Q&A数据到知识库 (store是integrate_data的别名)
#             if not index_name:
#                 raise ValueError("集成数据需要指定index_name")
#             # 处理qa_dataset参数
#             if qa_dataset:
#                 try:
#                     # 解析qa_dataset JSON字符串
#                     if isinstance(qa_dataset, str):
#                         qa_data = json.loads(qa_dataset)
#                     else:
#                         qa_data = qa_dataset
                    
#                     # 验证qa_data结构
#                     if qa_data is None:
#                         raise ValueError("qa_data为None，无法处理")
                    
#                     # 修复：正确提取Q&A对数据
#                     actual_qa_pairs = None
                    
#                     # 情况1：直接是Q&A对列表
#                     if isinstance(qa_data, list):
#                         actual_qa_pairs = qa_data
#                         logger.info(f"直接使用Q&A对列表，数量: {len(actual_qa_pairs)}")
                    
#                     # 情况2：包含final_output的完整流水线结果
#                     elif isinstance(qa_data, dict):
#                         if 'final_output' in qa_data:
#                             final_output = qa_data['final_output']
#                             if 'qa_dataset' in final_output:
#                                 qa_dataset_obj = final_output['qa_dataset']
#                                 if isinstance(qa_dataset_obj, dict) and 'qa_pairs' in qa_dataset_obj:
#                                     actual_qa_pairs = qa_dataset_obj['qa_pairs']
#                                     logger.info(f"从final_output.qa_dataset.qa_pairs提取，数量: {len(actual_qa_pairs) if isinstance(actual_qa_pairs, list) else 'N/A'}")
#                                 else:
#                                     logger.error(f"qa_dataset结构错误，类型: {type(qa_dataset_obj)}, 键: {list(qa_dataset_obj.keys()) if isinstance(qa_dataset_obj, dict) else 'N/A'}")
#                             else:
#                                 logger.error(f"final_output中缺少qa_dataset字段，可用字段: {list(final_output.keys())}")
#                         elif 'qa_pairs' in qa_data:
#                             actual_qa_pairs = qa_data['qa_pairs']
#                             logger.info(f"从根级qa_pairs提取，数量: {len(actual_qa_pairs) if isinstance(actual_qa_pairs, list) else 'N/A'}")
#                         else:
#                             logger.error(f"无法找到Q&A数据，可用字段: {list(qa_data.keys())}")
                    
#                     # 验证提取的Q&A对
#                     if actual_qa_pairs is None:
#                         raise ValueError("无法从输入数据中提取有效的Q&A对")
                    
#                     if not isinstance(actual_qa_pairs, list):
#                         raise ValueError(f"Q&A对必须是列表格式，当前类型: {type(actual_qa_pairs)}")
                    
#                     if len(actual_qa_pairs) == 0:
#                         raise ValueError("Q&A对列表为空")
                    
#                     logger.info(f"成功提取 {len(actual_qa_pairs)} 个Q&A对")
                    
#                     # 使用提取的Q&A对进行集成
#                     result = integrator.integrate_qa_dataset(actual_qa_pairs, index_name)
                    
#                 except json.JSONDecodeError as e:
#                     raise ValueError(f"无法解析qa_dataset JSON: {e}")
#                 except Exception as e:
#                     raise ValueError(f"处理qa_dataset时出错: {e}")
#             else:
#                 result = integrator.integrate_qa_dataset([], index_name)
            
#         elif action == "search":
#             # 搜索医疗知识库
#             if not index_name:
#                 raise ValueError("搜索需要指定index_name")
#             if not query:
#                 raise ValueError("搜索需要提供query")
            
#             result = integrator.search_medical_knowledge(index_name, query, search_type, top_k)
            
#         elif action == "list_indices":
#             # 列出所有医疗知识库索引
#             result = integrator.get_medical_indices()
            
#         elif action == "delete_index":
#             # 删除医疗知识库索引
#             if not index_name:
#                 raise ValueError("删除索引需要指定index_name")
            
#             result = integrator.delete_medical_index(index_name)
            
#         else:
#             raise ValueError(f"不支持的操作类型: {action}。支持的操作: create_index, integrate_data, store, search, list_indices, delete_index")
        
#         return json.dumps(result, ensure_ascii=False, indent=2)
        
#     except Exception as e:
#         error_result = {
#             "success": False,
#             "error": f"医疗知识库集成操作失败: {str(e)}",
#             "action": action,
#             "suggestion": "请检查参数设置和Elasticsearch连接状态"
#         }
#         return json.dumps(error_result, ensure_ascii=False, indent=2)




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
        
        # 统一对比 basename，提升匹配鲁棒性
        target_name = os.path.basename(ui_filename) if ui_filename else ui_filename
        
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
                            if file_name and os.path.basename(file_name) == target_name:
                                return file_info.get('object_name', '')
                                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                continue
    
    # 如果没找到，返回 basename（作为fallback）
    return target_name
