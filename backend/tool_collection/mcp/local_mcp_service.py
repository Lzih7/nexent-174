from fastmcp import FastMCP
from .medical_image_analysis import MedicalCaseAnalyzer
from .medical_data_processing.data_cleaner import MedicalDataCleaner
from .medical_data_processing.professional_annotator import MedicalProfessionalAnnotator
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

local_mcp_service = FastMCP("local")
logger = logging.getLogger(__name__)

def get_file_stream_with_size_limit(minio_object_name: str, max_bytes: int) -> Optional[BinaryIO]:
    """
    从MinIO获取文件流并限制读取大小
    
    Args:
        minio_object_name: MinIO对象名
        max_bytes: 最大读取字节数
        
    Returns:
        限制大小的文件流，如果失败返回None
    """
    try:
        # 使用SDK的get_file_stream获取完整流
        file_stream = get_file_stream(minio_object_name)
        if not file_stream:
            return None
        
        # 读取限制大小的数据
        limited_data = file_stream.read(max_bytes)
        file_stream.close()  # 关闭原始流
        
        print(f"从MinIO读取了 {len(limited_data)} 字节数据（限制: {max_bytes} 字节）")
        
        # 返回包装后的BytesIO流
        return io.BytesIO(limited_data)
        
    except Exception as e:
        print(f"读取MinIO文件流时出错: {e}")
        return None

async def _read_pdf_from_local(file_path: str, max_chars: int = 30000) -> str:
    """从本地读取PDF文件内容，限制最大字符数
    
    Args:
        file_path: PDF文件路径
        max_chars: 最大字符数限制，默认30000字符（约15000-20000 tokens）
    
    Returns:
        截取后的PDF内容
    """
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            
            logger.info(f"开始读取本地PDF: {file_path}，总页数: {len(reader.pages)}")
            
            # 逐页读取，直到达到字符限制
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() + "\n"
                    
                    # 检查是否会超过限制
                    if len(content) + len(page_text) > max_chars:
                        # 计算剩余可用字符数
                        remaining_chars = max_chars - len(content)
                        if remaining_chars > 0:
                            # 截取部分页面内容
                            content += page_text[:remaining_chars]
                            logger.info(f"PDF内容已截取到{max_chars}字符，处理了{page_num + 1}页中的部分内容")
                        else:
                            logger.info(f"PDF内容已截取到{max_chars}字符，处理了{page_num}页完整内容")
                        break
                    else:
                        content += page_text
                        
                except Exception as page_error:
                    logger.warning(f"读取第{page_num + 1}页时出错: {page_error}，跳过该页")
                    continue
            
            logger.info(f"PDF读取完成：总字符数={len(content)}，成功处理页数={page_num + 1}/{len(reader.pages)}")
            
            if not content.strip():
                logger.warning("PDF内容为空，可能是扫描版PDF或加密PDF")
                raise Exception("PDF内容提取失败：文档可能是扫描版或受保护的PDF")
                
            return content
            
    except ImportError as e:
        logger.error(f"PyPDF2库未安装: {e}")
        raise ImportError("需要安装PyPDF2库。请检查Docker容器中的依赖安装。")
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise FileNotFoundError(f"PDF文件不存在: {file_path}")
    except Exception as e:
        logger.error(f"PDF读取失败: {e}")
        if "encrypted" in str(e).lower():
            raise Exception("PDF处理错误: 文档已加密，无法读取")
        elif "corrupted" in str(e).lower() or "invalid" in str(e).lower():
            raise Exception("PDF处理错误: 文档已损坏或格式无效")
        else:
            raise Exception(f"PDF处理错误: {str(e)}")

async def _read_pdf_from_stream(file_stream, max_chars: int = None) -> str:
    """从文件流读取PDF内容，可选择性限制最大字符数
    
    Args:
        file_stream: PDF文件流
        max_chars: 最大字符数限制，如果为None则不限制
    
    Returns:
        PDF内容（如果设置了max_chars则可能被截取）
    """
    try:
        import PyPDF2
        
        # 将文件流内容读取到内存
        pdf_bytes = file_stream.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        
        reader = PyPDF2.PdfReader(pdf_stream)
        content = ""
        
        logger.info(f"开始读取PDF，总页数: {len(reader.pages)}")
        
        # 逐页读取
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() + "\n"
                
                # 如果设置了字符限制，检查是否会超过限制
                if max_chars is not None and len(content) + len(page_text) > max_chars:
                    # 计算剩余可用字符数
                    remaining_chars = max_chars - len(content)
                    if remaining_chars > 0:
                        # 截取部分页面内容
                        content += page_text[:remaining_chars]
                        logger.info(f"PDF内容已截取到{max_chars}字符，处理了{page_num + 1}页中的部分内容")
                    else:
                        logger.info(f"PDF内容已截取到{max_chars}字符，处理了{page_num}页完整内容")
                    break
                else:
                    content += page_text
                    
            except Exception as page_error:
                logger.warning(f"读取第{page_num + 1}页时出错: {page_error}，跳过该页")
                continue
        
        logger.info(f"PDF读取完成：总字符数={len(content)}，成功处理页数={page_num + 1}/{len(reader.pages)}")
        
        if not content.strip():
            logger.warning("PDF内容为空，可能是扫描版PDF或加密PDF")
            raise Exception("PDF内容提取失败：文档可能是扫描版或受保护的PDF")
            
        return content
        
    except ImportError as e:
        logger.error(f"PyPDF2库未安装: {e}")
        raise ImportError("需要安装PyPDF2库。请检查Docker容器中的依赖安装。当前环境可能缺少PDF处理依赖。")
    except Exception as e:
        logger.error(f"PDF读取失败: {e}")
        if "encrypted" in str(e).lower():
            raise Exception("PDF处理错误: 文档已加密，无法读取")
        elif "corrupted" in str(e).lower() or "invalid" in str(e).lower():
            raise Exception("PDF处理错误: 文档已损坏或格式无效")
        else:
            raise Exception(f"PDF处理错误: {str(e)}")

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
        logger.debug(f"获取到的租户ID: {tenant_id}")
        logger.debug(f"租户ID类型: {type(tenant_id)}")
        logger.debug(f"租户ID是否为空: {tenant_id is None}")
        
        # 初始化医疗病例分析器
        logger.debug(f"开始初始化 MedicalCaseAnalyzer，传入 tenant_id: {tenant_id}")
        analyzer = MedicalCaseAnalyzer(tenant_id=tenant_id)
        logger.debug("MedicalCaseAnalyzer 初始化完成")
        
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
            
            if result and hasattr(result[0], 'tenant_id') and result[0].tenant_id:
                actual_tenant_id = result[0].tenant_id
                logger.debug(f"从数据库获取到的 tenant_id: {actual_tenant_id}")
                return actual_tenant_id
            else:
                logger.warning(f"无法从数据库获取 tenant_id，使用默认值: {DEFAULT_TENANT_ID}")
                return DEFAULT_TENANT_ID  # 确保返回默认值而不是 None
                
    except Exception as e:
        logger.error(f"从数据库获取 tenant_id 时出错: {e}")
        logger.warning(f"异常情况下使用默认值: {DEFAULT_TENANT_ID}")
        return DEFAULT_TENANT_ID  # 异常情况下也返回默认值


@local_mcp_service.tool(name="medical_data_cleaner",
                        description="医疗数据清洗工具，专门用于清洗病理教科书等医疗专业素材，去除噪声并提取医学术语")
async def medical_data_cleaner(text_content: str = "",
                             file_path: str = "",
                             max_chars: int = None) -> str:  # 改为None，表示不限制
    """
    医疗数据清洗工具
    
    Args:
        text_content: 直接输入的医疗文本内容
        file_path: 单个文件路径（必须是已上传到MinIO的文件）
        max_chars: 文件最大字符数限制，如果为None则不限制
    
    Returns:
        JSON格式的清洗结果
    """
    
    logger.info(f"原始文件名: {file_path}")
    
    try:
        cleaner = MedicalDataCleaner()
        
        if file_path:
            # 处理单个文件
            logger.info(f"原始文件名: {file_path}")
            
            # 获取 MinIO 对象名
            minio_object_name = find_minio_object_name_by_ui_filename(file_path)
            logger.info(f"MinIO对象名: {minio_object_name}")
            
            # 只支持从MinIO读取文件，找不到就报异常
            if not minio_object_name or minio_object_name == file_path:
                raise FileNotFoundError(f"无法在MinIO中找到文件: {file_path}，请确保文件已正确上传")
            
            # 从 MinIO 获取文件流，不限制字节数
            file_stream = get_file_stream(minio_object_name)
            
            if not file_stream:
                raise FileNotFoundError(f"无法从MinIO获取文件流: {minio_object_name}")
            
            # 根据文件扩展名处理不同类型的文件
            if file_path.lower().endswith('.pdf'):
                content = await _read_pdf_from_stream(file_stream, max_chars)
            else:
                # 读取文本文件内容
                content = file_stream.read().decode('utf-8')
                # 对文本文件进行长度限制（如果设置了max_chars）
                if max_chars is not None and len(content) > max_chars:
                    content = content[:max_chars]
                    logger.info(f"文本内容已截取到{max_chars}字符")
            
            logger.info("从MinIO读取内容成功")
            
            result = cleaner.clean_medical_text(content)
            result['source_file'] = file_path
            result['content_truncated'] = max_chars is not None and len(content) >= max_chars
            result['final_content_length'] = len(content)
        elif text_content:
            # 处理直接输入的文本，进行长度限制（如果设置了max_chars）
            if max_chars is not None and len(text_content) > max_chars:
                text_content = text_content[:max_chars]
                logger.info(f"输入文本已截取到{max_chars}字符")
            
            result = cleaner.clean_medical_text(text_content)
            result['source_type'] = 'direct_input'
            result['content_truncated'] = max_chars is not None and len(text_content) >= max_chars
            result['final_content_length'] = len(text_content)
        else:
            raise ValueError("必须提供text_content或file_path中的一个参数")
        
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
                              content_type: str = "general",
                              max_chars: int = 20000) -> str:  # 从50000减少到20000
    """
    医疗数据工程流水线工具
    
    Args:
        input_content: 输入的原始医疗文本
        input_file_path: 输入文件路径
        qa_count: 生成的Q&A对数量
        content_type: 内容类型
        max_chars: PDF文件最大字符数限制，默认50000字符
    
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
            print(f"原始文件名: {input_file_path}")
            
            # 获取 MinIO 对象名
            minio_object_name = find_minio_object_name_by_ui_filename(input_file_path)
            print(f"MinIO对象名: {minio_object_name}")
            
            if not minio_object_name or minio_object_name == input_file_path:
                # 如果找不到MinIO对象，尝试本地文件
                if not os.path.exists(input_file_path):
                    raise FileNotFoundError(f"文件不存在: {input_file_path}，且无法在MinIO中找到对应文件")
                
                # 根据文件扩展名处理不同类型的文件
                if input_file_path.lower().endswith('.pdf'):
                    raw_content = await _read_pdf_from_local(input_file_path, max_chars)  # 添加长度限制
                else:
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        raw_content = f.read()
                        # 对文本文件也进行长度限制
                        if len(raw_content) > max_chars:
                            raw_content = raw_content[:max_chars]
                            print(f"文本内容已截取到{max_chars}字符")
                print("从本地文件读取内容")
            else:
                # 从 MinIO 获取文件流
                file_stream = get_file_stream(minio_object_name)
                
                if not file_stream:
                    raise FileNotFoundError(f"无法从MinIO获取文件流: {minio_object_name}")
                
                # 根据文件扩展名处理不同类型的文件
                if input_file_path.lower().endswith('.pdf'):
                    raw_content = await _read_pdf_from_stream(file_stream, max_chars)  # 添加长度限制
                else:
                    # 读取文本文件内容
                    raw_content = file_stream.read().decode('utf-8')
                    # 对文本文件也进行长度限制
                    if len(raw_content) > max_chars:
                        raw_content = raw_content[:max_chars]
                        print(f"文本内容已截取到{max_chars}字符")
                print("从MinIO读取内容成功")
                
        elif input_content:
            raw_content = input_content
            # 对直接输入的内容也进行长度限制
            if len(raw_content) > max_chars:
                raw_content = raw_content[:max_chars]
                print(f"输入内容已截取到{max_chars}字符")
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
        # 获取 tenant_id
        tenant_id = get_recent_tenant_id_from_conversation()
        print(f"[DEBUG] 医疗知识库集成器使用的租户ID: {tenant_id}")
        
        integrator = MedicalKnowledgeBaseIntegrator(tenant_id=tenant_id)
        
        if action == "create_index":
            # 创建医疗知识库索引
            result = integrator.create_medical_knowledge_index(index_name if index_name else None)
            
        elif action == "integrate_data" or action == "store":
            # 集成Q&A数据到知识库 (store是integrate_data的别名)
            if not index_name:
                raise ValueError("集成数据需要指定index_name")
            if not qa_dataset:
                raise ValueError("集成数据需要提供qa_dataset")
            
            # 解析Q&A数据集 - 修复JSON解析逻辑
            qa_data = None  # 初始化变量
            try:
                print(f"[DEBUG] 原始qa_dataset类型: {type(qa_dataset)}")
                print(f"[DEBUG] 原始qa_dataset长度: {len(qa_dataset) if hasattr(qa_dataset, '__len__') else 'N/A'}")
                
                # 如果是字符串，尝试解析JSON
                if isinstance(qa_dataset, str):
                    qa_data = json.loads(qa_dataset)
                    print(f"[DEBUG] JSON解析成功，解析后类型: {type(qa_data)}")
                else:
                    qa_data = qa_dataset
                    print(f"[DEBUG] 直接使用原始数据，类型: {type(qa_data)}")
                
                # 添加数据验证
                if qa_data is None:
                    raise ValueError("qa_data为None，无法处理")
                
                print(f"[DEBUG] 准备调用integrate_qa_dataset，参数: index_name={index_name}, qa_data类型={type(qa_data)}")
                
            except json.JSONDecodeError as e:
                raise ValueError(f"无法解析qa_dataset JSON: {e}")
            except Exception as e:
                raise ValueError(f"处理qa_dataset时出错: {e}")
            
            # 调用集成方法
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


@local_mcp_service.tool(name="elasticsearch_document_browser",
                        description="Elasticsearch文档浏览工具，用于查看医疗知识库中存储的Q&A文档内容，支持分页浏览和内容预览")
async def elasticsearch_document_browser(index_name: str = "",
                                       limit: int = 10,
                                       offset: int = 0,
                                       show_content: bool = False) -> str:
    """
    Elasticsearch文档浏览工具
    
    Args:
        index_name: 要浏览的索引名称，为空则显示所有可用的医疗索引
        limit: 返回文档数量限制 (默认10，最大100)
        offset: 偏移量，用于分页浏览 (默认0)
        show_content: 是否显示文档完整内容 (默认False，只显示摘要)
    
    Returns:
        JSON格式的文档浏览结果
    """
    try:
        # 限制参数范围
        limit = max(1, min(limit, 100))  # 限制在1-100之间
        offset = max(0, offset)  # 不能小于0
        
        integrator = MedicalKnowledgeBaseIntegrator()
        result = integrator.browse_medical_documents(
            index_name=index_name,
            limit=limit,
            offset=offset,
            show_content=show_content
        )
        
        # 添加使用提示
        if result.get('success'):
            if not index_name and result.get('total_indices', 0) > 1:
                result['usage_tip'] = "发现多个医疗索引，请使用 index_name 参数指定要浏览的索引"
            elif result.get('has_more'):
                next_offset = offset + limit
                result['usage_tip'] = f"还有更多文档，使用 offset={next_offset} 查看下一页"
            elif result.get('total_documents', 0) > 0:
                result['usage_tip'] = "使用 show_content=true 查看完整文档内容"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"Elasticsearch文档浏览工具执行失败: {str(e)}"
        print(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "documents": []
        }, ensure_ascii=False, indent=2)

