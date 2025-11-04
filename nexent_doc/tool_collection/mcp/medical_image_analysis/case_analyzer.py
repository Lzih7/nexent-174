import io
import os
import logging
from typing import Dict, Any, Optional
from nexent.core.models.openai_vlm import OpenAIVLModel
from nexent.core.utils.observer import MessageObserver
from utils.config_utils import tenant_config_manager, get_model_name_from_config
from consts.const import MODEL_CONFIG_MAPPING
from .image_processor import MedicalImageProcessor
from .medical_prompts import CASE_ANALYSIS_PROMPTS, SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

class MedicalCaseAnalyzer:
    """医疗病例分析器"""
    
    def __init__(self, tenant_id: str = None):
        self.tenant_id = tenant_id
        self.image_processor = MedicalImageProcessor()
        self.observer = MessageObserver()
        self.vlm_model = None
        
    def _init_vlm_model(self):
        """初始化视觉语言模型"""
        if self.vlm_model is None:
            try:
                # 添加调试日志 - 检查 tenant_id
                logger.info(f"[DEBUG] 开始初始化VLM模型，tenant_id: {self.tenant_id}")
                logger.info(f"[DEBUG] tenant_id 类型: {type(self.tenant_id)}")
                logger.info(f"[DEBUG] tenant_id 是否为空: {self.tenant_id is None}")
                logger.info(f"[DEBUG] MODEL_CONFIG_MAPPING['vlm']: {MODEL_CONFIG_MAPPING.get('vlm', 'KEY_NOT_FOUND')}")
                
                # 获取 VLM 模型配置
                vlm_config = tenant_config_manager.get_model_config(
                    MODEL_CONFIG_MAPPING["vlm"], 
                    tenant_id=self.tenant_id
                )
                
                # 添加调试日志 - 检查配置获取结果
                logger.info(f"[DEBUG] 获取到的VLM配置: {vlm_config}")
                logger.info(f"[DEBUG] 配置是否为空: {vlm_config is None}")
                if vlm_config:
                    logger.info(f"[DEBUG] 配置内容: {list(vlm_config.keys()) if isinstance(vlm_config, dict) else 'NOT_DICT'}")
                
                if not vlm_config:
                    # 添加更详细的错误信息
                    logger.error(f"[DEBUG] 未找到租户 {self.tenant_id} 的 VLM 模型配置")
                    logger.error(f"[DEBUG] 请检查数据库中 tenant_config_t 表是否存在 tenant_id='{self.tenant_id}' 且 config_key='{MODEL_CONFIG_MAPPING['vlm']}' 的记录")
                    raise ValueError(f"未找到租户 {self.tenant_id} 的 VLM 模型配置")
                
                # 使用 get_model_name_from_config 函数正确组合模型名称
                model_name = get_model_name_from_config(vlm_config)
                api_key = vlm_config.get("api_key")
                base_url = vlm_config.get("base_url")
                
                # 添加调试日志 - 检查配置参数
                logger.info(f"[DEBUG] 组合后的完整模型名称: {model_name}")
                logger.info(f"[DEBUG] model_repo: {vlm_config.get('model_repo', 'EMPTY')}")
                logger.info(f"[DEBUG] model_name (原始): {vlm_config.get('model_name', 'EMPTY')}")
                logger.info(f"[DEBUG] api_key: {'***' if api_key else None}")
                logger.info(f"[DEBUG] base_url: {base_url}")
                
                if not model_name:
                    logger.error("[DEBUG] VLM 模型配置中缺少 model_name")
                    raise ValueError("VLM 模型配置中缺少 model_name")
                if not api_key:
                    logger.error("[DEBUG] VLM 模型配置中缺少 api_key")
                    raise ValueError("VLM 模型配置中缺少 api_key")
                if not base_url:
                    logger.error("[DEBUG] VLM 模型配置中缺少 base_url")
                    raise ValueError("VLM 模型配置中缺少 base_url")
                
                # 初始化 VLM 模型 - 使用正确的参数
                logger.info(f"[DEBUG] 开始初始化 OpenAIVLModel")
                self.vlm_model = OpenAIVLModel(
                    observer=self.observer,
                    model_id=model_name,  # 使用组合后的完整模型名称
                    api_base=base_url,
                    api_key=api_key,
                    temperature=0.7,
                    top_p=0.7,
                    frequency_penalty=0.5,
                    max_tokens=512
                )
                logger.info(f"[DEBUG] 视觉语言模型初始化成功: {model_name}")
                
            except Exception as e:
                logger.error(f"[DEBUG] 视觉语言模型初始化失败: {e}")
                logger.error(f"[DEBUG] 异常类型: {type(e).__name__}")
                import traceback
                logger.error(f"[DEBUG] 完整堆栈跟踪:\n{traceback.format_exc()}")
                raise
    def analyze_medical_image_from_stream(self, 
                        image_stream: io.BytesIO, 
                        analysis_type: str = "general_analysis",
                        custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """从图像流分析医学图像
        
        Args:
            image_stream: 图像数据流
            analysis_type: 分析类型 (general_analysis, x_ray_analysis, ct_mri_analysis)
            custom_prompt: 自定义提示词
            
        Returns:
            分析结果字典
        """
        try:
            # 验证图像流
            if not self.image_processor.validate_image_stream(image_stream):
                return {
                    "success": False,
                    "error": "无效的图像文件或格式不支持",
                    "image_info": None,
                    "analysis": None
                }
            
            # 获取图像信息
            image_info = self.image_processor.get_image_info_from_stream(image_stream)
            
            # 初始化模型
            self._init_vlm_model()
            
            # 选择提示词
            if custom_prompt:
                system_prompt = custom_prompt
            else:
                prompt_config = CASE_ANALYSIS_PROMPTS.get(analysis_type, CASE_ANALYSIS_PROMPTS["general_analysis"])
                system_prompt = prompt_config["system_prompt"]
            
            # 将图像流保存为临时文件用于VLM分析
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                image_stream.seek(0)
                temp_file.write(image_stream.read())
                temp_file_path = temp_file.name
            
            try:
                # 分析图像
                analysis_result = self.vlm_model.analyze_image(
                    image_input=temp_file_path,
                    system_prompt=system_prompt
                )
                
                # 将ChatMessage对象转换为可序列化的格式
                if hasattr(analysis_result, 'content'):
                    analysis_content = analysis_result.content
                else:
                    analysis_content = str(analysis_result)
                
                # 构建完整结果
                result = {
                    "success": True,
                    "image_info": image_info,
                    "analysis_type": analysis_type,
                    "analysis": analysis_content,  # 使用转换后的内容而不是原始ChatMessage对象
                    "disclaimer": SAFETY_DISCLAIMER,
                    "timestamp": self._get_timestamp()
                }
                
                logger.info(f"医学图像流分析完成")
                return result
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_e:
                    logger.warning(f"清理临时文件失败: {cleanup_e}")
                
        except Exception as e:
            logger.error(f"医学图像流分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_info": None,
                "analysis": None
            }
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")