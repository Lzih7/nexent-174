import base64
import io
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import logging
logger = logging.getLogger(__name__)

# 添加 MinIO 支持
try:
    from database.attachment_db import get_file_stream
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("MinIO 支持不可用，仅支持本地文件")



class MedicalImageProcessor:
    """医学图像处理工具类，支持本地文件和 MinIO 存储"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        self.max_image_size = (2048, 2048)  # 最大图像尺寸
    
    
    def validate_image_stream(self, image_stream: io.BytesIO) -> bool:
        """验证图像流是否有效"""
        try:
            image_stream.seek(0)  # 确保从头开始读取
            with Image.open(image_stream) as img:
                # 检查图像格式
                valid_formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
                if img.format not in valid_formats:
                    logger.error(f"不支持的图像格式: {img.format}")
                    return False
                
                # 检查图像尺寸
                if img.size[0] > 4096 or img.size[1] > 4096:
                    logger.warning(f"图像尺寸过大: {img.size}")
                
                return True
        except Exception as e:
            logger.error(f"图像流验证失败: {e}")
            return False
    
    def get_image_info_from_stream(self, image_stream: io.BytesIO) -> Dict[str, Any]:
        """从图像流获取基本信息"""
        try:
            image_stream.seek(0)  # 确保从头开始读取
            with Image.open(image_stream) as img:
                info = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'source_type': 'stream'
                }
                return info
        except Exception as e:
            logger.error(f"从图像流获取信息失败: {e}")
            return {}
    