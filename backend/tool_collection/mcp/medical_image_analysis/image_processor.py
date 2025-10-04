import base64
import io
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import logging

# 添加 MinIO 支持
try:
    from database.attachment_db import get_file_stream
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("MinIO 支持不可用，仅支持本地文件")

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """医学图像处理工具类，支持本地文件和 MinIO 存储"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        self.max_image_size = (2048, 2048)  # 最大图像尺寸
    
    def _is_minio_path(self, path: str) -> bool:
        """判断是否为 MinIO 对象路径"""
        # MinIO 路径通常不以 / 开头，且不包含操作系统路径分隔符
        return not os.path.isabs(path) and ('/' in path or not os.path.exists(path))
    
    def _get_image_stream(self, image_path: str) -> Optional[io.BytesIO]:
        """获取图像流，支持本地文件和 MinIO"""
        try:
            logger.info(f"开始获取图像流: {image_path}")
            logger.info(f"MINIO_AVAILABLE: {MINIO_AVAILABLE}")
            
            is_minio = self._is_minio_path(image_path)
            logger.info(f"判断为 MinIO 路径: {is_minio}")
            logger.info(f"路径是否为绝对路径: {os.path.isabs(image_path)}")
            logger.info(f"本地文件是否存在: {os.path.exists(image_path)}")
            
            if is_minio and MINIO_AVAILABLE:
                # 从 MinIO 获取文件流
                logger.info(f"从 MinIO 读取图像: {image_path}")
                try:
                    file_stream = get_file_stream(image_path)
                    logger.info(f"get_file_stream 返回结果: {file_stream is not None}")
                    if file_stream is None:
                        logger.error(f"无法从 MinIO 获取文件流: {image_path}")
                        return None
                    logger.info(f"成功从 MinIO 获取文件流，类型: {type(file_stream)}")
                    return file_stream
                except Exception as minio_e:
                    logger.error(f"MinIO 获取文件流异常: {minio_e}")
                    return None
            else:
                # 从本地文件系统读取
                logger.info(f"从本地文件系统读取图像: {image_path}")
                if not os.path.exists(image_path):
                    logger.error(f"本地文件不存在: {image_path}")
                    return None
                try:
                    with open(image_path, 'rb') as f:
                        data = f.read()
                        logger.info(f"成功读取本地文件，大小: {len(data)} 字节")
                        return io.BytesIO(data)
                except Exception as local_e:
                    logger.error(f"本地文件读取异常: {local_e}")
                    return None
        except Exception as e:
            logger.error(f"获取图像流失败: {e}")
            return None
    
    def validate_image(self, image_path: str) -> bool:
        """验证图像文件是否有效，支持本地文件和 MinIO"""
        try:
            image_stream = self._get_image_stream(image_path)
            if image_stream is None:
                return False
            
            with Image.open(image_stream) as img:
                # 检查图像格式 - 修复格式检查逻辑
                valid_formats = ['jpeg','jpg', 'png', 'bmp', 'tiff']
                if img.format.lower() not in valid_formats:
                    logger.error(f"不支持的图像格式: {img.format}")
                    return False
                
                # 检查图像尺寸
                if img.size[0] > 4096 or img.size[1] > 4096:
                    logger.warning(f"图像尺寸过大: {img.size}")
                
                return True
        except Exception as e:
            logger.error(f"图像验证失败: {e}")
            return False
    
    def preprocess_image(self, image_path: str) -> Optional[str]:
        """预处理医学图像并转换为base64，支持本地文件和 MinIO"""
        try:
            image_stream = self._get_image_stream(image_path)
            if image_stream is None:
                return None
            
            with Image.open(image_stream) as img:
                # 转换为RGB模式（如果需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图像大小（如果太大）
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                
                # 转换为base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """获取图像基本信息，支持本地文件和 MinIO"""
        try:
            image_stream = self._get_image_stream(image_path)
            if image_stream is None:
                return {}
            
            with Image.open(image_stream) as img:
                info = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'source_type': 'minio' if self._is_minio_path(image_path) else 'local'
                }
                return info
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            return {}
    
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
    
    def preprocess_image_stream(self, image_stream: io.BytesIO) -> Optional[str]:
        """预处理图像流并转换为base64"""
        try:
            image_stream.seek(0)  # 确保从头开始读取
            with Image.open(image_stream) as img:
                # 转换为RGB模式（如果需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图像大小（如果太大）
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                
                # 转换为base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
        except Exception as e:
            logger.error(f"图像流预处理失败: {e}")
            return None