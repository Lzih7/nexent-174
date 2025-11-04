"""
医疗数据工程模块
用于处理病理教科书等专业医疗素材，生成高质量Q&A数据集
"""

from .qa_generator import MedicalQAGenerator

__all__ = [
    'MedicalQAGenerator'
]