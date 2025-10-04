"""
医疗数据清洗模块
专门用于处理病理教科书等医疗专业素材的数据清洗
"""

import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MedicalDataCleaner:
    """医疗数据清洗器"""
    
    def __init__(self):
        # 医学术语正则表达式模式
        self.medical_patterns = {
            # 疾病名称模式
            'diseases': r'(?:癌|肿瘤|炎|症|病|综合征|缺陷|畸形|损伤|破裂|出血|梗死|坏死)',
            # 解剖结构模式
            'anatomy': r'(?:心脏|肺|肝|肾|脑|胃|肠|骨|肌肉|神经|血管|淋巴)',
            # 病理术语模式
            'pathology': r'(?:良性|恶性|转移|浸润|增生|萎缩|纤维化|钙化|囊性|实性)',
            # 诊断术语模式
            'diagnosis': r'(?:诊断|鉴别诊断|临床表现|症状|体征|检查|治疗|预后)',
            # 组织学特征模式
            'histology': r'(?:细胞|组织|上皮|间质|胶原|弹性纤维|血管|神经纤维)'
        }
        
        # 需要清理的噪声模式
        self.noise_patterns = [
            r'\[图\s*\d+[-\.\d]*\]',  # 图片引用
            r'\[表\s*\d+[-\.\d]*\]',  # 表格引用
            r'第\s*\d+\s*页',         # 页码
            r'参考文献\s*\[\d+\]',     # 参考文献
            r'见附录\s*[A-Z]',        # 附录引用
            r'\s{3,}',               # 多余空格
            r'\n{3,}',               # 多余换行
        ]
    
    def clean_medical_text(self, text: str) -> Dict[str, Any]:
        """清洗医疗文本数据
        
        Args:
            text: 原始医疗文本
            
        Returns:
            清洗结果字典
        """
        try:
            original_length = len(text)
            
            # 1. 移除噪声
            cleaned_text = self._remove_noise(text)
            
            # 2. 标准化格式
            cleaned_text = self._normalize_format(cleaned_text)
            
            # 3. 提取医学术语
            medical_terms = self._extract_medical_terms(cleaned_text)
            
            # 4. 分段处理
            segments = self._segment_text(cleaned_text)
            
            # 5. 质量评估
            quality_score = self._assess_quality(cleaned_text, medical_terms)
            
            result = {
                'success': True,
                'original_length': original_length,
                'cleaned_length': len(cleaned_text),
                'cleaned_text': cleaned_text,
                'medical_terms': medical_terms,
                'segments': segments,
                'quality_score': quality_score,
                'cleaning_stats': {
                    'noise_removed': original_length - len(cleaned_text),
                    'medical_terms_count': len(medical_terms),
                    'segments_count': len(segments)
                }
            }
            
            logger.info(f"医疗文本清洗完成，质量评分: {quality_score}")
            return result
            
        except Exception as e:
            logger.error(f"医疗文本清洗失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'cleaned_text': text,
                'medical_terms': [],
                'segments': [],
                'quality_score': 0.0
            }
    
    def _remove_noise(self, text: str) -> str:
        """移除文本噪声"""
        cleaned = text
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # 清理多余的空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _normalize_format(self, text: str) -> str:
        """标准化文本格式"""
        # 统一标点符号
        text = re.sub(r'[，,]', '，', text)
        text = re.sub(r'[。。]', '。', text)
        text = re.sub(r'[；；]', '；', text)
        text = re.sub(r'[：：]', '：', text)
        
        # 统一括号
        text = re.sub(r'[（(]', '（', text)
        text = re.sub(r'[）)]', '）', text)
        
        return text
    
    def _extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """提取医学术语"""
        terms = {}
        
        for category, pattern in self.medical_patterns.items():
            matches = re.findall(f'[^，。；：\s]*{pattern}[^，。；：\s]*', text)
            # 去重并过滤短词
            unique_matches = list(set([m for m in matches if len(m) >= 2]))
            terms[category] = unique_matches
        
        return terms
    
    def _segment_text(self, text: str) -> List[Dict[str, Any]]:
        """分段处理文本"""
        # 按句号分段
        sentences = re.split(r'[。！？]', text)
        segments = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # 过滤过短的句子
                continue
                
            segment = {
                'id': i,
                'content': sentence,
                'length': len(sentence),
                'medical_terms': self._extract_medical_terms(sentence),
                'segment_type': self._classify_segment(sentence)
            }
            segments.append(segment)
        
        return segments
    
    def _classify_segment(self, text: str) -> str:
        """分类文本段落类型"""
        if re.search(r'定义|概念|是指', text):
            return 'definition'
        elif re.search(r'症状|表现|特征', text):
            return 'symptoms'
        elif re.search(r'诊断|检查|鉴别', text):
            return 'diagnosis'
        elif re.search(r'治疗|用药|手术', text):
            return 'treatment'
        elif re.search(r'病因|发病机制', text):
            return 'etiology'
        else:
            return 'general'
    
    def _assess_quality(self, text: str, medical_terms: Dict[str, List[str]]) -> float:
        """评估文本质量"""
        score = 0.0
        
        # 长度评分 (0-30分)
        length_score = min(len(text) / 1000 * 30, 30)
        score += length_score
        
        # 医学术语丰富度评分 (0-40分)
        total_terms = sum(len(terms) for terms in medical_terms.values())
        terms_score = min(total_terms / 20 * 40, 40)
        score += terms_score
        
        # 结构完整性评分 (0-30分)
        structure_indicators = ['定义', '症状', '诊断', '治疗']
        structure_score = sum(10 for indicator in structure_indicators if indicator in text)
        score += min(structure_score, 30)
        
        return round(score / 100, 2)

    def batch_clean_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量清洗文件"""
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                # 读取文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 清洗数据
                result = self.clean_medical_text(content)
                result['file_path'] = file_path
                result['file_index'] = i + 1
                
                results.append(result)
                logger.info(f"已处理文件 {i+1}/{total_files}: {file_path}")
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'file_path': file_path,
                    'file_index': i + 1,
                    'error': str(e)
                }
                results.append(error_result)
                logger.error(f"处理文件失败 {file_path}: {e}")
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        failed = total_files - successful
        
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'results': results,
            'summary': {
                'success_rate': successful / total_files if total_files > 0 else 0,
                'average_quality': sum(r.get('quality_score', 0) for r in results if r.get('success')) / successful if successful > 0 else 0
            }
        }