"""
医疗专业标注模块
用于对清洗后的医疗数据进行专业标注
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalProfessionalAnnotator:
    """医疗专业标注器"""
    
    def __init__(self):
        # 病理学术语词典
        self.pathology_terms = {
            '肿瘤分类': {
                '良性肿瘤': ['腺瘤', '纤维瘤', '脂肪瘤', '血管瘤', '神经瘤'],
                '恶性肿瘤': ['癌', '肉瘤', '白血病', '淋巴瘤', '黑色素瘤'],
                '交界性肿瘤': ['交界性肿瘤', '潜在恶性肿瘤']
            },
            '疾病分类': {
                '炎症性疾病': ['炎症', '感染', '脓肿', '蜂窝织炎'],
                '退行性疾病': ['萎缩', '变性', '坏死', '纤维化'],
                '发育异常': ['畸形', '发育不良', '异位', '重复']
            },
            '组织学特征': {
                '细胞特征': ['异型性', '多形性', '核分裂', '核质比'],
                '组织结构': ['腺体结构', '鳞状上皮', '间质反应', '血管侵犯'],
                '特殊染色': ['HE染色', '免疫组化', '特殊染色', '分子标记']
            }
        }
        
        # 诊断相关术语
        self.diagnosis_terms = {
            '临床表现': ['症状', '体征', '主诉', '现病史'],
            '辅助检查': ['实验室检查', '影像学检查', '病理检查', '内镜检查'],
            '诊断方法': ['临床诊断', '病理诊断', '影像诊断', '鉴别诊断'],
            '治疗方案': ['手术治疗', '药物治疗', '放疗', '化疗', '免疫治疗']
        }
        
        # 严重程度分级
        self.severity_levels = {
            '轻度': ['轻微', '少量', '局限性', '早期'],
            '中度': ['中等', '部分', '局部', '进展期'],
            '重度': ['严重', '广泛', '弥漫性', '晚期']
        }
    
    def annotate_medical_content(self, content: str, content_type: str = 'general') -> Dict[str, Any]:
        """对医疗内容进行专业标注
        
        Args:
            content: 医疗文本内容
            content_type: 内容类型 (general, pathology, diagnosis, treatment)
            
        Returns:
            标注结果字典
        """
        try:
            # 1. 基础信息提取
            basic_info = self._extract_basic_info(content)
            
            # 2. 病理学术语标注
            pathology_annotations = self._annotate_pathology_terms(content)
            
            # 3. 疾病诊断标注
            diagnosis_annotations = self._annotate_diagnosis_terms(content)
            
            # 4. 组织学特征标注
            histology_annotations = self._annotate_histology_features(content)
            
            # 5. 严重程度评估
            severity_assessment = self._assess_severity(content)
            
            # 6. 关键实体识别
            key_entities = self._extract_key_entities(content)
            
            # 7. 关系抽取
            relationships = self._extract_relationships(content, key_entities)
            
            # 8. 生成结构化标注
            structured_annotation = self._create_structured_annotation(
                content, pathology_annotations, diagnosis_annotations, 
                histology_annotations, key_entities, relationships
            )
            
            result = {
                'success': True,
                'content_type': content_type,
                'basic_info': basic_info,
                'annotations': {
                    'pathology': pathology_annotations,
                    'diagnosis': diagnosis_annotations,
                    'histology': histology_annotations,
                    'severity': severity_assessment,
                    'entities': key_entities,
                    'relationships': relationships
                },
                'structured_annotation': structured_annotation,
                'annotation_stats': {
                    'total_annotations': len(pathology_annotations) + len(diagnosis_annotations) + len(histology_annotations),
                    'entity_count': len(key_entities),
                    'relationship_count': len(relationships)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"医疗内容标注完成，共生成 {result['annotation_stats']['total_annotations']} 个标注")
            return result
            
        except Exception as e:
            logger.error(f"医疗内容标注失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'content_type': content_type,
                'annotations': {},
                'structured_annotation': {}
            }
    
    def _extract_basic_info(self, content: str) -> Dict[str, Any]:
        """提取基础信息"""
        return {
            'length': len(content),
            'sentence_count': len(re.findall(r'[。！？]', content)),
            'paragraph_count': len(content.split('\n\n')),
            'contains_medical_terms': bool(re.search(r'[病症疾癌肿瘤炎]', content))
        }
    
    def _annotate_pathology_terms(self, content: str) -> List[Dict[str, Any]]:
        """标注病理学术语"""
        annotations = []
        
        for category, subcategories in self.pathology_terms.items():
            for subcategory, terms in subcategories.items():
                for term in terms:
                    matches = list(re.finditer(re.escape(term), content))
                    for match in matches:
                        annotation = {
                            'term': term,
                            'category': category,
                            'subcategory': subcategory,
                            'start_pos': match.start(),
                            'end_pos': match.end(),
                            'context': content[max(0, match.start()-20):match.end()+20],
                            'confidence': self._calculate_confidence(term, content, match.start())
                        }
                        annotations.append(annotation)
        
        return annotations
    
    def _annotate_diagnosis_terms(self, content: str) -> List[Dict[str, Any]]:
        """标注诊断相关术语"""
        annotations = []
        
        for category, terms in self.diagnosis_terms.items():
            for term in terms:
                matches = list(re.finditer(re.escape(term), content))
                for match in matches:
                    annotation = {
                        'term': term,
                        'category': category,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'context': content[max(0, match.start()-20):match.end()+20],
                        'confidence': self._calculate_confidence(term, content, match.start())
                    }
                    annotations.append(annotation)
        
        return annotations
    
    def _annotate_histology_features(self, content: str) -> List[Dict[str, Any]]:
        """标注组织学特征"""
        annotations = []
        histology_patterns = [
            r'细胞[异多]型性',
            r'核分裂象',
            r'血管侵犯',
            r'淋巴结转移',
            r'免疫组化',
            r'HE染色',
            r'腺体结构',
            r'鳞状上皮'
        ]
        
        for pattern in histology_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                annotation = {
                    'feature': match.group(),
                    'pattern': pattern,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'context': content[max(0, match.start()-30):match.end()+30],
                    'feature_type': self._classify_histology_feature(match.group())
                }
                annotations.append(annotation)
        
        return annotations
    
    def _assess_severity(self, content: str) -> Dict[str, Any]:
        """评估严重程度"""
        severity_scores = {'轻度': 0, '中度': 0, '重度': 0}
        
        for level, indicators in self.severity_levels.items():
            for indicator in indicators:
                count = len(re.findall(re.escape(indicator), content))
                severity_scores[level] += count
        
        # 确定主要严重程度
        primary_severity = max(severity_scores, key=severity_scores.get)
        
        return {
            'scores': severity_scores,
            'primary_severity': primary_severity,
            'confidence': severity_scores[primary_severity] / sum(severity_scores.values()) if sum(severity_scores.values()) > 0 else 0
        }
    
    def _extract_key_entities(self, content: str) -> List[Dict[str, Any]]:
        """提取关键实体"""
        entities = []
        
        # 疾病实体
        disease_pattern = r'[^，。；：\s]*(?:病|症|癌|瘤|炎)[^，。；：\s]*'
        diseases = re.findall(disease_pattern, content)
        for disease in set(diseases):
            if len(disease) >= 2:
                entities.append({
                    'text': disease,
                    'type': 'disease',
                    'confidence': 0.8
                })
        
        # 解剖部位实体
        anatomy_pattern = r'[^，。；：\s]*(?:心|肺|肝|肾|脑|胃|肠|骨)[^，。；：\s]*'
        anatomies = re.findall(anatomy_pattern, content)
        for anatomy in set(anatomies):
            if len(anatomy) >= 2:
                entities.append({
                    'text': anatomy,
                    'type': 'anatomy',
                    'confidence': 0.7
                })
        
        return entities
    
    def _extract_relationships(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽取实体关系"""
        relationships = []
        
        # 简单的关系抽取：疾病-部位关系
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j and entity1['type'] == 'disease' and entity2['type'] == 'anatomy':
                    # 检查两个实体是否在同一句话中
                    sentences = re.split(r'[。！？]', content)
                    for sentence in sentences:
                        if entity1['text'] in sentence and entity2['text'] in sentence:
                            relationships.append({
                                'entity1': entity1['text'],
                                'entity2': entity2['text'],
                                'relation_type': 'disease_location',
                                'confidence': 0.6,
                                'context': sentence.strip()
                            })
                            break
        
        return relationships
    
    def _create_structured_annotation(self, content: str, pathology_annotations: List, 
                                    diagnosis_annotations: List, histology_annotations: List,
                                    entities: List, relationships: List) -> Dict[str, Any]:
        """创建结构化标注"""
        return {
            'content_summary': content[:200] + '...' if len(content) > 200 else content,
            'key_pathology_terms': [ann['term'] for ann in pathology_annotations[:10]],
            'key_diagnosis_terms': [ann['term'] for ann in diagnosis_annotations[:10]],
            'key_histology_features': [ann['feature'] for ann in histology_annotations[:10]],
            'primary_entities': [ent['text'] for ent in entities[:10]],
            'main_relationships': [f"{rel['entity1']} -> {rel['entity2']}" for rel in relationships[:5]],
            'annotation_quality': self._assess_annotation_quality(pathology_annotations, diagnosis_annotations, histology_annotations)
        }
    
    def _calculate_confidence(self, term: str, content: str, position: int) -> float:
        """计算标注置信度"""
        # 基础置信度
        base_confidence = 0.5
        
        # 术语长度加分
        length_bonus = min(len(term) / 10, 0.3)
        
        # 上下文相关性加分
        context = content[max(0, position-50):position+50]
        medical_context_words = ['诊断', '治疗', '症状', '病理', '临床']
        context_bonus = sum(0.05 for word in medical_context_words if word in context)
        
        return min(base_confidence + length_bonus + context_bonus, 1.0)
    
    def _classify_histology_feature(self, feature: str) -> str:
        """分类组织学特征"""
        if '细胞' in feature:
            return 'cellular'
        elif '血管' in feature:
            return 'vascular'
        elif '染色' in feature:
            return 'staining'
        elif '结构' in feature:
            return 'structural'
        else:
            return 'general'
    
    def _assess_annotation_quality(self, pathology_annotations: List, 
                                 diagnosis_annotations: List, histology_annotations: List) -> float:
        """评估标注质量"""
        total_annotations = len(pathology_annotations) + len(diagnosis_annotations) + len(histology_annotations)
        
        if total_annotations == 0:
            return 0.0
        
        # 基于标注数量和多样性计算质量分数
        diversity_score = min(len(set([ann.get('category', ann.get('feature', '')) 
                                     for ann in pathology_annotations + diagnosis_annotations + histology_annotations])) / 10, 1.0)
        
        quantity_score = min(total_annotations / 20, 1.0)
        
        return (diversity_score + quantity_score) / 2