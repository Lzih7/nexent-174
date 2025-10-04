"""
医疗知识标注器
对医疗文本进行专业标注和语义增强
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AnnotationType(Enum):
    """标注类型枚举"""
    MEDICAL_TERM = "medical_term"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    TREATMENT = "treatment"
    ANATOMY = "anatomy"
    PATHOLOGY = "pathology"
    DIAGNOSTIC_METHOD = "diagnostic_method"
    MEDICATION = "medication"

@dataclass
class MedicalAnnotation:
    """医疗标注数据结构"""
    text: str
    start_pos: int
    end_pos: int
    annotation_type: AnnotationType
    confidence: float
    metadata: Dict[str, Any]

class MedicalKnowledgeAnnotator:
    """医疗知识标注器"""
    
    def __init__(self):
        self.medical_dictionaries = self._load_medical_dictionaries()
        self.annotation_rules = self._load_annotation_rules()
        self.semantic_patterns = self._load_semantic_patterns()
    
    def _load_medical_dictionaries(self) -> Dict[str, Set[str]]:
        """加载医学词典"""
        return {
            "diseases": {
                "恶性肿瘤", "良性肿瘤", "腺癌", "鳞癌", "肉瘤", "淋巴瘤", "白血病",
                "炎症", "感染", "坏死", "增生", "化生", "不典型增生", "原位癌",
                "浸润性癌", "转移性肿瘤", "间质瘤", "神经内分泌肿瘤"
            },
            "anatomy": {
                "上皮", "间质", "血管", "神经", "淋巴管", "腺体", "导管",
                "基底膜", "细胞核", "细胞质", "胞膜", "线粒体", "内质网"
            },
            "pathology": {
                "异型性", "多形性", "核分裂象", "核质比", "细胞排列", "组织结构",
                "纤维化", "钙化", "出血", "水肿", "坏死", "凋亡", "增殖"
            },
            "diagnostic_methods": {
                "HE染色", "免疫组化", "特殊染色", "分子检测", "基因检测",
                "原位杂交", "电镜检查", "冰冻切片", "石蜡切片"
            },
            "treatments": {
                "手术切除", "化疗", "放疗", "靶向治疗", "免疫治疗",
                "内分泌治疗", "支持治疗", "姑息治疗"
            }
        }
    
    def _load_annotation_rules(self) -> Dict[str, List[str]]:
        """加载标注规则"""
        return {
            "disease_patterns": [
                r'[A-Za-z]*癌',
                r'[A-Za-z]*瘤',
                r'[A-Za-z]*病',
                r'[A-Za-z]*症',
                r'恶性[肿瘤|肿块]*',
                r'良性[肿瘤|肿块]*'
            ],
            "pathology_patterns": [
                r'细胞[异型性|多形性|排列]*',
                r'核[分裂象|质比|形态]*',
                r'组织[结构|形态|排列]*',
                r'[纤维化|钙化|坏死|出血|水肿]'
            ],
            "anatomy_patterns": [
                r'[上皮|间质|血管|神经|淋巴]*[细胞|组织|结构]*',
                r'[腺体|导管|基底膜]*[结构|形态]*'
            ]
        }
    
    def _load_semantic_patterns(self) -> Dict[str, str]:
        """加载语义模式"""
        return {
            "definition": r'(.+?)是指(.+?)。',
            "characteristic": r'(.+?)的特征[是为](.+?)。',
            "diagnosis": r'诊断(.+?)需要(.+?)。',
            "treatment": r'(.+?)的治疗[方法|方案][包括|为](.+?)。',
            "prognosis": r'(.+?)的预后(.+?)。'
        }
    
    async def annotate_medical_text(self, text: str, 
                                  annotation_types: Optional[List[AnnotationType]] = None) -> Dict[str, Any]:
        """
        标注医疗文本
        
        Args:
            text: 待标注的医疗文本
            annotation_types: 指定的标注类型，None表示全部类型
            
        Returns:
            标注结果
        """
        try:
            logger.info(f"开始标注医疗文本，长度: {len(text)}")
            
            if annotation_types is None:
                annotation_types = list(AnnotationType)
            
            annotations = []
            
            # 1. 基于词典的标注
            dict_annotations = await self._annotate_by_dictionary(text, annotation_types)
            annotations.extend(dict_annotations)
            
            # 2. 基于规则的标注
            rule_annotations = await self._annotate_by_rules(text, annotation_types)
            annotations.extend(rule_annotations)
            
            # 3. 语义关系提取
            semantic_relations = await self._extract_semantic_relations(text)
            
            # 4. 去重和合并重叠标注
            merged_annotations = self._merge_overlapping_annotations(annotations)
            
            # 5. 生成标注统计
            statistics = self._generate_annotation_statistics(merged_annotations)
            
            result = {
                "success": True,
                "original_text": text,
                "annotations": [
                    {
                        "text": ann.text,
                        "start_pos": ann.start_pos,
                        "end_pos": ann.end_pos,
                        "type": ann.annotation_type.value,
                        "confidence": ann.confidence,
                        "metadata": ann.metadata
                    }
                    for ann in merged_annotations
                ],
                "semantic_relations": semantic_relations,
                "statistics": statistics
            }
            
            logger.info(f"医疗文本标注完成，生成 {len(merged_annotations)} 个标注")
            return result
            
        except Exception as e:
            logger.error(f"医疗文本标注失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "annotations": [],
                "semantic_relations": []
            }
    
    async def _annotate_by_dictionary(self, text: str, 
                                    annotation_types: List[AnnotationType]) -> List[MedicalAnnotation]:
        """基于词典的标注"""
        annotations = []
        
        for dict_name, terms in self.medical_dictionaries.items():
            # 确定标注类型
            if dict_name == "diseases" and AnnotationType.DISEASE in annotation_types:
                ann_type = AnnotationType.DISEASE
            elif dict_name == "anatomy" and AnnotationType.ANATOMY in annotation_types:
                ann_type = AnnotationType.ANATOMY
            elif dict_name == "pathology" and AnnotationType.PATHOLOGY in annotation_types:
                ann_type = AnnotationType.PATHOLOGY
            elif dict_name == "diagnostic_methods" and AnnotationType.DIAGNOSTIC_METHOD in annotation_types:
                ann_type = AnnotationType.DIAGNOSTIC_METHOD
            elif dict_name == "treatments" and AnnotationType.TREATMENT in annotation_types:
                ann_type = AnnotationType.TREATMENT
            else:
                continue
            
            # 查找词典中的术语
            for term in terms:
                for match in re.finditer(re.escape(term), text):
                    annotation = MedicalAnnotation(
                        text=term,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        annotation_type=ann_type,
                        confidence=0.9,  # 词典匹配置信度较高
                        metadata={
                            "source": "dictionary",
                            "dictionary": dict_name
                        }
                    )
                    annotations.append(annotation)
        
        return annotations
    
    async def _annotate_by_rules(self, text: str, 
                               annotation_types: List[AnnotationType]) -> List[MedicalAnnotation]:
        """基于规则的标注"""
        annotations = []
        
        for rule_name, patterns in self.annotation_rules.items():
            # 确定标注类型
            if rule_name == "disease_patterns" and AnnotationType.DISEASE in annotation_types:
                ann_type = AnnotationType.DISEASE
            elif rule_name == "pathology_patterns" and AnnotationType.PATHOLOGY in annotation_types:
                ann_type = AnnotationType.PATHOLOGY
            elif rule_name == "anatomy_patterns" and AnnotationType.ANATOMY in annotation_types:
                ann_type = AnnotationType.ANATOMY
            else:
                continue
            
            # 应用规则模式
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    annotation = MedicalAnnotation(
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        annotation_type=ann_type,
                        confidence=0.7,  # 规则匹配置信度中等
                        metadata={
                            "source": "rule",
                            "pattern": pattern
                        }
                    )
                    annotations.append(annotation)
        
        return annotations
    
    async def _extract_semantic_relations(self, text: str) -> List[Dict[str, Any]]:
        """提取语义关系"""
        relations = []
        
        for relation_type, pattern in self.semantic_patterns.items():
            for match in re.finditer(pattern, text):
                if len(match.groups()) >= 2:
                    relation = {
                        "type": relation_type,
                        "subject": match.group(1).strip(),
                        "object": match.group(2).strip(),
                        "context": match.group().strip(),
                        "start_pos": match.start(),
                        "end_pos": match.end()
                    }
                    relations.append(relation)
        
        return relations
    
    def _merge_overlapping_annotations(self, annotations: List[MedicalAnnotation]) -> List[MedicalAnnotation]:
        """合并重叠的标注"""
        if not annotations:
            return []
        
        # 按位置排序
        sorted_annotations = sorted(annotations, key=lambda x: (x.start_pos, x.end_pos))
        merged = []
        
        for current in sorted_annotations:
            if not merged:
                merged.append(current)
                continue
            
            last = merged[-1]
            
            # 检查是否重叠
            if current.start_pos <= last.end_pos:
                # 重叠，选择置信度更高的或者更长的
                if (current.confidence > last.confidence or 
                    (current.confidence == last.confidence and 
                     (current.end_pos - current.start_pos) > (last.end_pos - last.start_pos))):
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged
    
    def _generate_annotation_statistics(self, annotations: List[MedicalAnnotation]) -> Dict[str, Any]:
        """生成标注统计信息"""
        type_counts = {}
        confidence_scores = []
        
        for ann in annotations:
            ann_type = ann.annotation_type.value
            type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
            confidence_scores.append(ann.confidence)
        
        return {
            "total_annotations": len(annotations),
            "type_distribution": type_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0
        }
    
    async def batch_annotate_documents(self, documents: List[str]) -> Dict[str, Any]:
        """批量标注文档"""
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"标注文档 {i+1}/{len(documents)}")
            result = await self.annotate_medical_text(doc)
            results.append(result)
        
        # 汇总统计
        total_annotations = sum(len(r.get("annotations", [])) for r in results)
        successful_docs = sum(1 for r in results if r.get("success", False))
        
        return {
            "success": True,
            "total_documents": len(documents),
            "successful_documents": successful_docs,
            "total_annotations": total_annotations,
            "results": results
        }