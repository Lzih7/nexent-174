"""
医疗Q&A数据集生成模块
基于标注后的医疗数据生成高质量的问答对
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import json
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalQAGenerator:
    """医疗Q&A数据集生成器"""
    
    def __init__(self):
        # 问题模板
        self.question_templates = {
            'definition': [
                "什么是{term}？",
                "{term}的定义是什么？",
                "请解释{term}的含义。",
                "{term}是指什么？"
            ],
            'symptoms': [
                "{disease}有哪些症状？",
                "{disease}的临床表现是什么？",
                "{disease}患者会出现什么症状？",
                "如何识别{disease}的症状？"
            ],
            'diagnosis': [
                "如何诊断{disease}？",
                "{disease}的诊断方法有哪些？",
                "{disease}需要做哪些检查？",
                "怎样确诊{disease}？"
            ],
            'treatment': [
                "{disease}如何治疗？",
                "{disease}的治疗方案是什么？",
                "{disease}有哪些治疗方法？",
                "如何治疗{disease}患者？"
            ],
            'pathology': [
                "{disease}的病理特征是什么？",
                "{disease}在显微镜下有什么表现？",
                "{disease}的组织学特点有哪些？",
                "{disease}的病理改变包括什么？"
            ],
            'differential': [
                "{disease}需要与哪些疾病鉴别？",
                "如何鉴别{disease}和{other_disease}？",
                "{disease}的鉴别诊断要点是什么？",
                "{disease}容易与什么疾病混淆？"
            ]
        }
        
        # 难度等级定义
        self.difficulty_levels = {
            'easy': {
                'description': '基础概念和常见疾病',
                'keywords': ['定义', '是什么', '基本', '常见'],
                'score_range': (1, 3)
            },
            'medium': {
                'description': '临床诊断和治疗',
                'keywords': ['诊断', '治疗', '鉴别', '临床'],
                'score_range': (4, 6)
            },
            'hard': {
                'description': '病理机制和复杂病例',
                'keywords': ['机制', '病理', '复杂', '罕见'],
                'score_range': (7, 10)
            }
        }
    
    def generate_qa_dataset(self, annotated_content: Dict[str, Any], 
                          qa_count: int = 10) -> Dict[str, Any]:
        """生成Q&A数据集
        
        Args:
            annotated_content: 标注后的医疗内容
            qa_count: 生成的Q&A对数量
            
        Returns:
            Q&A数据集
        """
        try:
            if not annotated_content.get('success', False):
                raise ValueError("输入的标注内容无效")
            
            annotations = annotated_content.get('annotations', {})
            content = annotated_content.get('structured_annotation', {}).get('content_summary', '')
            
            # 1. 提取关键信息
            key_info = self._extract_key_information(annotations)
            
            # 2. 生成不同类型的Q&A对
            qa_pairs = []
            
            # 定义问题类型分配
            question_types = ['definition', 'symptoms', 'diagnosis', 'treatment', 'pathology', 'differential']
            type_counts = self._distribute_question_types(qa_count, len(question_types))
            
            for question_type, count in zip(question_types, type_counts):
                pairs = self._generate_qa_by_type(question_type, key_info, content, count)
                qa_pairs.extend(pairs)
            
            # 3. 随机打乱并限制数量
            random.shuffle(qa_pairs)
            qa_pairs = qa_pairs[:qa_count]
            
            # 4. 质量评估和优化
            optimized_pairs = self._optimize_qa_pairs(qa_pairs)
            
            # 5. 生成数据集统计
            dataset_stats = self._generate_dataset_stats(optimized_pairs)
            
            result = {
                'success': True,
                'dataset_info': {
                    'total_qa_pairs': len(optimized_pairs),
                    'generation_time': datetime.now().isoformat(),
                    'source_content_length': len(content),
                    'annotation_count': sum(len(annotations.get(key, [])) for key in annotations)
                },
                'qa_pairs': optimized_pairs,
                'statistics': dataset_stats,
                'quality_metrics': self._calculate_quality_metrics(optimized_pairs)
            }
            
            logger.info(f"成功生成 {len(optimized_pairs)} 个Q&A对")
            return result
            
        except Exception as e:
            logger.error(f"Q&A数据集生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'qa_pairs': [],
                'statistics': {},
                'quality_metrics': {}
            }
    
    def _extract_key_information(self, annotations: Dict[str, Any]) -> Dict[str, List[str]]:
        """从标注中提取关键信息"""
        key_info = {
            'diseases': [],
            'symptoms': [],
            'treatments': [],
            'anatomy': [],
            'pathology_terms': [],
            'diagnosis_terms': []
        }
        
        # 从病理标注中提取
        pathology_annotations = annotations.get('pathology', [])
        for ann in pathology_annotations:
            if ann.get('category') == '疾病分类':
                key_info['diseases'].append(ann.get('term', ''))
            elif ann.get('category') == '组织学特征':
                key_info['pathology_terms'].append(ann.get('term', ''))
        
        # 从诊断标注中提取
        diagnosis_annotations = annotations.get('diagnosis', [])
        for ann in diagnosis_annotations:
            category = ann.get('category', '')
            term = ann.get('term', '')
            if '临床表现' in category:
                key_info['symptoms'].append(term)
            elif '治疗' in category:
                key_info['treatments'].append(term)
            else:
                key_info['diagnosis_terms'].append(term)
        
        # 从实体中提取
        entities = annotations.get('entities', [])
        for entity in entities:
            entity_type = entity.get('type', '')
            text = entity.get('text', '')
            if entity_type == 'disease':
                key_info['diseases'].append(text)
            elif entity_type == 'anatomy':
                key_info['anatomy'].append(text)
        
        # 去重
        for key in key_info:
            key_info[key] = list(set(filter(None, key_info[key])))
        
        return key_info
    
    def _distribute_question_types(self, total_count: int, type_count: int) -> List[int]:
        """分配问题类型数量"""
        base_count = total_count // type_count
        remainder = total_count % type_count
        
        counts = [base_count] * type_count
        for i in range(remainder):
            counts[i] += 1
        
        return counts
    
    def _generate_qa_by_type(self, question_type: str, key_info: Dict[str, List[str]], 
                           content: str, count: int) -> List[Dict[str, Any]]:
        """根据类型生成Q&A对"""
        qa_pairs = []
        templates = self.question_templates.get(question_type, [])
        
        if not templates:
            return qa_pairs
        
        for _ in range(count):
            try:
                # 选择合适的实体
                entity = self._select_entity_for_type(question_type, key_info)
                if not entity:
                    continue
                
                # 选择问题模板
                template = random.choice(templates)
                
                # 生成问题
                question = self._generate_question(template, entity, key_info)
                
                # 生成答案
                answer = self._generate_answer(question_type, entity, content, key_info)
                
                # 评估难度
                difficulty = self._assess_difficulty(question, answer)
                
                # 生成关键词
                keywords = self._extract_keywords(question, answer)
                
                qa_pair = {
                    'id': f"qa_{len(qa_pairs) + 1}",
                    'question': question,
                    'answer': answer,
                    'question_type': question_type,
                    'difficulty': difficulty,
                    'keywords': keywords,
                    'entity': entity,
                    'quality_score': self._calculate_qa_quality(question, answer)
                }
                
                qa_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"生成{question_type}类型Q&A失败: {e}")
                continue
        
        return qa_pairs
    
    def _select_entity_for_type(self, question_type: str, key_info: Dict[str, List[str]]) -> Optional[str]:
        """为问题类型选择合适的实体"""
        if question_type in ['definition', 'symptoms', 'diagnosis', 'treatment', 'differential']:
            entities = key_info.get('diseases', [])
        elif question_type == 'pathology':
            entities = key_info.get('pathology_terms', []) + key_info.get('diseases', [])
        else:
            entities = key_info.get('diseases', [])
        
        return random.choice(entities) if entities else None
    
    def _generate_question(self, template: str, entity: str, key_info: Dict[str, List[str]]) -> str:
        """生成问题"""
        question = template.format(term=entity, disease=entity)
        
        # 如果需要其他疾病进行对比
        if '{other_disease}' in template:
            other_diseases = [d for d in key_info.get('diseases', []) if d != entity]
            other_disease = random.choice(other_diseases) if other_diseases else '其他疾病'
            question = question.format(other_disease=other_disease)
        
        return question
    
    def _generate_answer(self, question_type: str, entity: str, content: str, 
                        key_info: Dict[str, List[str]]) -> str:
        """生成答案"""
        # 从内容中查找相关信息
        relevant_sentences = self._find_relevant_sentences(entity, content)
        
        if not relevant_sentences:
            # 如果找不到相关内容，生成基础答案
            return self._generate_basic_answer(question_type, entity, key_info)
        
        # 基于相关句子生成答案
        answer_parts = []
        
        if question_type == 'definition':
            answer_parts.append(f"{entity}是指")
            answer_parts.extend(relevant_sentences[:2])
        elif question_type == 'symptoms':
            answer_parts.append(f"{entity}的主要症状包括：")
            answer_parts.extend(relevant_sentences[:3])
        elif question_type == 'diagnosis':
            answer_parts.append(f"{entity}的诊断主要依据：")
            answer_parts.extend(relevant_sentences[:3])
        elif question_type == 'treatment':
            answer_parts.append(f"{entity}的治疗方法包括：")
            answer_parts.extend(relevant_sentences[:3])
        elif question_type == 'pathology':
            answer_parts.append(f"{entity}的病理特征表现为：")
            answer_parts.extend(relevant_sentences[:2])
        else:
            answer_parts.extend(relevant_sentences[:3])
        
        return ' '.join(answer_parts)
    
    def _find_relevant_sentences(self, entity: str, content: str) -> List[str]:
        """查找相关句子"""
        sentences = re.split(r'[。！？]', content)
        relevant = []
        
        for sentence in sentences:
            if entity in sentence and len(sentence.strip()) > 10:
                relevant.append(sentence.strip())
        
        return relevant
    
    def _generate_basic_answer(self, question_type: str, entity: str, 
                             key_info: Dict[str, List[str]]) -> str:
        """生成基础答案"""
        basic_answers = {
            'definition': f"{entity}是一种医学概念，需要进一步的专业解释。",
            'symptoms': f"{entity}的症状需要根据具体情况进行临床评估。",
            'diagnosis': f"{entity}的诊断需要结合临床表现和相关检查。",
            'treatment': f"{entity}的治疗应该根据患者具体情况制定个体化方案。",
            'pathology': f"{entity}的病理特征需要通过组织学检查确定。",
            'differential': f"{entity}需要与相关疾病进行鉴别诊断。"
        }
        
        return basic_answers.get(question_type, f"关于{entity}的详细信息需要咨询专业医生。")
    
    def _assess_difficulty(self, question: str, answer: str) -> str:
        """评估问题难度"""
        # 基于问题和答案的复杂度评估
        complexity_indicators = {
            'easy': ['什么是', '定义', '基本', '常见'],
            'medium': ['如何', '诊断', '治疗', '症状'],
            'hard': ['机制', '病理', '鉴别', '复杂']
        }
        
        text = question + ' ' + answer
        scores = {}
        
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            scores[level] = score
        
        return max(scores, key=scores.get) if scores else 'medium'
    
    def _extract_keywords(self, question: str, answer: str) -> List[str]:
        """提取关键词"""
        text = question + ' ' + answer
        
        # 医学关键词模式
        medical_patterns = [
            r'[^，。；：\s]*(?:病|症|癌|瘤|炎)[^，。；：\s]*',
            r'[^，。；：\s]*(?:诊断|治疗|检查|手术)[^，。；：\s]*',
            r'[^，。；：\s]*(?:细胞|组织|器官|系统)[^，。；：\s]*'
        ]
        
        keywords = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text)
            keywords.extend([m for m in matches if len(m) >= 2])
        
        return list(set(keywords))[:10]  # 限制关键词数量
    
    def _calculate_qa_quality(self, question: str, answer: str) -> float:
        """计算Q&A质量分数"""
        score = 0.0
        
        # 问题质量 (0-50分)
        if len(question) >= 5:
            score += 20
        if '？' in question or '?' in question:
            score += 10
        if any(word in question for word in ['什么', '如何', '哪些', '怎样']):
            score += 20
        
        # 答案质量 (0-50分)
        if len(answer) >= 20:
            score += 20
        if len(answer.split()) >= 5:
            score += 15
        if any(word in answer for word in ['包括', '表现为', '主要', '需要']):
            score += 15
        
        return min(score / 100, 1.0)
    
    def _optimize_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化Q&A对"""
        # 按质量分数排序
        qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # 去重相似问题
        unique_pairs = []
        seen_questions = set()
        
        for pair in qa_pairs:
            question_key = self._normalize_question(pair['question'])
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_pairs.append(pair)
        
        return unique_pairs
    
    def _normalize_question(self, question: str) -> str:
        """标准化问题用于去重"""
        # 移除标点符号和空格，转换为小写
        normalized = re.sub(r'[^\w]', '', question.lower())
        return normalized
    
    def _generate_dataset_stats(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据集统计"""
        if not qa_pairs:
            return {}
        
        # 难度分布
        difficulty_dist = {}
        for pair in qa_pairs:
            diff = pair.get('difficulty', 'medium')
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        # 问题类型分布
        type_dist = {}
        for pair in qa_pairs:
            q_type = pair.get('question_type', 'general')
            type_dist[q_type] = type_dist.get(q_type, 0) + 1
        
        # 质量分布
        quality_scores = [pair.get('quality_score', 0) for pair in qa_pairs]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'difficulty_distribution': difficulty_dist,
            'question_type_distribution': type_dist,
            'average_quality_score': round(avg_quality, 3),
            'total_keywords': len(set(kw for pair in qa_pairs for kw in pair.get('keywords', []))),
            'average_question_length': sum(len(pair.get('question', '')) for pair in qa_pairs) / len(qa_pairs),
            'average_answer_length': sum(len(pair.get('answer', '')) for pair in qa_pairs) / len(qa_pairs)
        }
    
    def _calculate_quality_metrics(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算质量指标"""
        if not qa_pairs:
            return {}
        
        # 完整性指标
        complete_pairs = sum(1 for pair in qa_pairs if pair.get('question') and pair.get('answer'))
        completeness = complete_pairs / len(qa_pairs)
        
        # 多样性指标
        unique_questions = len(set(pair.get('question', '') for pair in qa_pairs))
        diversity = unique_questions / len(qa_pairs)
        
        # 专业性指标
        medical_terms_count = sum(len(pair.get('keywords', [])) for pair in qa_pairs)
        professionalism = min(medical_terms_count / (len(qa_pairs) * 5), 1.0)
        
        return {
            'completeness': round(completeness, 3),
            'diversity': round(diversity, 3),
            'professionalism': round(professionalism, 3),
            'overall_quality': round((completeness + diversity + professionalism) / 3, 3)
        }