"""
医疗数据工程演示处理器
展示完整的医疗数据处理流程：数据清洗 → 专业标注 → Q&A生成 → 知识库集成
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os

from .data_cleaner import MedicalDataCleaner
from .professional_annotator import MedicalProfessionalAnnotator
from .qa_generator import MedicalQAGenerator
from .knowledge_base_integrator import MedicalKnowledgeBaseIntegrator

logger = logging.getLogger(__name__)

class MedicalDataEngineeringDemo:
    """医疗数据工程演示处理器"""
    
    def __init__(self):
        """初始化演示处理器"""
        self.cleaner = MedicalDataCleaner()
        self.annotator = MedicalProfessionalAnnotator()
        self.generator = MedicalQAGenerator()
        self.integrator = MedicalKnowledgeBaseIntegrator()
        
        # 演示用的病理教科书样本数据
        self.sample_pathology_text = """
        肺癌是最常见的恶性肿瘤之一，根据组织学特征可分为小细胞肺癌和非小细胞肺癌。
        
        非小细胞肺癌包括腺癌、鳞状细胞癌和大细胞癌。腺癌是最常见的肺癌类型，
        常见于女性和非吸烟者。镜下可见腺体结构，细胞异型性明显，核分裂象增多。
        
        临床表现包括咳嗽、咳痰、胸痛、呼吸困难等症状。早期肺癌症状不明显，
        多数患者确诊时已为中晚期。
        
        诊断方法包括胸部CT、支气管镜检查、痰细胞学检查和组织病理学检查。
        组织病理学检查是确诊的金标准。
        
        治疗方案根据病理类型、分期和患者一般状况制定，包括手术治疗、
        放射治疗、化学治疗和靶向治疗等。早期肺癌首选手术治疗，
        晚期肺癌以化疗和靶向治疗为主。
        
        预后与病理类型、分期、治疗方案等因素相关。早期发现和规范治疗
        可显著改善患者预后。
        """
    
    def run_complete_demo(self, input_text: str = None, 
                         qa_count: int = 8,
                         create_knowledge_base: bool = True) -> Dict[str, Any]:
        """运行完整的医疗数据工程演示
        
        Args:
            input_text: 输入的医疗文本，默认使用样本数据
            qa_count: 生成的Q&A对数量
            create_knowledge_base: 是否创建知识库并存储数据
            
        Returns:
            完整的演示结果
        """
        demo_result = {
            "demo_info": {
                "title": "医疗数据工程与模型训练演示",
                "description": "展示病理教科书数据处理的完整流程",
                "start_time": datetime.now().isoformat(),
                "input_text_length": 0,
                "qa_target_count": qa_count,
                "knowledge_base_integration": create_knowledge_base
            },
            "processing_steps": [],
            "final_results": {},
            "performance_metrics": {}
        }
        
        try:
            # 使用输入文本或样本数据
            text_to_process = input_text if input_text else self.sample_pathology_text
            demo_result["demo_info"]["input_text_length"] = len(text_to_process)
            
            # 步骤1: 数据清洗
            print("🔄 步骤1: 医疗数据清洗...")
            cleaning_start = datetime.now()
            
            cleaning_result = self.cleaner.clean_medical_text(text_to_process)
            
            cleaning_step = {
                "step_number": 1,
                "step_name": "医疗数据清洗",
                "status": "completed" if cleaning_result.get("success") else "failed",
                "processing_time": (datetime.now() - cleaning_start).total_seconds(),
                "input_length": len(text_to_process),
                "output_length": len(cleaning_result.get("cleaned_text", "")),
                "quality_score": cleaning_result.get("quality_score", 0),
                "medical_terms_extracted": len(cleaning_result.get("medical_terms", {})),
                "segments_created": len(cleaning_result.get("segments", []))
            }
            demo_result["processing_steps"].append(cleaning_step)
            
            if not cleaning_result.get("success"):
                raise Exception("数据清洗步骤失败")
            
            # 步骤2: 专业标注
            print("🔄 步骤2: 医疗专业标注...")
            annotation_start = datetime.now()
            
            cleaned_text = cleaning_result.get("cleaned_text", "")
            annotation_result = self.annotator.annotate_medical_content(cleaned_text, "pathology")
            
            annotation_step = {
                "step_number": 2,
                "step_name": "医疗专业标注",
                "status": "completed" if annotation_result.get("success") else "failed",
                "processing_time": (datetime.now() - annotation_start).total_seconds(),
                "pathology_annotations": len(annotation_result.get("annotations", {}).get("pathology", {})),
                "diagnosis_annotations": len(annotation_result.get("annotations", {}).get("diagnosis", {})),
                "key_entities": len(annotation_result.get("annotations", {}).get("entities", [])),
                "relationships": len(annotation_result.get("annotations", {}).get("relationships", []))
            }
            demo_result["processing_steps"].append(annotation_step)
            
            if not annotation_result.get("success"):
                raise Exception("专业标注步骤失败")
            
            # 步骤3: Q&A生成
            print("🔄 步骤3: 医疗Q&A数据集生成...")
            qa_start = datetime.now()
            
            qa_result = self.generator.generate_qa_dataset(annotation_result, qa_count)
            
            qa_step = {
                "step_number": 3,
                "step_name": "医疗Q&A数据集生成",
                "status": "completed" if qa_result.get("success") else "failed",
                "processing_time": (datetime.now() - qa_start).total_seconds(),
                "target_qa_count": qa_count,
                "actual_qa_count": len(qa_result.get("qa_pairs", [])),
                "average_quality": qa_result.get("quality_metrics", {}).get("overall_quality", 0),
                "difficulty_distribution": qa_result.get("quality_metrics", {}).get("difficulty_distribution", {})
            }
            demo_result["processing_steps"].append(qa_step)
            
            if not qa_result.get("success"):
                raise Exception("Q&A生成步骤失败")
            
            # 步骤4: 知识库集成（可选）
            knowledge_base_result = None
            if create_knowledge_base:
                print("🔄 步骤4: 知识库集成...")
                kb_start = datetime.now()
                
                # 创建医疗知识库索引
                index_creation = self.integrator.create_medical_knowledge_index()
                
                if index_creation.get("success"):
                    index_name = index_creation.get("index_name")
                    
                    # 集成Q&A数据到知识库
                    qa_pairs = qa_result.get("qa_pairs", [])
                    integration_result = self.integrator.integrate_qa_dataset(qa_pairs, index_name)
                    
                    knowledge_base_result = {
                        "index_creation": index_creation,
                        "data_integration": integration_result
                    }
                    
                    kb_step = {
                        "step_number": 4,
                        "step_name": "知识库集成",
                        "status": "completed" if integration_result.get("success") else "failed",
                        "processing_time": (datetime.now() - kb_start).total_seconds(),
                        "index_name": index_name,
                        "documents_indexed": integration_result.get("total_indexed", 0),
                        "index_stats": integration_result.get("index_stats", {})
                    }
                    demo_result["processing_steps"].append(kb_step)
                else:
                    kb_step = {
                        "step_number": 4,
                        "step_name": "知识库集成",
                        "status": "failed",
                        "processing_time": (datetime.now() - kb_start).total_seconds(),
                        "error": index_creation.get("error", "索引创建失败")
                    }
                    demo_result["processing_steps"].append(kb_step)
            
            # 汇总最终结果
            demo_result["final_results"] = {
                "data_cleaning": {
                    "original_length": len(text_to_process),
                    "cleaned_length": len(cleaned_text),
                    "quality_score": cleaning_result.get("quality_score", 0),
                    "medical_terms": cleaning_result.get("medical_terms", {}),
                    "segments_count": len(cleaning_result.get("segments", []))
                },
                "professional_annotation": {
                    "annotation_categories": list(annotation_result.get("annotations", {}).keys()),
                    "total_entities": len(annotation_result.get("annotations", {}).get("entities", [])),
                    "total_relationships": len(annotation_result.get("annotations", {}).get("relationships", []))
                },
                "qa_generation": {
                    "total_qa_pairs": len(qa_result.get("qa_pairs", [])),
                    "quality_metrics": qa_result.get("quality_metrics", {}),
                    "sample_qa_pairs": qa_result.get("qa_pairs", [])[:3]  # 显示前3个样本
                },
                "knowledge_base": knowledge_base_result if knowledge_base_result else {"status": "skipped"}
            }
            
            # 性能指标
            total_time = (datetime.now() - datetime.fromisoformat(demo_result["demo_info"]["start_time"])).total_seconds()
            demo_result["performance_metrics"] = {
                "total_processing_time": total_time,
                "steps_completed": len([s for s in demo_result["processing_steps"] if s["status"] == "completed"]),
                "steps_failed": len([s for s in demo_result["processing_steps"] if s["status"] == "failed"]),
                "success_rate": len([s for s in demo_result["processing_steps"] if s["status"] == "completed"]) / len(demo_result["processing_steps"]),
                "data_quality_improvement": cleaning_result.get("quality_score", 0),
                "knowledge_extraction_efficiency": len(qa_result.get("qa_pairs", [])) / len(text_to_process) * 1000  # 每千字符生成的Q&A对数
            }
            
            demo_result["demo_info"]["end_time"] = datetime.now().isoformat()
            demo_result["demo_info"]["status"] = "completed"
            
            print("✅ 医疗数据工程演示完成！")
            return demo_result
            
        except Exception as e:
            logger.error(f"医疗数据工程演示失败: {e}")
            demo_result["demo_info"]["end_time"] = datetime.now().isoformat()
            demo_result["demo_info"]["status"] = "failed"
            demo_result["demo_info"]["error"] = str(e)
            
            return demo_result
    
    def generate_demo_report(self, demo_result: Dict[str, Any]) -> str:
        """生成演示报告
        
        Args:
            demo_result: 演示结果
            
        Returns:
            格式化的演示报告
        """
        report_lines = []
        
        # 标题
        report_lines.append("=" * 60)
        report_lines.append("🏥 医疗数据工程与模型训练演示报告")
        report_lines.append("=" * 60)
        
        # 基本信息
        demo_info = demo_result.get("demo_info", {})
        report_lines.append(f"📊 演示状态: {demo_info.get('status', 'unknown')}")
        report_lines.append(f"📝 输入文本长度: {demo_info.get('input_text_length', 0)} 字符")
        report_lines.append(f"🎯 目标Q&A数量: {demo_info.get('qa_target_count', 0)}")
        report_lines.append(f"🗄️ 知识库集成: {'是' if demo_info.get('knowledge_base_integration') else '否'}")
        report_lines.append("")
        
        # 处理步骤
        report_lines.append("🔄 处理步骤详情:")
        report_lines.append("-" * 40)
        
        for step in demo_result.get("processing_steps", []):
            status_icon = "✅" if step["status"] == "completed" else "❌"
            report_lines.append(f"{status_icon} 步骤{step['step_number']}: {step['step_name']}")
            report_lines.append(f"   ⏱️ 处理时间: {step.get('processing_time', 0):.2f}秒")
            
            if step["step_name"] == "医疗数据清洗":
                report_lines.append(f"   📏 输入长度: {step.get('input_length', 0)} → 输出长度: {step.get('output_length', 0)}")
                report_lines.append(f"   🎯 质量评分: {step.get('quality_score', 0):.2f}")
                report_lines.append(f"   🏷️ 医学术语: {step.get('medical_terms_extracted', 0)} 类")
                report_lines.append(f"   📑 文本段落: {step.get('segments_created', 0)} 个")
                
            elif step["step_name"] == "医疗专业标注":
                report_lines.append(f"   🏷️ 病理标注: {step.get('pathology_annotations', 0)} 个")
                report_lines.append(f"   🩺 诊断标注: {step.get('diagnosis_annotations', 0)} 个")
                report_lines.append(f"   🔗 关键实体: {step.get('key_entities', 0)} 个")
                report_lines.append(f"   🔄 关系抽取: {step.get('relationships', 0)} 个")
                
            elif step["step_name"] == "医疗Q&A数据集生成":
                report_lines.append(f"   🎯 目标数量: {step.get('target_qa_count', 0)}")
                report_lines.append(f"   ✅ 实际生成: {step.get('actual_qa_count', 0)}")
                report_lines.append(f"   ⭐ 平均质量: {step.get('average_quality', 0):.2f}")
                
            elif step["step_name"] == "知识库集成":
                if step["status"] == "completed":
                    report_lines.append(f"   📚 索引名称: {step.get('index_name', 'unknown')}")
                    report_lines.append(f"   📄 文档索引: {step.get('documents_indexed', 0)} 个")
                else:
                    report_lines.append(f"   ❌ 错误信息: {step.get('error', 'unknown')}")
            
            report_lines.append("")
        
        # 最终结果
        final_results = demo_result.get("final_results", {})
        report_lines.append("📊 最终结果统计:")
        report_lines.append("-" * 40)
        
        # 数据清洗结果
        cleaning = final_results.get("data_cleaning", {})
        report_lines.append(f"🧹 数据清洗:")
        report_lines.append(f"   📏 文本压缩率: {(1 - cleaning.get('cleaned_length', 0) / max(cleaning.get('original_length', 1), 1)) * 100:.1f}%")
        report_lines.append(f"   🎯 数据质量: {cleaning.get('quality_score', 0):.2f}/1.0")
        report_lines.append(f"   🏷️ 医学术语类别: {len(cleaning.get('medical_terms', {}))}")
        
        # Q&A生成结果
        qa_gen = final_results.get("qa_generation", {})
        report_lines.append(f"❓ Q&A生成:")
        report_lines.append(f"   📝 问答对数量: {qa_gen.get('total_qa_pairs', 0)}")
        quality_metrics = qa_gen.get("quality_metrics", {})
        report_lines.append(f"   ⭐ 整体质量: {quality_metrics.get('overall_quality', 0):.2f}")
        
        # 性能指标
        performance = demo_result.get("performance_metrics", {})
        report_lines.append(f"⚡ 性能指标:")
        report_lines.append(f"   ⏱️ 总处理时间: {performance.get('total_processing_time', 0):.2f}秒")
        report_lines.append(f"   ✅ 成功率: {performance.get('success_rate', 0) * 100:.1f}%")
        report_lines.append(f"   📈 知识提取效率: {performance.get('knowledge_extraction_efficiency', 0):.2f} Q&A对/千字符")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("🎉 演示报告生成完成")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)