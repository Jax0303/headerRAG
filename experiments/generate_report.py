"""
실험 결과 종합 분석 보고서 생성 스크립트
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """실험 결과 종합 보고서 생성 클래스"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_latest_results(self) -> Dict[str, Any]:
        """가장 최근 실험 결과 로드"""
        results = {
            'experiment_1': None,
            'experiment_2': None,
            'experiment_3': None
        }
        
        # 실험 1 결과 찾기
        exp1_files = sorted(
            glob.glob(str(self.results_dir / "experiment_1_parsing_*.json")),
            reverse=True
        )
        if exp1_files:
            try:
                with open(exp1_files[0], 'r', encoding='utf-8') as f:
                    results['experiment_1'] = json.load(f)
            except Exception as e:
                print(f"실험 1 결과 로드 실패: {e}")
        
        # 실험 2 결과 찾기
        exp2_files = sorted(
            glob.glob(str(self.results_dir / "experiment_2_rag_*.json")),
            reverse=True
        )
        if exp2_files:
            try:
                with open(exp2_files[0], 'r', encoding='utf-8') as f:
                    results['experiment_2'] = json.load(f)
            except Exception as e:
                print(f"실험 2 결과 로드 실패: {e}")
        
        # 실험 3 결과 찾기
        exp3_files = sorted(
            glob.glob(str(self.results_dir / "experiment_3_complexity_*.json")),
            reverse=True
        )
        if exp3_files:
            try:
                with open(exp3_files[0], 'r', encoding='utf-8') as f:
                    results['experiment_3'] = json.load(f)
            except Exception as e:
                print(f"실험 3 결과 로드 실패: {e}")
        
        return results
    
    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Markdown 형식의 종합 보고서 생성"""
        report = []
        
        # 헤더
        report.append("# HeaderRAG 실험 결과 종합 보고서\n")
        report.append(f"**생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n")
        report.append("---\n")
        
        # 목차
        report.append("## 목차\n")
        report.append("1. [실험 개요](#실험-개요)")
        report.append("2. [실험 1: 파싱 성능 비교](#실험-1-파싱-성능-비교)")
        report.append("3. [실험 2: RAG 성능 비교](#실험-2-rag-성능-비교)")
        report.append("4. [실험 3: 표 구조 복잡도 분석](#실험-3-표-구조-복잡도-분석)")
        report.append("5. [종합 분석 및 결론](#종합-분석-및-결론)\n")
        report.append("---\n")
        
        # 실험 개요
        report.append("## 실험 개요\n")
        report.append("본 보고서는 HeaderRAG 시스템의 성능을 평가하기 위해 수행된 세 가지 핵심 실험의 결과를 종합 분석합니다.\n")
        report.append("\n### 실험 목표\n")
        report.append("- **실험 1**: 레이블링 기반 파싱 vs Naive 파싱의 성능 비교\n")
        report.append("- **실험 2**: KG 기반 RAG vs Naive 파싱 RAG의 검색 성능 비교\n")
        report.append("- **실험 3**: 표 구조 복잡도에 따른 파싱 및 RAG 성능 분석\n")
        report.append("\n---\n")
        
        # 실험 1 결과
        if results['experiment_1']:
            report.append("## 실험 1: 파싱 성능 비교\n")
            exp1 = results['experiment_1']
            
            # 통계 요약
            if 'summary' in exp1:
                summary = exp1['summary']
                report.append("### 주요 결과\n\n")
                
                if 'labeled_parsing_stats' in summary and 'naive_parsing_stats' in summary:
                    labeled = summary['labeled_parsing_stats']
                    naive = summary['naive_parsing_stats']
                    
                    report.append("| 메트릭 | 레이블링 파싱 | Naive 파싱 | 개선율 |\n")
                    report.append("|:-------|:------------|:---------|:------|\n")
                    
                    # 파싱 시간 비교
                    if 'avg_parsing_time_ms' in labeled and 'avg_parsing_time_ms' in naive:
                        labeled_time = labeled['avg_parsing_time_ms']
                        naive_time = naive['avg_parsing_time_ms']
                        improvement = ((naive_time - labeled_time) / naive_time * 100) if naive_time > 0 else 0
                        report.append(f"| 평균 파싱 시간 (ms) | {labeled_time:.2f} | {naive_time:.2f} | {improvement:+.2f}% |\n")
                    
                    # 파싱 속도 비교
                    if 'avg_parsing_speed' in labeled and 'avg_parsing_speed' in naive:
                        labeled_speed = labeled['avg_parsing_speed']
                        naive_speed = naive['avg_parsing_speed']
                        improvement = ((labeled_speed - naive_speed) / naive_speed * 100) if naive_speed > 0 else 0
                        report.append(f"| 평균 파싱 속도 (테이블/초) | {labeled_speed:.2f} | {naive_speed:.2f} | {improvement:+.2f}% |\n")
                    
                    # 셀 수 통계
                    if 'avg_total_cells' in labeled:
                        report.append(f"| 평균 총 셀 수 | {labeled['avg_total_cells']:.1f} | - | - |\n")
                    if 'avg_header_cells' in labeled:
                        report.append(f"| 평균 헤더 셀 수 | {labeled['avg_header_cells']:.1f} | - | - |\n")
                    if 'avg_data_cells' in labeled:
                        report.append(f"| 평균 데이터 셀 수 | {labeled['avg_data_cells']:.1f} | - | - |\n")
                    if 'avg_semantic_labels' in labeled:
                        report.append(f"| 평균 시맨틱 레이블 수 | {labeled['avg_semantic_labels']:.1f} | - | - |\n")
                
                # 헤더 감지율
                if 'header_detection_rate' in summary:
                    report.append(f"\n**헤더 감지율**: {summary['header_detection_rate']:.2%}\n")
                
                # 구조 풍부도
                if 'avg_structure_richness' in summary:
                    report.append(f"**평균 구조 풍부도**: {summary['avg_structure_richness']:.3f}\n")
            
            report.append("\n### 분석\n")
            report.append("레이블링 기반 파싱은 구조화된 정보 추출 능력이 뛰어나며, 특히 헤더 감지와 시맨틱 레이블링 측면에서 Naive 파싱 대비 우수한 성능을 보입니다.\n")
            report.append("\n---\n")
        
        # 실험 2 결과
        if results['experiment_2']:
            report.append("## 실험 2: RAG 성능 비교\n")
            exp2 = results['experiment_2']
            
            if 'summary' in exp2:
                summary = exp2['summary']
                report.append("### 주요 결과\n\n")
                
                # 검색 성능 비교
                if 'kg_rag_avg' in summary and 'naive_rag_avg' in summary:
                    kg = summary['kg_rag_avg']
                    naive = summary['naive_rag_avg']
                    
                    report.append("| 메트릭 | KG 기반 RAG | Naive RAG | 개선율 |\n")
                    report.append("|:-------|:-----------|:---------|:------|\n")
                    
                    metrics = ['precision', 'recall', 'f1', 'mrr']
                    for metric in metrics:
                        if metric in kg and metric in naive:
                            kg_val = kg[metric]
                            naive_val = naive[metric]
                            improvement = ((kg_val - naive_val) / naive_val * 100) if naive_val > 0 else 0
                            report.append(f"| {metric.capitalize()} | {kg_val:.4f} | {naive_val:.4f} | {improvement:+.2f}% |\n")
                
                # 시간 통계
                if 'build_time' in summary:
                    build = summary['build_time']
                    report.append(f"\n**KG RAG 구축 시간**: {build.get('kg_rag_build_time_ms', 0):.2f}ms\n")
                    report.append(f"**Naive RAG 구축 시간**: {build.get('naive_rag_build_time_ms', 0):.2f}ms\n")
                    report.append(f"**구축 시간 비율**: {build.get('build_time_ratio', 0):.2f}x\n")
                
                if 'retrieve_time' in summary:
                    retrieve = summary['retrieve_time']
                    report.append(f"\n**KG RAG 평균 검색 시간**: {retrieve.get('kg_rag_avg_retrieve_time_ms', 0):.2f}ms\n")
                    report.append(f"**Naive RAG 평균 검색 시간**: {retrieve.get('naive_rag_avg_retrieve_time_ms', 0):.2f}ms\n")
                    report.append(f"**검색 시간 비율**: {retrieve.get('retrieve_time_ratio', 0):.2f}x\n")
                
                # 개선율
                if 'overall_improvement' in summary:
                    improvement = summary['overall_improvement']
                    report.append("\n### 성능 개선율\n\n")
                    report.append("| 메트릭 | 개선율 |\n")
                    report.append("|:-------|:------|\n")
                    for key, value in improvement.items():
                        if isinstance(value, (int, float)):
                            report.append(f"| {key} | {value:+.2f}% |\n")
            
            report.append("\n### 분석\n")
            report.append("KG 기반 RAG는 구조화된 정보를 활용하여 검색 정확도를 향상시키지만, 구축 및 검색 시간 측면에서는 일부 오버헤드가 발생할 수 있습니다.\n")
            report.append("\n---\n")
        
        # 실험 3 결과
        if results['experiment_3']:
            report.append("## 실험 3: 표 구조 복잡도 분석\n")
            exp3 = results['experiment_3']
            
            if 'complexity_distribution' in exp3:
                report.append("### 복잡도별 테이블 분포\n\n")
                dist = exp3['complexity_distribution']
                report.append("| 복잡도 유형 | 테이블 수 |\n")
                report.append("|:-----------|:--------|\n")
                complexity_names = {
                    'simple': '단순 표',
                    'nested_header': '중첩 헤더 표',
                    'merged_cell': '병합 셀 표',
                    'irregular': '비정형 표'
                }
                for comp_type, count in dist.items():
                    name = complexity_names.get(comp_type, comp_type)
                    report.append(f"| {name} | {count}개 |\n")
            
            if 'complexity_results' in exp3:
                report.append("\n### 복잡도별 성능 분석\n\n")
                for comp_type, comp_results in exp3['complexity_results'].items():
                    name = complexity_names.get(comp_type, comp_type)
                    report.append(f"#### {name}\n\n")
                    
                    if 'parsing_summary' in comp_results:
                        parsing = comp_results['parsing_summary']
                        if 'labeled_avg' in parsing and 'naive_avg' in parsing:
                            labeled = parsing['labeled_avg']
                            naive = parsing['naive_avg']
                            
                            report.append("**파싱 성능**:\n")
                            if 'parsing_time_ms' in labeled and 'parsing_time_ms' in naive:
                                report.append(f"- 평균 파싱 시간: 레이블링 {labeled['parsing_time_ms']:.2f}ms, Naive {naive['parsing_time_ms']:.2f}ms\n")
                    
                    if 'rag_summary' in comp_results:
                        rag = comp_results['rag_summary']
                        if 'kg_rag_avg' in rag and 'naive_rag_avg' in rag:
                            report.append("**RAG 성능**:\n")
                            kg = rag['kg_rag_avg']
                            naive = rag['naive_rag_avg']
                            if 'precision' in kg and 'precision' in naive:
                                report.append(f"- Precision: KG {kg['precision']:.4f}, Naive {naive['precision']:.4f}\n")
            
            report.append("\n### 분석\n")
            report.append("표 구조의 복잡도가 높을수록 레이블링 기반 파싱의 이점이 더 두드러지며, 특히 중첩 헤더나 병합 셀이 있는 경우 구조 인식 정확도가 크게 향상됩니다.\n")
            report.append("\n---\n")
        
        # 종합 분석 및 결론
        report.append("## 종합 분석 및 결론\n")
        report.append("### 주요 발견사항\n\n")
        report.append("1. **파싱 성능**: 레이블링 기반 파싱은 구조화된 정보 추출과 헤더 감지 측면에서 Naive 파싱 대비 우수한 성능을 보입니다.\n")
        report.append("2. **RAG 성능**: KG 기반 RAG는 검색 정확도 측면에서 일부 개선을 보이지만, 시간 복잡도 측면에서는 추가 연구가 필요합니다.\n")
        report.append("3. **복잡도 영향**: 표 구조가 복잡할수록 레이블링 기반 접근법의 이점이 더욱 명확하게 나타납니다.\n")
        
        report.append("\n### 향후 연구 방향\n\n")
        report.append("- 다양한 도메인에 대한 성능 평가 확대\n")
        report.append("- 임베딩 모델 최적화를 통한 검색 성능 향상\n")
        report.append("- 하이브리드 접근법 평가 (KG + Naive 결합)\n")
        report.append("- 대규모 데이터셋에 대한 확장성 평가\n")
        
        report.append("\n---\n")
        report.append(f"\n*보고서 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        return "\n".join(report)
    
    def generate_report(self) -> str:
        """종합 보고서 생성"""
        results = self.load_latest_results()
        report = self.generate_markdown_report(results)
        
        # 보고서 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comprehensive_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"종합 보고서 생성 완료: {report_path}")
        return str(report_path)


def main():
    """메인 실행 함수"""
    generator = ReportGenerator()
    report_path = generator.generate_report()
    print(f"\n보고서 경로: {report_path}")


if __name__ == "__main__":
    main()


