"""
평가 결과 로깅 시스템
실험마다 TXT 파일로 평가 결과 기록
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class EvaluationLogger:
    """평가 결과 로깅 클래스"""
    
    def __init__(self, base_dir: str = "results"):
        """
        Args:
            base_dir: 평가 결과 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험별 디렉토리 생성
        self.experiment_dirs = {
            'experiment_1_parsing': self.base_dir / "experiment_1",
            'experiment_2_rag': self.base_dir / "experiment_2",
            'experiment_3_complexity': self.base_dir / "experiment_3"
        }
        for exp_dir in self.experiment_dirs.values():
            exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험별 로그 디렉토리 생성
        for exp_name, exp_dir in self.experiment_dirs.items():
            log_dir = exp_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험별 결과 파일 경로 (사이클별로 구분하기 위해)
        self.parsing_results_file = self.experiment_dirs['experiment_1_parsing'] / "logs" / "experiment_results.txt"
        self.rag_results_file = self.experiment_dirs['experiment_2_rag'] / "logs" / "experiment_results.txt"
    
    def log_parsing_evaluation(self,
                             experiment_name: str,
                             table_id: str,
                             results: Dict[str, Any],
                             timestamp: Optional[str] = None) -> str:
        """
        파싱 평가 결과 로깅 (append 모드)
        
        Args:
            experiment_name: 실험 이름
            table_id: 테이블 ID
            results: 평가 결과 딕셔너리
            timestamp: 타임스탬프 (None이면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # TXT 내용 작성
        content = self._format_parsing_results(experiment_name, table_id, results, timestamp)
        
        # 파일에 append 모드로 저장
        with open(self.parsing_results_file, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write("\n\n")  # 실험 간 구분을 위한 빈 줄
        
        return str(self.parsing_results_file)
    
    def log_rag_evaluation(self,
                          experiment_name: str,
                          query_id: str,
                          results: Dict[str, Any],
                          timestamp: Optional[str] = None) -> str:
        """
        RAG 평가 결과 로깅 (append 모드)
        
        Args:
            experiment_name: 실험 이름
            query_id: 쿼리 ID
            results: 평가 결과 딕셔너리
            timestamp: 타임스탬프 (None이면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # TXT 내용 작성
        content = self._format_rag_results(experiment_name, query_id, results, timestamp)
        
        # 파일에 append 모드로 저장
        with open(self.rag_results_file, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write("\n\n")  # 실험 간 구분을 위한 빈 줄
        
        return str(self.rag_results_file)
    
    def log_summary(self,
                   experiment_name: str,
                   evaluation_type: str,
                   summary: Dict[str, Any],
                   timestamp: Optional[str] = None) -> str:
        """
        평가 요약 로깅 (append 모드)
        
        Args:
            experiment_name: 실험 이름
            evaluation_type: 평가 타입 ('parsing' or 'rag')
            summary: 요약 딕셔너리
            timestamp: 타임스탬프
        
        Returns:
            저장된 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 평가 타입에 따라 파일 선택
        if evaluation_type == 'parsing':
            filepath = self.parsing_results_file
        elif evaluation_type == 'rag':
            filepath = self.rag_results_file
        else:
            filepath = self.base_dir / f"{evaluation_type}_results.txt"
        
        content = self._format_summary(experiment_name, evaluation_type, summary, timestamp)
        
        # 파일에 append 모드로 저장
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write("\n\n")  # 실험 간 구분을 위한 빈 줄
        
        return str(filepath)
    
    def _format_parsing_results(self,
                               experiment_name: str,
                               table_id: str,
                               results: Dict[str, Any],
                               timestamp: str) -> str:
        """파싱 평가 결과 포맷팅 (표 형태)"""
        labeled_stats = results.get('labeled_parsing', {}).get('stats', {})
        naive_stats = results.get('naive_parsing', {}).get('stats', {})
        comp = results.get('comparison', {})
        
        # 표 형태로 포맷팅
        lines = [
            f"## 파싱 평가 결과 - {table_id}",
            f"**실험명**: {experiment_name} | **평가 시간**: {timestamp}",
            "",
            "### 평가 메트릭 비교표",
            "",
            "| 구분 | 레이블링 파싱 | Naive 파싱 |",
            "|:-----|:-------------|:----------|",
        ]
        
        # 총 셀 수
        lines.append(f"| 총 셀 수 | {labeled_stats.get('total_cells', 'N/A')} | {naive_stats.get('total_cells', 'N/A')} |")
        
        # 헤더 셀 수
        lines.append(f"| 헤더 셀 수 | {labeled_stats.get('header_cells', 'N/A')} | - |")
        
        # 데이터 셀 수
        lines.append(f"| 데이터 셀 수 | {labeled_stats.get('data_cells', 'N/A')} | - |")
        
        # 컬럼 수
        lines.append(f"| 컬럼 수 | - | {naive_stats.get('columns', 'N/A')} |")
        
        # 시맨틱 레이블 수
        lines.append(f"| 시맨틱 레이블 수 | {labeled_stats.get('semantic_labels', 'N/A')} | - |")
        
        # 파싱 시간
        if 'parsing_time_ms' in labeled_stats and 'parsing_time_ms' in naive_stats:
            lines.append(f"| 파싱 시간 (ms) | {labeled_stats.get('parsing_time_ms', 0):.2f} | {naive_stats.get('parsing_time_ms', 0):.2f} |")
        
        # 파싱 속도
        if 'parsing_speed_tables_per_sec' in labeled_stats and 'parsing_speed_tables_per_sec' in naive_stats:
            lines.append(f"| 파싱 속도 (테이블/초) | {labeled_stats.get('parsing_speed_tables_per_sec', 0):.2f} | {naive_stats.get('parsing_speed_tables_per_sec', 0):.2f} |")
        
        lines.append("")
        lines.append("### 비교 결과")
        lines.append("")
        lines.append("| 메트릭 | 값 |")
        lines.append("|:-------|:---|")
        lines.append(f"| 구조 풍부도 | {comp.get('structure_richness', 'N/A'):.4f} |")
        lines.append(f"| 헤더 감지 | {'성공' if comp.get('header_detection', False) else '실패'} |")
        
        if 'speed_ratio' in comp:
            speed_ratio = comp.get('speed_ratio', 0)
            speed_improvement = comp.get('speed_improvement_percent', 0)
            lines.append(f"| 속도 비율 (Naive/Labeled) | {speed_ratio:.2f}x |")
            lines.append(f"| 속도 개선율 | {speed_improvement:.2f}% |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_rag_results(self,
                           experiment_name: str,
                           query_id: str,
                           results: Dict[str, Any],
                           timestamp: str) -> str:
        """RAG 평가 결과 포맷팅 (표 형태)"""
        kg = results.get('kg_rag', {})
        naive = results.get('naive_rag', {})
        comp = results.get('comparison', {})
        kg_metrics = kg.get('metrics', {})
        naive_metrics = naive.get('metrics', {})
        
        lines = [
            f"## RAG 평가 결과 - {query_id}",
            f"**실험명**: {experiment_name} | **평가 시간**: {timestamp}",
            "",
        ]
        
        # 쿼리 정보
        if 'query' in results:
            lines.extend([
                f"**쿼리**: {results['query']}",
                "",
            ])
        
        # 평가 메트릭 비교표
        lines.extend([
            "### 평가 메트릭 비교표",
            "",
            "| 메트릭 | KG 기반 RAG | Naive RAG |",
            "|:-------|:-----------|:---------|",
        ])
        
        # 검색 성능 메트릭
        if kg_metrics or naive_metrics:
            def format_metric(val):
                if isinstance(val, (int, float)):
                    return f"{val:.4f}"
                return "N/A"
            
            kg_prec = format_metric(kg_metrics.get('precision'))
            naive_prec = format_metric(naive_metrics.get('precision'))
            kg_rec = format_metric(kg_metrics.get('recall'))
            naive_rec = format_metric(naive_metrics.get('recall'))
            kg_f1 = format_metric(kg_metrics.get('f1'))
            naive_f1 = format_metric(naive_metrics.get('f1'))
            kg_mrr = format_metric(kg_metrics.get('mrr'))
            naive_mrr = format_metric(naive_metrics.get('mrr'))
            
            lines.append(f"| Precision | {kg_prec} | {naive_prec} |")
            lines.append(f"| Recall | {kg_rec} | {naive_rec} |")
            lines.append(f"| F1 Score | {kg_f1} | {naive_f1} |")
            lines.append(f"| MRR | {kg_mrr} | {naive_mrr} |")
        
        # ROUGE 스코어
        def format_metric(val):
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return "N/A"
        
        if 'rouge1' in kg or 'rouge1' in naive:
            kg_r1 = format_metric(kg.get('rouge1'))
            naive_r1 = format_metric(naive.get('rouge1'))
            kg_r2 = format_metric(kg.get('rouge2'))
            naive_r2 = format_metric(naive.get('rouge2'))
            kg_rl = format_metric(kg.get('rougeL'))
            naive_rl = format_metric(naive.get('rougeL'))
            
            lines.append(f"| ROUGE-1 | {kg_r1} | {naive_r1} |")
            lines.append(f"| ROUGE-2 | {kg_r2} | {naive_r2} |")
            lines.append(f"| ROUGE-L | {kg_rl} | {naive_rl} |")
        
        # 정답 여부
        if 'correct' in kg or 'correct' in naive:
            kg_ox = 'O' if kg.get('correct', False) else 'X'
            naive_ox = 'O' if naive.get('correct', False) else 'X'
            lines.append(f"| 정답 여부 | {kg_ox} | {naive_ox} |")
        
        # 검색 시간
        if 'retrieve_time_ms' in kg or 'retrieve_time_ms' in naive:
            kg_time = format_metric(kg.get('retrieve_time_ms')) if isinstance(kg.get('retrieve_time_ms'), (int, float)) else "N/A"
            naive_time = format_metric(naive.get('retrieve_time_ms')) if isinstance(naive.get('retrieve_time_ms'), (int, float)) else "N/A"
            lines.append(f"| 검색 시간 (ms) | {kg_time} | {naive_time} |")
        
        lines.append("")
        
        # 비교 결과
        if 'improvement' in comp and comp['improvement']:
            lines.append("### 비교 결과")
            lines.append("")
            lines.append("| 메트릭 | 개선율 |")
            lines.append("|:-------|:------|")
            for metric, value in comp['improvement'].items():
                lines.append(f"| {metric} | {value:.2f}% |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_summary(self,
                       experiment_name: str,
                       evaluation_type: str,
                       summary: Dict[str, Any],
                       timestamp: str) -> str:
        """요약 결과 포맷팅 (표 형태)"""
        lines = [
            f"# {evaluation_type.upper()} 평가 요약",
            f"**실험명**: {experiment_name} | **평가 시간**: {timestamp}",
            "",
        ]
        
        if evaluation_type == 'parsing':
            # 파싱 요약 표 형태
            lines.append("## 전체 통계")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|:-----|:---|")
            
            if 'total_tables' in summary:
                lines.append(f"| 총 테이블 수 | {summary['total_tables']} |")
            if 'avg_structure_richness' in summary:
                lines.append(f"| 평균 구조 풍부도 | {summary['avg_structure_richness']:.4f} |")
            if 'header_detection_rate' in summary:
                lines.append(f"| 헤더 감지율 | {summary['header_detection_rate']:.4f} |")
            if 'header_detection_count' in summary:
                lines.append(f"| 헤더 감지 성공 수 | {summary['header_detection_count']} |")
            
            lines.append("")
            
            # 레이블링 파싱 통계
            if 'labeled_parsing_stats' in summary:
                labeled = summary['labeled_parsing_stats']
                lines.append("### 레이블링 파싱 통계")
                lines.append("")
                lines.append("| 메트릭 | 값 |")
                lines.append("|:-------|:---|")
                if 'avg_total_cells' in labeled:
                    lines.append(f"| 평균 총 셀 수 | {labeled['avg_total_cells']:.2f} |")
                if 'avg_header_cells' in labeled:
                    lines.append(f"| 평균 헤더 셀 수 | {labeled['avg_header_cells']:.2f} |")
                if 'avg_data_cells' in labeled:
                    lines.append(f"| 평균 데이터 셀 수 | {labeled['avg_data_cells']:.2f} |")
                if 'avg_semantic_labels' in labeled:
                    lines.append(f"| 평균 시맨틱 레이블 수 | {labeled['avg_semantic_labels']:.2f} |")
                if 'avg_parsing_time_ms' in labeled:
                    lines.append(f"| 평균 파싱 시간 (ms) | {labeled['avg_parsing_time_ms']:.2f} |")
                if 'min_parsing_time_ms' in labeled:
                    lines.append(f"| 최소 파싱 시간 (ms) | {labeled['min_parsing_time_ms']:.2f} |")
                if 'max_parsing_time_ms' in labeled:
                    lines.append(f"| 최대 파싱 시간 (ms) | {labeled['max_parsing_time_ms']:.2f} |")
                if 'total_parsing_time_ms' in labeled:
                    lines.append(f"| 총 파싱 시간 (ms) | {labeled['total_parsing_time_ms']:.2f} |")
                if 'avg_parsing_speed_tables_per_sec' in labeled:
                    lines.append(f"| 평균 파싱 속도 (테이블/초) | {labeled['avg_parsing_speed_tables_per_sec']:.2f} |")
                lines.append("")
            
            # Naive 파싱 통계
            if 'naive_parsing_stats' in summary:
                naive = summary['naive_parsing_stats']
                lines.append("### Naive 파싱 통계")
                lines.append("")
                lines.append("| 메트릭 | 값 |")
                lines.append("|:-------|:---|")
                if 'avg_total_cells' in naive:
                    lines.append(f"| 평균 총 셀 수 | {naive['avg_total_cells']:.2f} |")
                if 'avg_columns' in naive:
                    lines.append(f"| 평균 컬럼 수 | {naive['avg_columns']:.2f} |")
                if 'avg_parsing_time_ms' in naive:
                    lines.append(f"| 평균 파싱 시간 (ms) | {naive['avg_parsing_time_ms']:.2f} |")
                if 'min_parsing_time_ms' in naive:
                    lines.append(f"| 최소 파싱 시간 (ms) | {naive['min_parsing_time_ms']:.2f} |")
                if 'max_parsing_time_ms' in naive:
                    lines.append(f"| 최대 파싱 시간 (ms) | {naive['max_parsing_time_ms']:.2f} |")
                if 'total_parsing_time_ms' in naive:
                    lines.append(f"| 총 파싱 시간 (ms) | {naive['total_parsing_time_ms']:.2f} |")
                if 'avg_parsing_speed_tables_per_sec' in naive:
                    lines.append(f"| 평균 파싱 속도 (테이블/초) | {naive['avg_parsing_speed_tables_per_sec']:.2f} |")
                lines.append("")
            
            # 속도 비교
            if 'speed_comparison' in summary:
                speed = summary['speed_comparison']
                lines.append("### 속도 비교")
                lines.append("")
                lines.append("| 메트릭 | 값 |")
                lines.append("|:-------|:---|")
                if 'avg_speed_ratio' in speed:
                    lines.append(f"| 평균 속도 비율 (Naive/Labeled) | {speed['avg_speed_ratio']:.2f}x |")
                if 'avg_speed_improvement_percent' in speed:
                    lines.append(f"| 평균 속도 개선율 | {speed['avg_speed_improvement_percent']:.2f}% |")
                if 'total_time_saved_ms' in speed:
                    time_saved = speed['total_time_saved_ms']
                    if time_saved >= 1000:
                        lines.append(f"| 총 절약 시간 | {time_saved/1000:.2f} 초 |")
                    else:
                        lines.append(f"| 총 절약 시간 | {time_saved:.2f} ms |")
                lines.append("")
        
        elif evaluation_type == 'rag':
            # RAG 요약 표 형태
            lines.append("## 전체 통계")
            lines.append("")
            
            # 평균 메트릭 비교표
            if 'kg_rag_avg' in summary and 'naive_rag_avg' in summary:
                kg_avg = summary['kg_rag_avg']
                naive_avg = summary['naive_rag_avg']
                lines.append("### 평균 검색 성능 비교")
                lines.append("")
                lines.append("| 메트릭 | KG 기반 RAG | Naive RAG |")
                lines.append("|:-------|:-----------|:---------|")
                for key in ['precision', 'recall', 'f1', 'mrr']:
                    if key in kg_avg and key in naive_avg:
                        lines.append(f"| {key.capitalize()} | {kg_avg[key]:.4f} | {naive_avg[key]:.4f} |")
                lines.append("")
            
            # 구축 시간
            if 'build_time' in summary:
                build = summary['build_time']
                lines.append("### 시스템 구축 시간")
                lines.append("")
                lines.append("| 항목 | 시간 (ms) |")
                lines.append("|:-----|:---------|")
                if 'kg_rag_build_time_ms' in build:
                    lines.append(f"| KG RAG 구축 시간 | {build['kg_rag_build_time_ms']:.2f} |")
                if 'naive_rag_build_time_ms' in build:
                    lines.append(f"| Naive RAG 구축 시간 | {build['naive_rag_build_time_ms']:.2f} |")
                if 'build_time_ratio' in build:
                    lines.append(f"| 구축 시간 비율 (KG/Naive) | {build['build_time_ratio']:.2f}x |")
                lines.append("")
            
            # 검색 시간
            if 'retrieve_time' in summary:
                retrieve = summary['retrieve_time']
                lines.append("### 검색 시간 통계")
                lines.append("")
                lines.append("| 메트릭 | KG 기반 RAG | Naive RAG |")
                lines.append("|:-------|:-----------|:---------|")
                
                def format_time(val):
                    if isinstance(val, (int, float)):
                        return f"{val:.2f}"
                    return "N/A"
                
                if 'kg_rag_avg_retrieve_time_ms' in retrieve:
                    kg_avg = format_time(retrieve['kg_rag_avg_retrieve_time_ms'])
                    naive_avg = format_time(retrieve.get('naive_rag_avg_retrieve_time_ms'))
                    lines.append(f"| 평균 검색 시간 (ms) | {kg_avg} | {naive_avg} |")
                if 'kg_rag_min_retrieve_time_ms' in retrieve:
                    kg_min = format_time(retrieve['kg_rag_min_retrieve_time_ms'])
                    naive_min = format_time(retrieve.get('naive_rag_min_retrieve_time_ms'))
                    lines.append(f"| 최소 검색 시간 (ms) | {kg_min} | {naive_min} |")
                if 'kg_rag_max_retrieve_time_ms' in retrieve:
                    kg_max = format_time(retrieve['kg_rag_max_retrieve_time_ms'])
                    naive_max = format_time(retrieve.get('naive_rag_max_retrieve_time_ms'))
                    lines.append(f"| 최대 검색 시간 (ms) | {kg_max} | {naive_max} |")
                if 'retrieve_time_ratio' in retrieve:
                    ratio = format_time(retrieve['retrieve_time_ratio'])
                    lines.append(f"| 검색 시간 비율 (KG/Naive) | {ratio}x |")
                lines.append("")
            
            # 개선율
            if 'overall_improvement' in summary and 'improvement' in summary['overall_improvement']:
                improvement = summary['overall_improvement']['improvement']
                lines.append("### 성능 개선율")
                lines.append("")
                lines.append("| 메트릭 | 개선율 (%) |")
                lines.append("|:-------|:----------|")
                for metric, value in improvement.items():
                    lines.append(f"| {metric.capitalize()} | {value:.2f} |")
                lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return "\n".join(lines)

