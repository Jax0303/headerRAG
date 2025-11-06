"""
결과 분석 및 시각화 도구
- 성능 비교 테이블 생성
- 복잡도별 성능 분석
- 오류 분석
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ResultAnalyzer:
    """결과 분석 클래스"""
    
    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def create_performance_table(self,
                                results: Dict[str, Dict[str, float]],
                                metrics: List[str],
                                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        성능 비교 테이블 생성
        
        Args:
            results: {method_name: {metric: score}} 딕셔너리
            metrics: 표시할 메트릭 리스트
            output_path: 저장 경로 (선택적)
        
        Returns:
            성능 비교 DataFrame
        """
        # DataFrame 생성
        df_data = {}
        for method_name, method_results in results.items():
            df_data[method_name] = [
                method_results.get(metric, 0.0) for metric in metrics
            ]
        
        df = pd.DataFrame(df_data, index=metrics).T
        
        # LaTeX 표 형식으로 저장
        if output_path:
            latex_path = output_path.replace('.csv', '.tex')
            df.to_latex(latex_path, float_format="%.3f")
            print(f"LaTeX 표 저장: {latex_path}")
        
        # CSV로도 저장
        if output_path:
            df.to_csv(output_path)
            print(f"CSV 저장: {output_path}")
        
        return df
    
    def plot_complexity_analysis(self,
                                complexity_results: Dict[str, Dict[str, Any]],
                                metric: str = 'grits_overall',
                                output_path: Optional[str] = None):
        """
        복잡도별 성능 분석 차트 생성
        
        Args:
            complexity_results: 복잡도별 실험 결과
                {complexity_level: {method: scores}}
            metric: 분석할 메트릭
            output_path: 저장 경로
        """
        # 데이터 준비
        data = []
        for level, level_results in complexity_results.items():
            if level_results is None:
                continue
            
            parsing_results = level_results.get('parsing', {})
            if not parsing_results:
                continue
            
            # 각 방법별 점수 추출
            for method_name, method_results in parsing_results.items():
                if isinstance(method_results, list):
                    scores = [
                        r.get('metrics', {}).get(metric, 0.0)
                        for r in method_results
                        if 'metrics' in r
                    ]
                    if scores:
                        data.append({
                            'complexity_level': level,
                            'method': method_name,
                            'score': np.mean(scores)
                        })
        
        if not data:
            print("플롯할 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(data)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 복잡도 레벨 순서
        level_order = ['low', 'medium', 'high']
        df['complexity_level'] = pd.Categorical(
            df['complexity_level'],
            categories=level_order,
            ordered=True
        )
        
        # 막대 그래프
        sns.barplot(
            data=df,
            x='complexity_level',
            y='score',
            hue='method',
            ax=ax
        )
        
        ax.set_xlabel('Table Complexity', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'Performance by Table Complexity ({metric})', fontsize=14)
        ax.legend(title='Method', loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"차트 저장: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_errors(self,
                      failed_cases: List[Dict[str, Any]],
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        오류 분석
        
        Args:
            failed_cases: 실패한 케이스 리스트
                각 케이스는 다음 키를 포함:
                - table_id: 테이블 ID
                - error_type: 오류 유형
                - error_message: 오류 메시지
                - complexity_level: 복잡도 레벨 (선택적)
                - domain: 도메인 (선택적)
        
        Returns:
            오류 분석 결과 딕셔너리
        """
        if not failed_cases:
            return {}
        
        df = pd.DataFrame(failed_cases)
        
        # 오류 유형별 분류
        error_type_counts = df['error_type'].value_counts() if 'error_type' in df.columns else {}
        
        # 복잡도별 오류 분포
        complexity_error_counts = None
        if 'complexity_level' in df.columns:
            complexity_error_counts = pd.crosstab(
                df['complexity_level'],
                df['error_type']
            ) if 'error_type' in df.columns else None
        
        # 도메인별 오류 분포
        domain_error_counts = None
        if 'domain' in df.columns:
            domain_error_counts = pd.crosstab(
                df['domain'],
                df['error_type']
            ) if 'error_type' in df.columns else None
        
        analysis = {
            'total_errors': len(failed_cases),
            'error_type_distribution': error_type_counts.to_dict() if isinstance(error_type_counts, pd.Series) else {},
            'complexity_error_distribution': complexity_error_counts.to_dict() if complexity_error_counts is not None else {},
            'domain_error_distribution': domain_error_counts.to_dict() if domain_error_counts is not None else {}
        }
        
        # 시각화
        if output_path:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 오류 유형별 분포
            if error_type_counts is not None and len(error_type_counts) > 0:
                error_type_counts.plot(kind='bar', ax=axes[0])
                axes[0].set_title('Error Distribution by Type')
                axes[0].set_xlabel('Error Type')
                axes[0].set_ylabel('Count')
                axes[0].tick_params(axis='x', rotation=45)
            
            # 복잡도별 오류 분포
            if complexity_error_counts is not None and len(complexity_error_counts) > 0:
                complexity_error_counts.plot(kind='bar', ax=axes[1], stacked=True)
                axes[1].set_title('Error Distribution by Complexity Level')
                axes[1].set_xlabel('Complexity Level')
                axes[1].set_ylabel('Count')
                axes[1].legend(title='Error Type')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"오류 분석 차트 저장: {output_path}")
            plt.close()
        
        return analysis
    
    def create_summary_report(self,
                            experiment_results: Dict[str, Any],
                            output_path: str):
        """
        종합 리포트 생성
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            output_path: 리포트 저장 경로
        """
        report_lines = []
        report_lines.append("# 실험 결과 종합 리포트\n")
        report_lines.append(f"생성 시간: {pd.Timestamp.now()}\n\n")
        
        # 실험 1: 파싱 성능
        if 'parsing' in experiment_results:
            report_lines.append("## 실험 1: 파싱 성능 비교\n\n")
            parsing_results = experiment_results['parsing']
            
            # 메트릭 요약
            for method_name, method_results in parsing_results.items():
                if isinstance(method_results, list):
                    metrics_list = [
                        r.get('metrics', {})
                        for r in method_results
                        if 'metrics' in r
                    ]
                    
                    if metrics_list:
                        metrics_df = pd.DataFrame(metrics_list)
                        report_lines.append(f"### {method_name}\n\n")
                        report_lines.append(metrics_df.describe().to_markdown())
                        report_lines.append("\n\n")
        
        # 실험 2: RAG 성능
        if 'rag' in experiment_results:
            report_lines.append("## 실험 2: RAG 성능 비교\n\n")
            rag_results = experiment_results['rag']
            
            # 메트릭 요약
            for method_name, method_results in rag_results.items():
                if isinstance(method_results, list):
                    metrics_list = [
                        r.get('metrics', {})
                        for r in method_results
                        if 'metrics' in r
                    ]
                    
                    if metrics_list:
                        metrics_df = pd.DataFrame(metrics_list)
                        report_lines.append(f"### {method_name}\n\n")
                        report_lines.append(metrics_df.describe().to_markdown())
                        report_lines.append("\n\n")
        
        # 복잡도 분석
        if 'complexity' in experiment_results:
            report_lines.append("## 복잡도별 성능 분석\n\n")
            complexity_results = experiment_results['complexity']
            
            for level, level_results in complexity_results.items():
                if level_results is None:
                    continue
                
                report_lines.append(f"### {level.upper()} Complexity\n\n")
                
                # 간단한 요약 통계
                if 'parsing' in level_results:
                    parsing_results = level_results['parsing']
                    for method_name, method_results in parsing_results.items():
                        if isinstance(method_results, list):
                            scores = [
                                r.get('metrics', {}).get('grits_overall', 0.0)
                                for r in method_results
                                if 'metrics' in r
                            ]
                            if scores:
                                mean_score = np.mean(scores)
                                std_score = np.std(scores)
                                report_lines.append(
                                    f"- **{method_name}**: {mean_score:.3f} ± {std_score:.3f}\n"
                                )
        
        # 리포트 저장
        report_text = '\n'.join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"리포트 저장: {output_path}")



