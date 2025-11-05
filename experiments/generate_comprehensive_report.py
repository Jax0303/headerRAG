"""
종합 실험 리포트 생성 스크립트
실험 결과를 종합하여 PDF/HTML 리포트 생성
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ComprehensiveReportGenerator:
    """종합 실험 리포트 생성기"""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "reports"):
        """
        Args:
            results_dir: 실험 결과 디렉토리
            output_dir: 리포트 출력 디렉토리
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, 
                       experiment_1_results: Optional[Dict] = None,
                       experiment_2_results: Optional[Dict] = None,
                       experiment_3_results: Optional[Dict] = None,
                       output_format: str = "html") -> str:
        """
        종합 리포트 생성
        
        Args:
            experiment_1_results: 실험 1 결과
            experiment_2_results: 실험 2 결과
            experiment_3_results: 실험 3 결과
            output_format: 출력 형식 ('html', 'md', 'txt')
        
        Returns:
            생성된 리포트 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "html":
            return self._generate_html_report(
                experiment_1_results, experiment_2_results, experiment_3_results, timestamp
            )
        elif output_format == "md":
            return self._generate_markdown_report(
                experiment_1_results, experiment_2_results, experiment_3_results, timestamp
            )
        else:
            return self._generate_text_report(
                experiment_1_results, experiment_2_results, experiment_3_results, timestamp
            )
    
    def _generate_html_report(self,
                             exp1: Optional[Dict],
                             exp2: Optional[Dict],
                             exp3: Optional[Dict],
                             timestamp: str) -> str:
        """HTML 리포트 생성"""
        html_content = self._build_html_content(exp1, exp2, exp3, timestamp)
        
        output_path = self.output_dir / f"comprehensive_report_{timestamp}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_markdown_report(self,
                                 exp1: Optional[Dict],
                                 exp2: Optional[Dict],
                                 exp3: Optional[Dict],
                                 timestamp: str) -> str:
        """Markdown 리포트 생성"""
        md_content = self._build_markdown_content(exp1, exp2, exp3, timestamp)
        
        output_path = self.output_dir / f"comprehensive_report_{timestamp}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(output_path)
    
    def _generate_text_report(self,
                             exp1: Optional[Dict],
                             exp2: Optional[Dict],
                             exp3: Optional[Dict],
                             timestamp: str) -> str:
        """텍스트 리포트 생성"""
        text_content = self._build_text_content(exp1, exp2, exp3, timestamp)
        
        output_path = self.output_dir / f"comprehensive_report_{timestamp}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        return str(output_path)
    
    def _build_html_content(self, exp1, exp2, exp3, timestamp) -> str:
        """HTML 콘텐츠 생성"""
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HeaderRAG 실험 리포트 - {timestamp}</title>
    <style>
        body {{ font-family: 'Nanum Gothic', sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #27ae60; }}
        .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .comparison {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .method {{ flex: 1; padding: 15px; margin: 10px; background-color: #fff; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>HeaderRAG 실험 종합 리포트</h1>
    <p><strong>생성 시간:</strong> {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}</p>
    
    {self._build_experiment_1_html(exp1)}
    {self._build_experiment_2_html(exp2)}
    {self._build_experiment_3_html(exp3)}
    
    <h2>결론</h2>
    {self._build_conclusion_html(exp1, exp2, exp3)}
</body>
</html>"""
        return html
    
    def _build_experiment_1_html(self, exp1: Optional[Dict]) -> str:
        """실험 1 HTML 콘텐츠"""
        if not exp1:
            return "<h2>실험 1: 파싱 성능 비교</h2><p>결과 없음</p>"
        
        summary = exp1.get('summary', {})
        
        html = """
    <h2>실험 1: 파싱 성능 비교</h2>
    <div class="summary-box">
        <h3>요약</h3>
        <ul>
            <li><strong>총 테이블 수:</strong> {total_tables}</li>
            <li><strong>평균 구조 풍부도:</strong> {richness:.3f}</li>
            <li><strong>헤더 감지율:</strong> {header_rate:.2%}</li>
        </ul>
    </div>
    
    <h3>파싱 방법별 성능 비교</h3>
    <table>
        <tr>
            <th>방법</th>
            <th>평균 파싱 시간 (ms)</th>
            <th>평균 셀 개수</th>
            <th>구조 풍부도</th>
        </tr>
        <tr>
            <td>레이블링 기반 파싱</td>
            <td class="metric">{labeled_time:.2f}</td>
            <td>{labeled_cells:.1f}</td>
            <td class="metric">{richness:.3f}</td>
        </tr>
        <tr>
            <td>Naive 파싱</td>
            <td>{naive_time:.2f}</td>
            <td>{naive_cells:.1f}</td>
            <td>-</td>
        </tr>
    </table>
        """.format(
            total_tables=summary.get('total_tables', 0),
            richness=summary.get('avg_structure_richness', 0),
            header_rate=summary.get('header_detection_rate', 0),
            labeled_time=summary.get('labeled_parsing_stats', {}).get('avg_parsing_time_ms', 0),
            labeled_cells=summary.get('labeled_parsing_stats', {}).get('avg_total_cells', 0),
            naive_time=summary.get('naive_parsing_stats', {}).get('avg_parsing_time_ms', 0),
            naive_cells=summary.get('naive_parsing_stats', {}).get('avg_total_cells', 0)
        )
        
        # 베이스라인 추가
        if 'tatr_parsing' in exp1 and exp1['tatr_parsing']:
            tatr_times = [r['stats']['parsing_time_ms'] for r in exp1['tatr_parsing']]
            tatr_cells = [r['stats']['cells_detected'] for r in exp1['tatr_parsing']]
            html += f"""
        <tr>
            <td>TATR (베이스라인)</td>
            <td>{np.mean(tatr_times):.2f}</td>
            <td>{np.mean(tatr_cells):.1f}</td>
            <td>-</td>
        </tr>
    </table>
            """
        
        return html
    
    def _build_experiment_2_html(self, exp2: Optional[Dict]) -> str:
        """실험 2 HTML 콘텐츠"""
        if not exp2:
            return "<h2>실험 2: RAG 성능 비교</h2><p>결과 없음</p>"
        
        summary = exp2.get('summary', {})
        kg_avg = summary.get('kg_rag_avg', {})
        naive_avg = summary.get('naive_rag_avg', {})
        
        html = f"""
    <h2>실험 2: RAG 성능 비교</h2>
    <div class="summary-box">
        <h3>요약</h3>
        <div class="comparison">
            <div class="method">
                <h4>KG 기반 RAG</h4>
                <ul>
                    <li>Precision: <span class="metric">{kg_avg.get('precision', 0):.3f}</span></li>
                    <li>Recall: <span class="metric">{kg_avg.get('recall', 0):.3f}</span></li>
                    <li>F1: <span class="metric">{kg_avg.get('f1', 0):.3f}</span></li>
                </ul>
            </div>
            <div class="method">
                <h4>Naive RAG</h4>
                <ul>
                    <li>Precision: {naive_avg.get('precision', 0):.3f}</li>
                    <li>Recall: {naive_avg.get('recall', 0):.3f}</li>
                    <li>F1: {naive_avg.get('f1', 0):.3f}</li>
                </ul>
            </div>
        </div>
    </div>
        """
        
        # TableRAG 베이스라인 추가
        if 'tablerag_baseline' in exp2 and exp2['tablerag_baseline']:
            tablerag_metrics = [r['metrics'] for r in exp2['tablerag_baseline']]
            tablerag_prec = np.mean([m.get('precision', 0) for m in tablerag_metrics])
            tablerag_rec = np.mean([m.get('recall', 0) for m in tablerag_metrics])
            tablerag_f1 = np.mean([m.get('f1', 0) for m in tablerag_metrics])
            
            html += f"""
            <div class="method">
                <h4>TableRAG (베이스라인)</h4>
                <ul>
                    <li>Precision: {tablerag_prec:.3f}</li>
                    <li>Recall: {tablerag_rec:.3f}</li>
                    <li>F1: {tablerag_f1:.3f}</li>
                </ul>
            </div>
        </div>
            """
        
        return html
    
    def _build_experiment_3_html(self, exp3: Optional[Dict]) -> str:
        """실험 3 HTML 콘텐츠"""
        if not exp3:
            return "<h2>실험 3: 복잡도 분석</h2><p>결과 없음</p>"
        
        complexity_dist = exp3.get('complexity_distribution', {})
        
        html = """
    <h2>실험 3: 복잡도 분석</h2>
    <table>
        <tr>
            <th>복잡도 유형</th>
            <th>테이블 수</th>
        </tr>
        """
        
        for comp_type, count in complexity_dist.items():
            html += f"<tr><td>{comp_type}</td><td>{count}</td></tr>"
        
        html += "</table>"
        return html
    
    def _build_conclusion_html(self, exp1, exp2, exp3) -> str:
        """결론 HTML 콘텐츠"""
        conclusions = []
        
        if exp1:
            conclusions.append("• 레이블링 기반 파싱이 구조 정보를 더 풍부하게 추출합니다.")
        
        if exp2:
            kg_avg = exp2.get('summary', {}).get('kg_rag_avg', {})
            naive_avg = exp2.get('summary', {}).get('naive_rag_avg', {})
            if kg_avg.get('f1', 0) > naive_avg.get('f1', 0):
                conclusions.append("• KG 기반 RAG가 Naive RAG보다 성능이 우수합니다.")
            else:
                conclusions.append("• Naive RAG와 KG 기반 RAG의 성능 차이는 크지 않습니다.")
        
        if not conclusions:
            conclusions.append("• 추가 실험 및 분석이 필요합니다.")
        
        return "<ul>" + "".join(f"<li>{c}</li>" for c in conclusions) + "</ul>"
    
    def _build_markdown_content(self, exp1, exp2, exp3, timestamp) -> str:
        """Markdown 콘텐츠 생성"""
        md = f"""# HeaderRAG 실험 종합 리포트

**생성 시간:** {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}

{self._build_experiment_1_md(exp1)}

{self._build_experiment_2_md(exp2)}

{self._build_experiment_3_md(exp3)}

## 결론

{self._build_conclusion_md(exp1, exp2, exp3)}
"""
        return md
    
    def _build_experiment_1_md(self, exp1: Optional[Dict]) -> str:
        """실험 1 Markdown"""
        if not exp1:
            return "## 실험 1: 파싱 성능 비교\n\n결과 없음\n"
        
        summary = exp1.get('summary', {})
        return f"""## 실험 1: 파싱 성능 비교

### 요약
- 총 테이블 수: {summary.get('total_tables', 0)}
- 평균 구조 풍부도: {summary.get('avg_structure_richness', 0):.3f}
- 헤더 감지율: {summary.get('header_detection_rate', 0):.2%}

### 성능 비교
| 방법 | 평균 파싱 시간 (ms) | 평균 셀 개수 |
|------|---------------------|-------------|
| 레이블링 기반 | {summary.get('labeled_parsing_stats', {}).get('avg_parsing_time_ms', 0):.2f} | {summary.get('labeled_parsing_stats', {}).get('avg_total_cells', 0):.1f} |
| Naive | {summary.get('naive_parsing_stats', {}).get('avg_parsing_time_ms', 0):.2f} | {summary.get('naive_parsing_stats', {}).get('avg_total_cells', 0):.1f} |
"""
    
    def _build_experiment_2_md(self, exp2: Optional[Dict]) -> str:
        """실험 2 Markdown"""
        if not exp2:
            return "## 실험 2: RAG 성능 비교\n\n결과 없음\n"
        
        summary = exp2.get('summary', {})
        kg_avg = summary.get('kg_rag_avg', {})
        naive_avg = summary.get('naive_rag_avg', {})
        
        md = f"""## 실험 2: RAG 성능 비교

### 성능 비교
| 방법 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| KG 기반 RAG | {kg_avg.get('precision', 0):.3f} | {kg_avg.get('recall', 0):.3f} | {kg_avg.get('f1', 0):.3f} |
| Naive RAG | {naive_avg.get('precision', 0):.3f} | {naive_avg.get('recall', 0):.3f} | {naive_avg.get('f1', 0):.3f} |
"""
        
        if 'tablerag_baseline' in exp2 and exp2['tablerag_baseline']:
            tablerag_metrics = [r['metrics'] for r in exp2['tablerag_baseline']]
            md += f"| TableRAG (베이스라인) | {np.mean([m.get('precision', 0) for m in tablerag_metrics]):.3f} | {np.mean([m.get('recall', 0) for m in tablerag_metrics]):.3f} | {np.mean([m.get('f1', 0) for m in tablerag_metrics]):.3f} |\n"
        
        return md
    
    def _build_experiment_3_md(self, exp3: Optional[Dict]) -> str:
        """실험 3 Markdown"""
        if not exp3:
            return "## 실험 3: 복잡도 분석\n\n결과 없음\n"
        
        complexity_dist = exp3.get('complexity_distribution', {})
        md = "## 실험 3: 복잡도 분석\n\n| 복잡도 유형 | 테이블 수 |\n|------------|----------|\n"
        for comp_type, count in complexity_dist.items():
            md += f"| {comp_type} | {count} |\n"
        return md
    
    def _build_conclusion_md(self, exp1, exp2, exp3) -> str:
        """결론 Markdown"""
        conclusions = []
        if exp1:
            conclusions.append("- 레이블링 기반 파싱이 구조 정보를 더 풍부하게 추출합니다.")
        if exp2:
            kg_avg = exp2.get('summary', {}).get('kg_rag_avg', {})
            naive_avg = exp2.get('summary', {}).get('naive_rag_avg', {})
            if kg_avg.get('f1', 0) > naive_avg.get('f1', 0):
                conclusions.append("- KG 기반 RAG가 Naive RAG보다 성능이 우수합니다.")
        if not conclusions:
            conclusions.append("- 추가 실험 및 분석이 필요합니다.")
        return "\n".join(conclusions)
    
    def _build_text_content(self, exp1, exp2, exp3, timestamp) -> str:
        """텍스트 콘텐츠 생성"""
        return self._build_markdown_content(exp1, exp2, exp3, timestamp)


def main():
    """메인 실행 함수"""
    import sys
    
    # 결과 파일에서 로드
    results_dir = Path("results")
    generator = ComprehensiveReportGenerator()
    
    exp1 = None
    exp2 = None
    exp3 = None
    
    # 실험 1 결과 로드
    exp1_dir = results_dir / "experiment_1"
    if exp1_dir.exists():
        cycle_files = list(exp1_dir.glob("cycle_*.json"))
        if cycle_files:
            with open(cycle_files[-1], 'r', encoding='utf-8') as f:
                exp1 = json.load(f)
    
    # 실험 2 결과 로드
    exp2_dir = results_dir / "experiment_2"
    if exp2_dir.exists():
        cycle_files = list(exp2_dir.glob("cycle_*.json"))
        if cycle_files:
            with open(cycle_files[-1], 'r', encoding='utf-8') as f:
                exp2 = json.load(f)
    
    # 실험 3 결과 로드
    exp3_dir = results_dir / "experiment_3"
    if exp3_dir.exists():
        cycle_files = list(exp3_dir.glob("cycle_*.json"))
        if cycle_files:
            with open(cycle_files[-1], 'r', encoding='utf-8') as f:
                exp3 = json.load(f)
    
    # 리포트 생성
    print("종합 리포트 생성 중...")
    report_path = generator.generate_report(exp1, exp2, exp3, output_format="html")
    print(f"리포트 생성 완료: {report_path}")
    
    # Markdown도 생성
    md_path = generator.generate_report(exp1, exp2, exp3, output_format="md")
    print(f"Markdown 리포트 생성 완료: {md_path}")


if __name__ == "__main__":
    main()

