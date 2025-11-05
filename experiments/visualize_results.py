"""
실험 결과 시각화 스크립트
텍스트 결과 파일을 파싱하여 직관적인 그래프 생성
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json

# 한글 폰트 설정
import platform
import os
from pathlib import Path

def setup_korean_font():
    """한글 폰트 설정 함수"""
    # 시스템에서 사용 가능한 한글 폰트 찾기
    font_list = [
        'NanumGothic', 'NanumBarunGothic', 'NanumBarunGothicOTF',
        'Noto Sans CJK KR', 'Noto Sans KR', 'NotoSansCJKkr-Regular',
        'AppleGothic', 'Malgun Gothic', 'Gulim', 'Dotum'
    ]
    
    # matplotlib에서 사용 가능한 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 한글 폰트 찾기
    korean_font = None
    for font in font_list:
        if font in available_fonts:
            korean_font = font
            break
    
    # 한글 폰트를 찾지 못한 경우, Noto Sans 폰트 다운로드 시도
    if korean_font is None:
        try:
            # 폰트 캐시 디렉토리 확인
            font_cache_dir = Path.home() / '.matplotlib' / 'fonts'
            font_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 시스템 폰트 경로 확인
            system_font_paths = [
                '/usr/share/fonts',
                '/usr/local/share/fonts',
                '/usr/share/fonts/truetype',
                '/usr/share/fonts/truetype/noto',
                '/usr/share/fonts/truetype/nanum',
            ]
            
            # 시스템 폰트 경로에서 한글 폰트 찾기
            for font_path in system_font_paths:
                if os.path.exists(font_path):
                    for root, dirs, files in os.walk(font_path):
                        for file in files:
                            if file.endswith(('.ttf', '.otf')):
                                if any(k in file.lower() for k in ['nanum', 'noto', 'gulim', 'dotum']):
                                    # 폰트 파일 경로를 matplotlib에 등록
                                    font_path_full = os.path.join(root, file)
                                    try:
                                        fm.fontManager.addfont(font_path_full)
                                        # 폰트 이름 추출
                                        font_prop = fm.FontProperties(fname=font_path_full)
                                        korean_font = font_prop.get_name()
                                        print(f"한글 폰트 발견: {file} ({korean_font})")
                                        break
                                    except:
                                        continue
                        if korean_font:
                            break
                    if korean_font:
                        break
        except Exception as e:
            print(f"폰트 검색 중 오류 발생: {e}")
    
    # 폰트 설정
    if korean_font:
        plt.rcParams['font.family'] = korean_font
        print(f"한글 폰트 설정 완료: {korean_font}")
    else:
        # 한글 폰트를 찾지 못한 경우, DejaVu Sans 사용 (한글 깨짐 방지를 위해)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("경고: 한글 폰트를 찾을 수 없습니다. 한글이 깨질 수 있습니다.")
        print("해결 방법: sudo apt-get install fonts-nanum 또는 fonts-noto-cjk 설치")
    
    plt.rcParams['axes.unicode_minus'] = False
    # 폰트 캐시 새로고침
    try:
        fm._rebuild()
    except:
        pass

# 한글 폰트 설정 실행
setup_korean_font()

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultVisualizer:
    """실험 결과 시각화 클래스"""
    
    def __init__(self, results_dir: str = "results"):
        """
        Args:
            results_dir: 결과 디렉토리 경로
        """
        self.results_dir = Path(results_dir)
        self.parsing_dir = self.results_dir / "parsing"
        self.rag_dir = self.results_dir / "rag"
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_text_results(self, filepath: Path) -> Dict[str, Any]:
        """
        텍스트 결과 파일 파싱
        
        Args:
            filepath: 결과 파일 경로
            
        Returns:
            파싱된 데이터 딕셔너리
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = {
            'individual_results': [],
            'summaries': []
        }
        
        # 개별 결과 파싱
        pattern = r'## 파싱 평가 결과 - (table_\d+).*?\*\*평가 시간\*\*: (\d{8}_\d{6})'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            table_id = match.group(1)
            timestamp = match.group(2)
            
            # 해당 테이블의 데이터 추출
            table_section = match.group(0)
            
            # 레이블링 파싱 데이터
            labeled_data = {}
            naive_data = {}
            
            # 총 셀 수: | 총 셀 수 | 16 | 4 |
            m = re.search(r'\| 총 셀 수 \| (\d+) \| (\d+) \|', table_section)
            if m:
                labeled_data['total_cells'] = int(m.group(1))
                naive_data['total_cells'] = int(m.group(2))
            
            # 헤더 셀 수: | 헤더 셀 수 | 0 | - |
            m = re.search(r'\| 헤더 셀 수 \| (\d+) \|', table_section)
            if m:
                labeled_data['header_cells'] = int(m.group(1))
            
            # 데이터 셀 수: | 데이터 셀 수 | 16 | - |
            m = re.search(r'\| 데이터 셀 수 \| (\d+) \|', table_section)
            if m:
                labeled_data['data_cells'] = int(m.group(1))
            
            # 컬럼 수: | 컬럼 수 | - | 4 |
            m = re.search(r'\| 컬럼 수 \| - \| (\d+) \|', table_section)
            if m:
                naive_data['columns'] = int(m.group(1))
            
            # 시맨틱 레이블 수: | 시맨틱 레이블 수 | 16 | - |
            m = re.search(r'\| 시맨틱 레이블 수 \| (\d+) \|', table_section)
            if m:
                labeled_data['semantic_labels'] = int(m.group(1))
            
            # 파싱 시간: | 파싱 시간 (ms) | 1.92 | 0.95 |
            m = re.search(r'\| 파싱 시간 \(ms\) \| ([\d.]+) \| ([\d.]+) \|', table_section)
            if m:
                labeled_data['parsing_time_ms'] = float(m.group(1))
                naive_data['parsing_time_ms'] = float(m.group(2))
            
            # 파싱 속도: | 파싱 속도 (테이블/초) | 520.84 | 1055.70 |
            m = re.search(r'\| 파싱 속도 \(테이블/초\) \| ([\d.]+) \| ([\d.]+) \|', table_section)
            if m:
                labeled_data['parsing_speed'] = float(m.group(1))
                naive_data['parsing_speed'] = float(m.group(2))
            
            # 비교 결과
            comparison = {}
            comp_patterns = {
                'structure_richness': r'\| 구조 풍부도 \| ([\d.]+) \|',
                'header_detection': r'\| 헤더 감지 \| (성공|실패) \|',
                'speed_ratio': r'\| 속도 비율 \(Naive/Labeled\) \| ([\d.]+)x \|',
                'speed_improvement': r'\| 속도 개선율 \| ([\d.-]+)% \|'
            }
            
            for key, pattern in comp_patterns.items():
                m = re.search(pattern, table_section)
                if m:
                    val = m.group(1)
                    if key == 'header_detection':
                        comparison[key] = val == '성공'
                    else:
                        comparison[key] = float(val)
            
            # naive_data가 비어있지 않은 경우에만 추가
            if not naive_data:
                naive_data = {}
            
            data['individual_results'].append({
                'table_id': table_id,
                'timestamp': timestamp,
                'labeled_parsing': labeled_data,
                'naive_parsing': naive_data,
                'comparison': comparison
            })
        
        # 요약 섹션 파싱
        summary_pattern = r'# PARSING 평가 요약.*?\*\*평가 시간\*\*: (\d{8}_\d{6})(.*?)(?=---|$)'
        summary_matches = re.finditer(summary_pattern, content, re.DOTALL)
        
        for match in summary_matches:
            timestamp = match.group(1)
            summary_section = match.group(2)
            
            summary_data = {}
            
            # 전체 통계
            summary_data['total_tables'] = self._extract_value(summary_section, r'\| 총 테이블 수 \| (\d+) \|')
            summary_data['avg_structure_richness'] = self._extract_value(summary_section, r'\| 평균 구조 풍부도 \| ([\d.]+) \|')
            summary_data['header_detection_rate'] = self._extract_value(summary_section, r'\| 헤더 감지율 \| ([\d.]+) \|')
            summary_data['header_detection_count'] = self._extract_value(summary_section, r'\| 헤더 감지 성공 수 \| (\d+) \|')
            
            # 레이블링 파싱 통계
            labeled_stats = {}
            labeled_stats['avg_total_cells'] = self._extract_value(summary_section, r'\| 평균 총 셀 수 \| ([\d.]+) \|')
            labeled_stats['avg_header_cells'] = self._extract_value(summary_section, r'\| 평균 헤더 셀 수 \| ([\d.]+) \|')
            labeled_stats['avg_data_cells'] = self._extract_value(summary_section, r'\| 평균 데이터 셀 수 \| ([\d.]+) \|')
            labeled_stats['avg_semantic_labels'] = self._extract_value(summary_section, r'\| 평균 시맨틱 레이블 수 \| ([\d.]+) \|')
            labeled_stats['avg_parsing_time_ms'] = self._extract_value(summary_section, r'\| 평균 파싱 시간 \(ms\) \| ([\d.]+) \|')
            labeled_stats['min_parsing_time_ms'] = self._extract_value(summary_section, r'\| 최소 파싱 시간 \(ms\) \| ([\d.]+) \|')
            labeled_stats['max_parsing_time_ms'] = self._extract_value(summary_section, r'\| 최대 파싱 시간 \(ms\) \| ([\d.]+) \|')
            labeled_stats['total_parsing_time_ms'] = self._extract_value(summary_section, r'\| 총 파싱 시간 \(ms\) \| ([\d.]+) \|')
            labeled_stats['avg_parsing_speed'] = self._extract_value(summary_section, r'\| 평균 파싱 속도 \(테이블/초\) \| ([\d.]+) \|')
            
            summary_data['labeled_parsing_stats'] = labeled_stats
            
            # Naive 파싱 통계
            naive_stats = {}
            naive_stats['avg_total_cells'] = self._extract_value(summary_section, r'\| 평균 총 셀 수 \| ([\d.]+) \|', after='Naive 파싱 통계')
            naive_stats['avg_columns'] = self._extract_value(summary_section, r'\| 평균 컬럼 수 \| ([\d.]+) \|')
            naive_stats['avg_parsing_time_ms'] = self._extract_value(summary_section, r'\| 평균 파싱 시간 \(ms\) \| ([\d.]+) \|', after='Naive 파싱 통계')
            naive_stats['min_parsing_time_ms'] = self._extract_value(summary_section, r'\| 최소 파싱 시간 \(ms\) \| ([\d.]+) \|', after='Naive 파싱 통계')
            naive_stats['max_parsing_time_ms'] = self._extract_value(summary_section, r'\| 최대 파싱 시간 \(ms\) \| ([\d.]+) \|', after='Naive 파싱 통계')
            naive_stats['total_parsing_time_ms'] = self._extract_value(summary_section, r'\| 총 파싱 시간 \(ms\) \| ([\d.]+) \|', after='Naive 파싱 통계')
            naive_stats['avg_parsing_speed'] = self._extract_value(summary_section, r'\| 평균 파싱 속도 \(테이블/초\) \| ([\d.]+) \|', after='Naive 파싱 통계')
            
            summary_data['naive_parsing_stats'] = naive_stats
            
            # 속도 비교
            speed_comparison = {}
            speed_comparison['avg_speed_ratio'] = self._extract_value(summary_section, r'\| 평균 속도 비율 \(Naive/Labeled\) \| ([\d.]+)x \|')
            speed_comparison['avg_speed_improvement'] = self._extract_value(summary_section, r'\| 평균 속도 개선율 \| ([\d.-]+)% \|')
            speed_comparison['total_time_saved_ms'] = self._extract_value(summary_section, r'\| 총 절약 시간 \| ([\d.-]+) (ms|초) \|')
            
            summary_data['speed_comparison'] = speed_comparison
            summary_data['timestamp'] = timestamp
            
            data['summaries'].append(summary_data)
        
        return data
    
    def _extract_value(self, text: str, pattern: str, after: Optional[str] = None) -> Optional[float]:
        """텍스트에서 패턴으로 값 추출"""
        if after:
            # 특정 텍스트 이후의 내용만 검색
            idx = text.find(after)
            if idx != -1:
                text = text[idx:]
        
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            try:
                return float(val)
            except ValueError:
                return None
        return None
    
    def visualize_parsing_results(self, save_prefix: str = "parsing"):
        """파싱 결과 시각화"""
        results_file = self.parsing_dir / "experiment_results.txt"
        
        if not results_file.exists():
            print(f"결과 파일을 찾을 수 없습니다: {results_file}")
            return
        
        data = self.parse_text_results(results_file)
        
        if not data['individual_results']:
            print("파싱할 개별 결과가 없습니다.")
            return
        
        # 개별 결과에서 데이터 추출
        table_ids = [r['table_id'] for r in data['individual_results']]
        labeled_times = [r['labeled_parsing'].get('parsing_time_ms', 0) for r in data['individual_results']]
        naive_times = [r['naive_parsing'].get('parsing_time_ms', 0) for r in data['individual_results']]
        labeled_speeds = [r['labeled_parsing'].get('parsing_speed', 0) for r in data['individual_results']]
        naive_speeds = [r['naive_parsing'].get('parsing_speed', 0) for r in data['individual_results']]
        structure_richness = [r['comparison'].get('structure_richness', 0) for r in data['individual_results']]
        header_detection = [r['comparison'].get('header_detection', False) for r in data['individual_results']]
        
        # 1. 파싱 시간 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(table_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, labeled_times, width, label='레이블링 파싱', alpha=0.8)
        bars2 = ax.bar(x + width/2, naive_times, width, label='Naive 파싱', alpha=0.8)
        
        ax.set_xlabel('테이블 ID', fontsize=12)
        ax.set_ylabel('파싱 시간 (ms)', fontsize=12)
        ax.set_title('테이블별 파싱 시간 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(table_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_parsing_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 파싱 속도 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, labeled_speeds, width, label='레이블링 파싱', alpha=0.8)
        bars2 = ax.bar(x + width/2, naive_speeds, width, label='Naive 파싱', alpha=0.8)
        
        ax.set_xlabel('테이블 ID', fontsize=12)
        ax.set_ylabel('파싱 속도 (테이블/초)', fontsize=12)
        ax.set_title('테이블별 파싱 속도 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(table_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_parsing_speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 구조 풍부도 분포 (Bar Chart)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if hd else 'red' for hd in header_detection]
        bars = ax.bar(x, structure_richness, alpha=0.7, color=colors)
        
        ax.set_xlabel('테이블 ID', fontsize=12)
        ax.set_ylabel('구조 풍부도', fontsize=12)
        ax.set_title('테이블별 구조 풍부도 (녹색: 헤더 감지 성공, 빨강: 실패)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(table_ids, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='헤더 감지 성공'),
            Patch(facecolor='red', alpha=0.7, label='헤더 감지 실패')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_structure_richness.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 헤더 감지 성공/실패 비율 (Pie Chart)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        success_count = sum(header_detection)
        fail_count = len(header_detection) - success_count
        
        sizes = [success_count, fail_count]
        labels = ['헤더 감지 성공', '헤더 감지 실패']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12})
        ax.set_title('헤더 감지 성공률', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_header_detection_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 평균 성능 비교 (Summary Chart)
        if data['summaries']:
            latest_summary = data['summaries'][-1]  # 가장 최근 요약 사용
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 5-1. 평균 파싱 시간 비교
            ax1 = axes[0, 0]
            methods = ['레이블링\n파싱', 'Naive\n파싱']
            avg_times = [
                latest_summary['labeled_parsing_stats'].get('avg_parsing_time_ms') or 0,
                latest_summary['naive_parsing_stats'].get('avg_parsing_time_ms') or 0
            ]
            bars = ax1.bar(methods, avg_times, alpha=0.8, color=['#3498db', '#e67e22'])
            ax1.set_ylabel('평균 파싱 시간 (ms)', fontsize=11)
            ax1.set_title('평균 파싱 시간 비교', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)
            
            # 5-2. 평균 파싱 속도 비교
            ax2 = axes[0, 1]
            avg_speeds = [
                latest_summary['labeled_parsing_stats'].get('avg_parsing_speed') or 0,
                latest_summary['naive_parsing_stats'].get('avg_parsing_speed') or 0
            ]
            bars = ax2.bar(methods, avg_speeds, alpha=0.8, color=['#3498db', '#e67e22'])
            ax2.set_ylabel('평균 파싱 속도 (테이블/초)', fontsize=11)
            ax2.set_title('평균 파싱 속도 비교', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
            
            # 5-3. 구조 풍부도 및 헤더 감지율
            ax3 = axes[1, 0]
            metrics = ['구조\n풍부도', '헤더\n감지율']
            values = [
                latest_summary.get('avg_structure_richness') or 0,
                latest_summary.get('header_detection_rate') or 0
            ]
            bars = ax3.bar(metrics, values, alpha=0.8, color=['#9b59b6', '#f39c12'])
            ax3.set_ylabel('비율', fontsize=11)
            ax3.set_title('구조 풍부도 및 헤더 감지율', fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 1.1])
            ax3.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}', ha='center', va='bottom', fontsize=10)
            
            # 5-4. 셀 수 통계 비교
            ax4 = axes[1, 1]
            cell_metrics = ['평균 총\n셀 수', '평균 헤더\n셀 수', '평균 데이터\n셀 수']
            labeled_cells = [
                latest_summary['labeled_parsing_stats'].get('avg_total_cells') or 0,
                latest_summary['labeled_parsing_stats'].get('avg_header_cells') or 0,
                latest_summary['labeled_parsing_stats'].get('avg_data_cells') or 0
            ]
            naive_cells = [
                latest_summary['naive_parsing_stats'].get('avg_total_cells') or 0,
                0,  # Naive는 헤더/데이터 구분 없음
                0
            ]
            
            x_pos = np.arange(len(cell_metrics))
            width = 0.35
            bars1 = ax4.bar(x_pos - width/2, labeled_cells, width, label='레이블링', alpha=0.8, color='#3498db')
            bars2 = ax4.bar(x_pos + width/2, naive_cells, width, label='Naive', alpha=0.8, color='#e67e22')
            ax4.set_ylabel('셀 수', fontsize=11)
            ax4.set_title('평균 셀 수 통계', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cell_metrics, fontsize=9)
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            
            plt.suptitle('파싱 실험 요약 통계', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_summary_statistics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"파싱 결과 시각화 완료: {self.output_dir}")
        print(f"생성된 파일:")
        print(f"  - {save_prefix}_parsing_time_comparison.png")
        print(f"  - {save_prefix}_parsing_speed_comparison.png")
        print(f"  - {save_prefix}_structure_richness.png")
        print(f"  - {save_prefix}_header_detection_rate.png")
        print(f"  - {save_prefix}_summary_statistics.png")
    
    def parse_rag_text_results(self, filepath: Path) -> Dict[str, Any]:
        """RAG 텍스트 결과 파일 파싱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = {
            'individual_results': [],
            'summary': None
        }
        
        # 개별 쿼리 결과 파싱
        query_pattern = r'## RAG 평가 결과 - query_(\d+)\s*\*\*실험명\*\*: (\S+) \| \*\*평가 시간\*\*: (\S+)\s*\*\*쿼리\*\*: (.+?)\s*### 평가 메트릭 비교표\s*\| 메트릭 \| KG 기반 RAG \| Naive RAG \|\s*\|:-------\|:-----------\|:---------\|\s*\| Precision \| ([\d.]+) \| ([\d.]+) \|\s*\| Recall \| ([\d.]+) \| ([\d.]+) \|\s*\| F1 Score \| ([\d.]+) \| ([\d.]+) \|\s*\| MRR \| ([\d.]+) \| ([\d.]+) \|\s*\| 검색 시간 \(ms\) \| ([\d.]+) \| ([\d.]+) \|'
        
        query_matches = re.finditer(query_pattern, content, re.DOTALL)
        
        for match in query_matches:
            query_id = match.group(1)
            query = match.group(4)
            kg_precision = float(match.group(5))
            naive_precision = float(match.group(6))
            kg_recall = float(match.group(7))
            naive_recall = float(match.group(8))
            kg_f1 = float(match.group(9))
            naive_f1 = float(match.group(10))
            kg_mrr = float(match.group(11))
            naive_mrr = float(match.group(12))
            kg_time = float(match.group(13))
            naive_time = float(match.group(14))
            
            data['individual_results'].append({
                'query_id': query_id,
                'query': query,
                'kg_rag': {
                    'precision': kg_precision,
                    'recall': kg_recall,
                    'f1': kg_f1,
                    'mrr': kg_mrr,
                    'retrieve_time_ms': kg_time
                },
                'naive_rag': {
                    'precision': naive_precision,
                    'recall': naive_recall,
                    'f1': naive_f1,
                    'mrr': naive_mrr,
                    'retrieve_time_ms': naive_time
                }
            })
        
        # 전체 통계 파싱
        summary_pattern = r'## 전체 통계\s*### 평균 검색 성능 비교\s*\| 메트릭 \| KG 기반 RAG \| Naive RAG \|\s*\|:-------\|:-----------\|:---------\|\s*\| Precision \| ([\d.]+) \| ([\d.]+) \|\s*\| Recall \| ([\d.]+) \| ([\d.]+) \|\s*\| F1 \| ([\d.]+) \| ([\d.]+) \|\s*\| Mrr \| ([\d.]+) \| ([\d.]+) \|\s*### 시스템 구축 시간\s*\| 항목 \| 시간 \(ms\) \|\s*\|:-----\|:---------\|\s*\| KG RAG 구축 시간 \| ([\d.]+) \|\s*\| Naive RAG 구축 시간 \| ([\d.]+) \|\s*\| 구축 시간 비율 \(KG/Naive\) \| ([\d.]+)x \|\s*### 검색 시간 통계\s*\| 메트릭 \| KG 기반 RAG \| Naive RAG \|\s*\|:-------\|:-----------\|:---------\|\s*\| 평균 검색 시간 \(ms\) \| ([\d.]+) \| ([\d.]+) \|\s*\| 최소 검색 시간 \(ms\) \| ([\d.-]+) \| ([\d.-]+) \|\s*\| 최대 검색 시간 \(ms\) \| ([\d.]+) \| ([\d.]+) \|\s*\| 검색 시간 비율 \(KG/Naive\) \| ([\d.]+)x \|\s*### 성능 개선율\s*\| 메트릭 \| 개선율 \(%\) \|\s*\|:-------\|:----------\|\s*\| Precision \| ([\d.-]+) \|\s*\| Recall \| ([\d.-]+) \|\s*\| F1 \| ([\d.-]+) \|\s*\| Mrr \| ([\d.-]+) \|'
        
        summary_match = re.search(summary_pattern, content, re.DOTALL)
        if summary_match:
            data['summary'] = {
                'kg_rag_avg': {
                    'precision': float(summary_match.group(1)),
                    'recall': float(summary_match.group(3)),
                    'f1': float(summary_match.group(5)),
                    'mrr': float(summary_match.group(7))
                },
                'naive_rag_avg': {
                    'precision': float(summary_match.group(2)),
                    'recall': float(summary_match.group(4)),
                    'f1': float(summary_match.group(6)),
                    'mrr': float(summary_match.group(8))
                },
                'build_time': {
                    'kg_rag_build_time_ms': float(summary_match.group(9)),
                    'naive_rag_build_time_ms': float(summary_match.group(10)),
                    'build_time_ratio': float(summary_match.group(11))
                },
                'retrieve_time': {
                    'kg_rag_avg_ms': float(summary_match.group(12)),
                    'naive_rag_avg_ms': float(summary_match.group(13)),
                    'kg_rag_min_ms': float(summary_match.group(14)),
                    'naive_rag_min_ms': float(summary_match.group(15)),
                    'kg_rag_max_ms': float(summary_match.group(16)),
                    'naive_rag_max_ms': float(summary_match.group(17)),
                    'retrieve_time_ratio': float(summary_match.group(18))
                },
                'improvement': {
                    'precision': float(summary_match.group(19)),
                    'recall': float(summary_match.group(20)),
                    'f1': float(summary_match.group(21)),
                    'mrr': float(summary_match.group(22))
                }
            }
        
        return data
    
    def visualize_baseline_comparison(self, results: Dict[str, Any], save_prefix: str = "baseline_comparison"):
        """
        베이스라인 모델 비교 시각화
        
        Args:
            results: 실험 결과 딕셔너리
            save_prefix: 저장 파일 접두사
        """
        # 실험 1: 파싱 성능 비교
        if 'tatr_parsing' in results or 'sato_semantic' in results:
            self._visualize_parsing_baselines(results, save_prefix)
        
        # 실험 2: RAG 성능 비교
        if 'tablerag_baseline' in results:
            self._visualize_rag_baselines(results, save_prefix)
    
    def _visualize_parsing_baselines(self, results: Dict[str, Any], save_prefix: str):
        """파싱 베이스라인 비교 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 파싱 시간 비교
        methods = ['Labeled', 'Naive']
        times = []
        
        if 'labeled_parsing' in results and results['labeled_parsing']:
            labeled_times = [r['stats']['parsing_time_ms'] for r in results['labeled_parsing']]
            times.append(np.mean(labeled_times))
        else:
            times.append(0)
        
        if 'naive_parsing' in results and results['naive_parsing']:
            naive_times = [r['stats']['parsing_time_ms'] for r in results['naive_parsing']]
            times.append(np.mean(naive_times))
        else:
            times.append(0)
        
        if 'tatr_parsing' in results and results['tatr_parsing']:
            methods.append('TATR')
            tatr_times = [r['stats']['parsing_time_ms'] for r in results['tatr_parsing']]
            times.append(np.mean(tatr_times))
        
        ax = axes[0, 0]
        bars = ax.bar(methods, times, color=['#3498db', '#e67e22', '#2ecc71'][:len(methods)])
        ax.set_ylabel('평균 파싱 시간 (ms)', fontsize=12)
        ax.set_title('파싱 방법별 평균 시간 비교', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                   f'{time:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # 2. 셀 감지 개수 비교
        ax = axes[0, 1]
        cell_counts = []
        if 'labeled_parsing' in results and results['labeled_parsing']:
            cell_counts.append(np.mean([r['stats']['total_cells'] for r in results['labeled_parsing']]))
        else:
            cell_counts.append(0)
        
        if 'tatr_parsing' in results and results['tatr_parsing']:
            cell_counts.append(np.mean([r['stats']['cells_detected'] for r in results['tatr_parsing']]))
        
        if cell_counts:
            methods_cells = ['Labeled', 'TATR'][:len(cell_counts)]
            bars = ax.bar(methods_cells, cell_counts, color=['#3498db', '#2ecc71'][:len(cell_counts)])
            ax.set_ylabel('평균 셀 개수', fontsize=12)
            ax.set_title('셀 감지 개수 비교', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for bar, count in zip(bars, cell_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cell_counts)*0.01,
                       f'{count:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Sato 시맨틱 타입 검출 통계
        if 'sato_semantic' in results and results['sato_semantic']:
            ax = axes[1, 0]
            sato_results = results['sato_semantic']
            total_types = sum(len(r['stats']['types']) for r in sato_results)
            avg_types = total_types / len(sato_results) if sato_results else 0
            
            ax.bar(['Sato'], [avg_types], color='#9b59b6')
            ax.set_ylabel('평균 검출된 타입 수', fontsize=12)
            ax.set_title('Sato 시맨틱 타입 검출 통계', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.text(0, avg_types + max([avg_types])*0.01, f'{avg_types:.1f}',
                   ha='center', va='bottom', fontsize=10)
        
        # 4. 종합 성능 비교 (스코어)
        ax = axes[1, 1]
        scores = []
        score_labels = []
        
        if 'labeled_parsing' in results and results['labeled_parsing']:
            # 구조 풍부도 기반 스코어
            richness = results.get('summary', {}).get('avg_structure_richness', 0)
            scores.append(richness * 100)
            score_labels.append('Labeled\n(구조 풍부도)')
        
        if 'tatr_parsing' in results and results['tatr_parsing']:
            # TATR은 셀 감지율 기반
            if results['labeled_parsing']:
                labeled_cells = np.mean([r['stats']['total_cells'] for r in results['labeled_parsing']])
                tatr_cells = np.mean([r['stats']['cells_detected'] for r in results['tatr_parsing']])
                if labeled_cells > 0:
                    tatr_score = (tatr_cells / labeled_cells) * 100
                    scores.append(tatr_score)
                    score_labels.append('TATR\n(셀 감지율)')
        
        if scores:
            bars = ax.bar(score_labels, scores, color=['#3498db', '#2ecc71'][:len(scores)])
            ax.set_ylabel('성능 스코어', fontsize=12)
            ax.set_title('종합 성능 비교', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(scores) * 1.2 if scores else 100)
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.01,
                       f'{score:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_parsing.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  베이스라인 파싱 비교 시각화 저장: {save_prefix}_parsing.png")
    
    def _visualize_rag_baselines(self, results: Dict[str, Any], save_prefix: str):
        """RAG 베이스라인 비교 시각화"""
        if 'tablerag_baseline' not in results or not results['tablerag_baseline']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 메트릭 추출
        kg_precisions = [r['metrics']['precision'] for r in results.get('kg_rag', [])]
        naive_precisions = [r['metrics']['precision'] for r in results.get('naive_rag', [])]
        tablerag_precisions = [r['metrics']['precision'] for r in results['tablerag_baseline']]
        
        kg_recalls = [r['metrics']['recall'] for r in results.get('kg_rag', [])]
        naive_recalls = [r['metrics']['recall'] for r in results.get('naive_rag', [])]
        tablerag_recalls = [r['metrics']['recall'] for r in results['tablerag_baseline']]
        
        kg_f1s = [r['metrics']['f1'] for r in results.get('kg_rag', [])]
        naive_f1s = [r['metrics']['f1'] for r in results.get('naive_rag', [])]
        tablerag_f1s = [r['metrics']['f1'] for r in results['tablerag_baseline']]
        
        # 1. Precision 비교
        ax = axes[0, 0]
        methods = ['KG RAG', 'Naive RAG', 'TableRAG']
        precisions = [
            np.mean(kg_precisions) if kg_precisions else 0,
            np.mean(naive_precisions) if naive_precisions else 0,
            np.mean(tablerag_precisions) if tablerag_precisions else 0
        ]
        bars = ax.bar(methods, precisions, color=['#3498db', '#e67e22', '#2ecc71'])
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('RAG 방법별 Precision 비교', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        for bar, prec in zip(bars, precisions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prec:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Recall 비교
        ax = axes[0, 1]
        recalls = [
            np.mean(kg_recalls) if kg_recalls else 0,
            np.mean(naive_recalls) if naive_recalls else 0,
            np.mean(tablerag_recalls) if tablerag_recalls else 0
        ]
        bars = ax.bar(methods, recalls, color=['#3498db', '#e67e22', '#2ecc71'])
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('RAG 방법별 Recall 비교', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        for bar, rec in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rec:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. F1 Score 비교
        ax = axes[1, 0]
        f1s = [
            np.mean(kg_f1s) if kg_f1s else 0,
            np.mean(naive_f1s) if naive_f1s else 0,
            np.mean(tablerag_f1s) if tablerag_f1s else 0
        ]
        bars = ax.bar(methods, f1s, color=['#3498db', '#e67e22', '#2ecc71'])
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('RAG 방법별 F1 Score 비교', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 종합 비교 (Radar Chart 스타일)
        ax = axes[1, 1]
        metrics = ['Precision', 'Recall', 'F1']
        x = np.arange(len(metrics))
        width = 0.25
        
        kg_values = [precisions[0], recalls[0], f1s[0]]
        naive_values = [precisions[1], recalls[1], f1s[1]]
        tablerag_values = [precisions[2], recalls[2], f1s[2]]
        
        ax.bar(x - width, kg_values, width, label='KG RAG', color='#3498db', alpha=0.8)
        ax.bar(x, naive_values, width, label='Naive RAG', color='#e67e22', alpha=0.8)
        ax.bar(x + width, tablerag_values, width, label='TableRAG', color='#2ecc71', alpha=0.8)
        
        ax.set_ylabel('스코어', fontsize=12)
        ax.set_title('메트릭별 종합 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_rag.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  베이스라인 RAG 비교 시각화 저장: {save_prefix}_rag.png")
    
    def visualize_rag_results(self, save_prefix: str = "rag"):
        """RAG 결과 시각화"""
        results_file = self.rag_dir / "experiment_results.txt"
        
        if not results_file.exists():
            print(f"결과 파일을 찾을 수 없습니다: {results_file}")
            return
        
        data = self.parse_rag_text_results(results_file)
        
        if not data['individual_results']:
            print("파싱할 개별 결과가 없습니다.")
            return
        
        # 개별 결과에서 데이터 추출
        query_ids = [f"Q{r['query_id']}" for r in data['individual_results']]
        kg_precisions = [r['kg_rag']['precision'] for r in data['individual_results']]
        naive_precisions = [r['naive_rag']['precision'] for r in data['individual_results']]
        kg_recalls = [r['kg_rag']['recall'] for r in data['individual_results']]
        naive_recalls = [r['naive_rag']['recall'] for r in data['individual_results']]
        kg_f1s = [r['kg_rag']['f1'] for r in data['individual_results']]
        naive_f1s = [r['naive_rag']['f1'] for r in data['individual_results']]
        kg_mrrs = [r['kg_rag']['mrr'] for r in data['individual_results']]
        naive_mrrs = [r['naive_rag']['mrr'] for r in data['individual_results']]
        kg_times = [r['kg_rag']['retrieve_time_ms'] for r in data['individual_results']]
        naive_times = [r['naive_rag']['retrieve_time_ms'] for r in data['individual_results']]
        
        # 1. Precision 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(query_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, kg_precisions, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, naive_precisions, width, label='Naive RAG', alpha=0.8, color='#e67e22')
        
        ax.set_xlabel('쿼리 ID', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('쿼리별 Precision 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_precision_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Recall 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, kg_recalls, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, naive_recalls, width, label='Naive RAG', alpha=0.8, color='#e67e22')
        
        ax.set_xlabel('쿼리 ID', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('쿼리별 Recall 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_recall_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. F1 Score 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, kg_f1s, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, naive_f1s, width, label='Naive RAG', alpha=0.8, color='#e67e22')
        
        ax.set_xlabel('쿼리 ID', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('쿼리별 F1 Score 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. MRR 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, kg_mrrs, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, naive_mrrs, width, label='Naive RAG', alpha=0.8, color='#e67e22')
        
        ax.set_xlabel('쿼리 ID', fontsize=12)
        ax.set_ylabel('MRR', fontsize=12)
        ax.set_title('쿼리별 MRR 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_mrr_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 검색 시간 비교 (Bar Chart)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, kg_times, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, naive_times, width, label='Naive RAG', alpha=0.8, color='#e67e22')
        
        ax.set_xlabel('쿼리 ID', fontsize=12)
        ax.set_ylabel('검색 시간 (ms)', fontsize=12)
        ax.set_title('쿼리별 검색 시간 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_prefix}_retrieve_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 평균 성능 비교 (Summary Chart)
        if data['summary']:
            summary = data['summary']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 6-1. 평균 검색 성능 메트릭 비교
            ax1 = axes[0, 0]
            metrics = ['Precision', 'Recall', 'F1', 'MRR']
            kg_values = [
                summary['kg_rag_avg']['precision'],
                summary['kg_rag_avg']['recall'],
                summary['kg_rag_avg']['f1'],
                summary['kg_rag_avg']['mrr']
            ]
            naive_values = [
                summary['naive_rag_avg']['precision'],
                summary['naive_rag_avg']['recall'],
                summary['naive_rag_avg']['f1'],
                summary['naive_rag_avg']['mrr']
            ]
            
            x_pos = np.arange(len(metrics))
            bars1 = ax1.bar(x_pos - width/2, kg_values, width, label='KG 기반 RAG', alpha=0.8, color='#3498db')
            bars2 = ax1.bar(x_pos + width/2, naive_values, width, label='Naive RAG', alpha=0.8, color='#e67e22')
            
            ax1.set_ylabel('점수', fontsize=11)
            ax1.set_title('평균 검색 성능 비교', fontsize=12, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(metrics, fontsize=10)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
            
            # 6-2. 성능 개선율
            ax2 = axes[0, 1]
            improvement_values = [
                summary['improvement']['precision'],
                summary['improvement']['recall'],
                summary['improvement']['f1'],
                summary['improvement']['mrr']
            ]
            colors = ['green' if v >= 0 else 'red' for v in improvement_values]
            bars = ax2.bar(metrics, improvement_values, alpha=0.8, color=colors)
            ax2.set_ylabel('개선율 (%)', fontsize=11)
            ax2.set_title('성능 개선율 (KG vs Naive)', fontsize=12, fontweight='bold')
            ax2.set_xticklabels(metrics, fontsize=10)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            # 6-3. 시스템 구축 시간 비교
            ax3 = axes[1, 0]
            build_times = [
                summary['build_time']['kg_rag_build_time_ms'],
                summary['build_time']['naive_rag_build_time_ms']
            ]
            methods = ['KG 기반\nRAG', 'Naive\nRAG']
            bars = ax3.bar(methods, build_times, alpha=0.8, color=['#3498db', '#e67e22'])
            ax3.set_ylabel('구축 시간 (ms)', fontsize=11)
            ax3.set_title('시스템 구축 시간 비교', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}ms', ha='center', va='bottom', fontsize=10)
            
            # 비율 표시
            ratio = summary['build_time']['build_time_ratio']
            ax3.text(0.5, 0.95, f'비율: {ratio:.2f}x', 
                    transform=ax3.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
            
            # 6-4. 평균 검색 시간 비교
            ax4 = axes[1, 1]
            retrieve_times = [
                summary['retrieve_time']['kg_rag_avg_ms'],
                summary['retrieve_time']['naive_rag_avg_ms']
            ]
            bars = ax4.bar(methods, retrieve_times, alpha=0.8, color=['#3498db', '#e67e22'])
            ax4.set_ylabel('평균 검색 시간 (ms)', fontsize=11)
            ax4.set_title('평균 검색 시간 비교', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)
            
            # 비율 표시
            ratio = summary['retrieve_time']['retrieve_time_ratio']
            ax4.text(0.5, 0.95, f'비율: {ratio:.2f}x', 
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
            
            plt.suptitle('RAG 실험 요약 통계', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_summary_statistics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"RAG 결과 시각화 완료: {self.output_dir}")
        print(f"생성된 파일:")
        print(f"  - {save_prefix}_precision_comparison.png")
        print(f"  - {save_prefix}_recall_comparison.png")
        print(f"  - {save_prefix}_f1_comparison.png")
        print(f"  - {save_prefix}_mrr_comparison.png")
        print(f"  - {save_prefix}_retrieve_time_comparison.png")
        print(f"  - {save_prefix}_summary_statistics.png")


def main():
    """메인 실행 함수"""
    visualizer = ResultVisualizer(results_dir="results")
    
    print("=== 실험 결과 시각화 시작 ===")
    
    # 파싱 결과 시각화
    print("\n1. 파싱 결과 시각화 중...")
    visualizer.visualize_parsing_results()
    
    # RAG 결과 시각화 (결과가 있는 경우)
    print("\n2. RAG 결과 시각화 중...")
    visualizer.visualize_rag_results()
    
    print("\n=== 시각화 완료 ===")


if __name__ == "__main__":
    main()

