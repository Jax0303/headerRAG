"""
표 복잡도 메트릭 정의 및 계산
- 구조적 복잡도 지표
- 시각적 복잡도 지표
- 복잡도 등급 분류
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class ComplexityMetrics:
    """표 복잡도 메트릭 계산 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_structural_complexity(self, table: Dict[str, Any]) -> Dict[str, float]:
        """
        구조적 복잡도 지표 계산
        
        Args:
            table: 테이블 구조 딕셔너리
                - cells: List[Dict] with keys: 'row', 'col', 'rowspan', 'colspan', 'text'
                - headers: List[Dict] (선택적)
        
        Returns:
            구조적 복잡도 메트릭 딕셔너리:
                - merged_cell_ratio: 병합 셀 비율
                - header_depth: 헤더 계층 깊이
                - nested_subtable_count: 중첩된 서브테이블 수
                - row_col_asymmetry: 행/열 비대칭성
                - empty_cell_ratio: 빈 셀 비율
                - structural_complexity_score: 종합 구조적 복잡도 점수
        """
        cells = table.get('cells', [])
        headers = table.get('headers', [])
        
        if not cells:
            return {
                'merged_cell_ratio': 0.0,
                'header_depth': 0.0,
                'nested_subtable_count': 0.0,
                'row_col_asymmetry': 0.0,
                'empty_cell_ratio': 0.0,
                'structural_complexity_score': 0.0
            }
        
        total_cells = len(cells)
        
        # 1. 병합 셀 비율
        merged_cells = [
            c for c in cells 
            if c.get('rowspan', 1) > 1 or c.get('colspan', 1) > 1
        ]
        merged_cell_ratio = len(merged_cells) / total_cells if total_cells > 0 else 0.0
        
        # 2. 헤더 계층 깊이
        header_depth = self._calculate_header_depth(headers, cells)
        
        # 3. 중첩된 서브테이블 수
        nested_subtable_count = self._count_nested_subtables(cells)
        
        # 4. 행/열 비대칭성
        max_row = max((c.get('row', 0) + c.get('rowspan', 1) - 1) for c in cells) if cells else 0
        max_col = max((c.get('col', 0) + c.get('colspan', 1) - 1) for c in cells) if cells else 0
        
        if max_row == 0 and max_col == 0:
            row_col_asymmetry = 0.0
        else:
            row_col_asymmetry = abs(max_row - max_col) / max(max_row, max_col, 1)
        
        # 5. 빈 셀 비율
        empty_cells = [c for c in cells if not str(c.get('text', '')).strip()]
        empty_cell_ratio = len(empty_cells) / total_cells if total_cells > 0 else 0.0
        
        # 6. 종합 구조적 복잡도 점수 (0-1)
        structural_complexity_score = (
            0.4 * merged_cell_ratio +
            0.3 * min(header_depth / 5.0, 1.0) +  # 정규화 (최대 5레벨)
            0.2 * min(nested_subtable_count / 3.0, 1.0) +  # 정규화 (최대 3개)
            0.05 * row_col_asymmetry +
            0.05 * empty_cell_ratio
        )
        
        return {
            'merged_cell_ratio': merged_cell_ratio,
            'header_depth': header_depth,
            'nested_subtable_count': nested_subtable_count,
            'row_col_asymmetry': row_col_asymmetry,
            'empty_cell_ratio': empty_cell_ratio,
            'structural_complexity_score': min(1.0, structural_complexity_score)
        }
    
    def _calculate_header_depth(self, headers: List[Dict[str, Any]], 
                               cells: List[Dict[str, Any]]) -> float:
        """헤더 계층 깊이 계산"""
        if not headers:
            return 0.0
        
        # 헤더 행들의 깊이 계산
        header_rows = {}
        for header in headers:
            if header.get('is_header', False):
                row = header.get('row', 0)
                if row not in header_rows:
                    header_rows[row] = []
                header_rows[row].append(header)
        
        if not header_rows:
            return 0.0
        
        # 헤더 행의 최대 깊이
        max_depth = len(header_rows)
        
        # 중첩 헤더 확인 (같은 열에 여러 헤더가 있는지)
        nested_depth = 0
        for row_idx in sorted(header_rows.keys()):
            cols_in_row = {h.get('col', 0) for h in header_rows[row_idx]}
            # 이전 행과의 중첩 확인
            for prev_row_idx in sorted(header_rows.keys()):
                if prev_row_idx >= row_idx:
                    break
                prev_cols = {h.get('col', 0) for h in header_rows[prev_row_idx]}
                if cols_in_row & prev_cols:
                    nested_depth += 1
                    break
        
        return max(max_depth, nested_depth)
    
    def _count_nested_subtables(self, cells: List[Dict[str, Any]]) -> int:
        """중첩된 서브테이블 수 계산"""
        if not cells:
            return 0
        
        # 그리드로 변환하여 분리된 영역 찾기
        grid = {}
        for cell in cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    grid[(r, c)] = True
        
        # 연결된 컴포넌트 찾기 (BFS)
        visited = set()
        components = []
        
        for pos in grid.keys():
            if pos in visited:
                continue
            
            # 새로운 컴포넌트 발견
            component = set()
            queue = [pos]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                component.add(current)
                
                # 인접 셀 추가
                r, c = current
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in grid and neighbor not in visited:
                        queue.append(neighbor)
            
            if len(component) > 1:  # 최소 2개 셀 이상인 컴포넌트만 카운트
                components.append(component)
        
        # 메인 테이블 외의 컴포넌트 수
        return max(0, len(components) - 1)
    
    def calculate_visual_complexity(self, table: Dict[str, Any]) -> Dict[str, float]:
        """
        시각적 복잡도 지표 계산 (WTW 데이터셋 기반)
        
        Args:
            table: 테이블 구조 딕셔너리
                - visual_info: Dict (선택적) with keys:
                    - angles: List[float] (기울기 각도)
                    - overlaps: List[bool] (텍스트 겹침)
                    - borders: Dict (테두리 정보)
        
        Returns:
            시각적 복잡도 메트릭 딕셔너리:
                - average_angle: 평균 기울기 각도
                - overlap_ratio: 텍스트 겹침 비율
                - border_completeness: 테두리 완전성
                - visual_complexity_score: 종합 시각적 복잡도 점수
        """
        visual_info = table.get('visual_info', {})
        
        # 1. 평균 기울기 각도
        angles = visual_info.get('angles', [])
        if angles:
            average_angle = np.mean([abs(a) for a in angles])
        else:
            average_angle = 0.0
        
        # 2. 텍스트 겹침 비율
        overlaps = visual_info.get('overlaps', [])
        if overlaps:
            overlap_ratio = sum(overlaps) / len(overlaps) if overlaps else 0.0
        else:
            overlap_ratio = 0.0
        
        # 3. 테두리 완전성
        borders = visual_info.get('borders', {})
        if borders:
            expected_borders = borders.get('expected', 0)
            actual_borders = borders.get('actual', 0)
            border_completeness = actual_borders / expected_borders if expected_borders > 0 else 1.0
        else:
            border_completeness = 1.0  # 정보가 없으면 완전하다고 가정
        
        # 4. 종합 시각적 복잡도 점수
        visual_complexity_score = (
            0.3 * min(average_angle / 45.0, 1.0) +  # 정규화 (최대 45도)
            0.4 * overlap_ratio +
            0.3 * (1.0 - border_completeness)  # 테두리가 불완전할수록 복잡
        )
        
        return {
            'average_angle': average_angle,
            'overlap_ratio': overlap_ratio,
            'border_completeness': border_completeness,
            'visual_complexity_score': min(1.0, visual_complexity_score)
        }
    
    def classify_complexity_level(self, 
                                 structural_score: float,
                                 visual_score: Optional[float] = None) -> str:
        """
        복잡도 등급 분류
        
        Args:
            structural_score: 구조적 복잡도 점수 (0-1)
            visual_score: 시각적 복잡도 점수 (0-1, 선택적)
        
        Returns:
            복잡도 등급: 'low', 'medium', 'high'
        """
        # 시각적 복잡도가 있으면 가중 평균
        if visual_score is not None:
            combined_score = 0.7 * structural_score + 0.3 * visual_score
        else:
            combined_score = structural_score
        
        # 기준 조정: 실제 데이터에 맞게 낮춤
        if combined_score < 0.2:
            return 'low'
        elif combined_score < 0.5:
            return 'medium'
        else:
            return 'high'
    
    def calculate_complexity(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        종합 복잡도 계산
        
        Args:
            table: 테이블 구조 딕셔너리
        
        Returns:
            모든 복잡도 메트릭을 포함한 딕셔너리
        """
        # 구조적 복잡도
        structural_metrics = self.calculate_structural_complexity(table)
        
        # 시각적 복잡도
        visual_metrics = self.calculate_visual_complexity(table)
        
        # 복잡도 등급 분류
        complexity_level = self.classify_complexity_level(
            structural_metrics['structural_complexity_score'],
            visual_metrics.get('visual_complexity_score')
        )
        
        return {
            **structural_metrics,
            **visual_metrics,
            'complexity_level': complexity_level,
            'overall_complexity_score': (
                0.7 * structural_metrics['structural_complexity_score'] +
                0.3 * visual_metrics.get('visual_complexity_score', 0.0)
            )
        }

