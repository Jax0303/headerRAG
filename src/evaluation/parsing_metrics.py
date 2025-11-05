"""
표 파싱 평가 메트릭
- TEDS (Tree Edit Distance-based Similarity)
- GriTS (Grid Table Similarity)
- 헤더 감지 정확도
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
import warnings

try:
    from zss import Node, simple_distance
    ZSS_AVAILABLE = True
except ImportError:
    ZSS_AVAILABLE = False
    Node = None  # 타입 힌트용
    simple_distance = None
    warnings.warn("zss 라이브러리가 설치되지 않았습니다. TEDS 메트릭을 사용하려면 설치해주세요: pip install zss")


class ParsingMetrics:
    """표 파싱 평가 메트릭 클래스"""
    
    def __init__(self):
        self.zss_available = ZSS_AVAILABLE
    
    def calculate_teds(self, 
                      predicted_html: str, 
                      ground_truth_html: str) -> float:
        """
        TEDS (Tree Edit Distance-based Similarity) 계산
        
        Args:
            predicted_html: 예측된 HTML 테이블
            ground_truth_html: 정답 HTML 테이블
            
        Returns:
            TEDS 점수 (0-1, 1이 완벽 일치)
        """
        if not self.zss_available:
            warnings.warn("zss 라이브러리가 없어 TEDS를 계산할 수 없습니다.")
            return 0.0
        
        try:
            # HTML을 트리로 변환
            pred_tree = self._html_to_tree(predicted_html)
            gt_tree = self._html_to_tree(ground_truth_html)
            
            # 트리 편집 거리 계산
            if simple_distance is None:
                return 0.0
            edit_distance = simple_distance(pred_tree, gt_tree)
            
            # 정규화: 최대 편집 거리는 두 트리의 총 노드 수 합
            max_distance = self._count_nodes(pred_tree) + self._count_nodes(gt_tree)
            
            if max_distance == 0:
                return 1.0
            
            # TEDS = 1 - (edit_distance / max_distance)
            teds_score = 1.0 - (edit_distance / max_distance)
            return max(0.0, min(1.0, teds_score))
            
        except Exception as e:
            warnings.warn(f"TEDS 계산 중 오류 발생: {e}")
            return 0.0
    
    def _html_to_tree(self, html: str) -> Optional[Any]:
        """HTML을 zss Node 트리로 변환"""
        if not self.zss_available:
            return None
        try:
            root = ET.fromstring(html)
            return self._element_to_node(root)
        except Exception as e:
            warnings.warn(f"HTML 파싱 오류: {e}")
            return None
    
    def _element_to_node(self, element: ET.Element) -> Any:
        """XML Element를 zss Node로 변환"""
        if not self.zss_available or Node is None:
            return None
        label = element.tag
        node = Node(label)
        
        # 속성도 레이블에 포함
        if element.attrib:
            attrs = ','.join(f"{k}={v}" for k, v in element.attrib.items())
            node.label = f"{label}[{attrs}]"
        
        # 텍스트 내용
        if element.text and element.text.strip():
            node.label += f":{element.text.strip()[:20]}"  # 처음 20자만
        
        # 자식 노드 재귀적으로 추가
        for child in element:
            child_node = self._element_to_node(child)
            node.addkid(child_node)
        
        return node
    
    def _count_nodes(self, node: Any) -> int:
        """트리의 노드 수 계산"""
        if node is None:
            return 0
        if not self.zss_available:
            return 0
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def calculate_grits(self,
                       predicted_table: Dict[str, Any],
                       ground_truth_table: Dict[str, Any]) -> Dict[str, float]:
        """
        GriTS (Grid Table Similarity) 계산
        
        Args:
            predicted_table: 예측된 테이블 구조
                - cells: List[Dict] with keys: 'row', 'col', 'rowspan', 'colspan', 'text'
            ground_truth_table: 정답 테이블 구조
                - cells: List[Dict] with keys: 'row', 'col', 'rowspan', 'colspan', 'text'
        
        Returns:
            GriTS 메트릭 딕셔너리:
                - grits_content: 셀 텍스트 편집 거리
                - grits_topology: rowspan/colspan 일치도
                - grits_location: 셀 공간 좌표 IoU
                - grits_overall: 전체 GriTS 점수
        """
        pred_cells = predicted_table.get('cells', [])
        gt_cells = ground_truth_table.get('cells', [])
        
        if not pred_cells and not gt_cells:
            return {
                'grits_content': 1.0,
                'grits_topology': 1.0,
                'grits_location': 1.0,
                'grits_overall': 1.0
            }
        
        # 2D 그리드로 변환
        pred_grid = self._cells_to_grid(pred_cells)
        gt_grid = self._cells_to_grid(gt_cells)
        
        # GriTS-Content: 셀 텍스트 유사도
        grits_content = self._calculate_content_similarity(pred_grid, gt_grid)
        
        # GriTS-Topology: rowspan/colspan 일치도
        grits_topology = self._calculate_topology_similarity(pred_cells, gt_cells)
        
        # GriTS-Location: 셀 공간 좌표 IoU
        grits_location = self._calculate_location_iou(pred_cells, gt_cells)
        
        # 전체 GriTS 점수 (가중 평균)
        grits_overall = (
            0.4 * grits_content +
            0.3 * grits_topology +
            0.3 * grits_location
        )
        
        return {
            'grits_content': grits_content,
            'grits_topology': grits_topology,
            'grits_location': grits_location,
            'grits_overall': grits_overall
        }
    
    def _cells_to_grid(self, cells: List[Dict[str, Any]]) -> Dict[Tuple[int, int], str]:
        """셀 리스트를 2D 그리드로 변환"""
        grid = {}
        for cell in cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            text = str(cell.get('text', '')).strip()
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            
            # 병합 셀의 모든 위치에 텍스트 저장
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    grid[(r, c)] = text
        
        return grid
    
    def _calculate_content_similarity(self,
                                     pred_grid: Dict[Tuple[int, int], str],
                                     gt_grid: Dict[Tuple[int, int], str]) -> float:
        """GriTS-Content: 셀 텍스트 유사도 계산"""
        all_positions = set(pred_grid.keys()) | set(gt_grid.keys())
        
        if not all_positions:
            return 1.0
        
        total_similarity = 0.0
        for pos in all_positions:
            pred_text = pred_grid.get(pos, '')
            gt_text = gt_grid.get(pos, '')
            
            # 문자열 유사도 계산 (SequenceMatcher 사용)
            similarity = SequenceMatcher(None, pred_text, gt_text).ratio()
            total_similarity += similarity
        
        return total_similarity / len(all_positions) if all_positions else 0.0
    
    def _calculate_topology_similarity(self,
                                      pred_cells: List[Dict[str, Any]],
                                      gt_cells: List[Dict[str, Any]]) -> float:
        """GriTS-Topology: rowspan/colspan 일치도 계산"""
        if not pred_cells and not gt_cells:
            return 1.0
        
        # 셀을 (row, col, rowspan, colspan) 튜플로 변환
        pred_topology = {
            (c.get('row', 0), c.get('col', 0)): (
                c.get('rowspan', 1),
                c.get('colspan', 1)
            )
            for c in pred_cells
        }
        
        gt_topology = {
            (c.get('row', 0), c.get('col', 0)): (
                c.get('rowspan', 1),
                c.get('colspan', 1)
            )
            for c in gt_cells
        }
        
        all_positions = set(pred_topology.keys()) | set(gt_topology.keys())
        
        if not all_positions:
            return 1.0
        
        matches = 0
        for pos in all_positions:
            pred_val = pred_topology.get(pos, (1, 1))
            gt_val = gt_topology.get(pos, (1, 1))
            
            if pred_val == gt_val:
                matches += 1
        
        return matches / len(all_positions) if all_positions else 0.0
    
    def _calculate_location_iou(self,
                                pred_cells: List[Dict[str, Any]],
                                gt_cells: List[Dict[str, Any]]) -> float:
        """GriTS-Location: 셀 공간 좌표 IoU 계산"""
        if not pred_cells and not gt_cells:
            return 1.0
        
        # 각 셀의 바운딩 박스 영역 계산
        pred_boxes = []
        gt_boxes = []
        
        for cell in pred_cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            pred_boxes.append((row, col, row + rowspan, col + colspan))
        
        for cell in gt_cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            gt_boxes.append((row, col, row + rowspan, col + colspan))
        
        # IoU 계산 (각 셀 쌍의 최대 IoU를 사용)
        total_iou = 0.0
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                
                iou = self._box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx >= 0:
                matched_gt.add(best_idx)
            
            total_iou += best_iou
        
        # 정규화
        avg_iou = total_iou / len(pred_boxes) if pred_boxes else 0.0
        
        # 역방향도 계산 (재현율)
        total_iou_reverse = 0.0
        matched_pred = set()
        
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for idx, pred_box in enumerate(pred_boxes):
                if idx in matched_pred:
                    continue
                
                iou = self._box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx >= 0:
                matched_pred.add(best_idx)
            
            total_iou_reverse += best_iou
        
        avg_iou_reverse = total_iou_reverse / len(gt_boxes) if gt_boxes else 0.0
        
        # F1 스타일 평균
        if avg_iou + avg_iou_reverse == 0:
            return 0.0
        
        return 2 * avg_iou * avg_iou_reverse / (avg_iou + avg_iou_reverse)
    
    def _box_iou(self, box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int]) -> float:
        """두 바운딩 박스의 IoU 계산"""
        r1_min, c1_min, r1_max, c1_max = box1
        r2_min, c2_min, r2_max, c2_max = box2
        
        # 교집합 영역
        inter_r_min = max(r1_min, r2_min)
        inter_c_min = max(c1_min, c2_min)
        inter_r_max = min(r1_max, r2_max)
        inter_c_max = min(c1_max, c2_max)
        
        if inter_r_max <= inter_r_min or inter_c_max <= inter_c_min:
            return 0.0
        
        inter_area = (inter_r_max - inter_r_min) * (inter_c_max - inter_c_min)
        
        # 합집합 영역
        box1_area = (r1_max - r1_min) * (c1_max - c1_min)
        box2_area = (r2_max - r2_min) * (c2_max - c2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def calculate_header_metrics(self,
                                predicted_structure: Dict[str, Any],
                                ground_truth_structure: Dict[str, Any]) -> Dict[str, float]:
        """
        헤더 감지 정확도 계산
        
        Args:
            predicted_structure: 예측된 구조
                - headers: List[Dict] with keys: 'row', 'col', 'text', 'is_header'
            ground_truth_structure: 정답 구조
                - headers: List[Dict] with keys: 'row', 'col', 'text', 'is_header'
        
        Returns:
            헤더 메트릭 딕셔너리:
                - header_precision
                - header_recall
                - header_f1
                - merged_cell_accuracy: 병합 셀 처리 정확도
        """
        pred_headers = predicted_structure.get('headers', [])
        gt_headers = ground_truth_structure.get('headers', [])
        
        # 헤더 셀을 위치로 변환
        pred_header_positions = {
            (h.get('row', 0), h.get('col', 0))
            for h in pred_headers if h.get('is_header', False)
        }
        
        gt_header_positions = {
            (h.get('row', 0), h.get('col', 0))
            for h in gt_headers if h.get('is_header', False)
        }
        
        # Precision, Recall, F1
        if not pred_header_positions and not gt_header_positions:
            precision = recall = f1 = 1.0
        elif not pred_header_positions:
            precision = recall = f1 = 0.0
        elif not gt_header_positions:
            precision = recall = f1 = 0.0
        else:
            true_positives = len(pred_header_positions & gt_header_positions)
            precision = true_positives / len(pred_header_positions) if pred_header_positions else 0.0
            recall = true_positives / len(gt_header_positions) if gt_header_positions else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 병합 셀 처리 정확도
        pred_merged = {
            (h.get('row', 0), h.get('col', 0))
            for h in pred_headers 
            if h.get('is_header', False) and (h.get('rowspan', 1) > 1 or h.get('colspan', 1) > 1)
        }
        
        gt_merged = {
            (h.get('row', 0), h.get('col', 0))
            for h in gt_headers 
            if h.get('is_header', False) and (h.get('rowspan', 1) > 1 or h.get('colspan', 1) > 1)
        }
        
        if not pred_merged and not gt_merged:
            merged_accuracy = 1.0
        elif not pred_merged or not gt_merged:
            merged_accuracy = 0.0
        else:
            merged_matches = len(pred_merged & gt_merged)
            merged_accuracy = merged_matches / len(gt_merged) if gt_merged else 0.0
        
        return {
            'header_precision': precision,
            'header_recall': recall,
            'header_f1': f1,
            'merged_cell_accuracy': merged_accuracy
        }
    
    def evaluate_parsing(self,
                        predicted_table: Dict[str, Any],
                        ground_truth_table: Dict[str, Any],
                        predicted_html: Optional[str] = None,
                        ground_truth_html: Optional[str] = None) -> Dict[str, float]:
        """
        종합 파싱 평가
        
        Args:
            predicted_table: 예측된 테이블 구조
            ground_truth_table: 정답 테이블 구조
            predicted_html: 예측된 HTML (TEDS용, 선택적)
            ground_truth_html: 정답 HTML (TEDS용, 선택적)
        
        Returns:
            모든 파싱 메트릭을 포함한 딕셔너리
        """
        metrics = {}
        
        # GriTS 메트릭
        grits_metrics = self.calculate_grits(predicted_table, ground_truth_table)
        metrics.update(grits_metrics)
        
        # 헤더 메트릭
        header_metrics = self.calculate_header_metrics(predicted_table, ground_truth_table)
        metrics.update(header_metrics)
        
        # TEDS 메트릭 (HTML이 제공된 경우)
        if predicted_html and ground_truth_html:
            teds_score = self.calculate_teds(predicted_html, ground_truth_html)
            metrics['teds'] = teds_score
        
        return metrics

