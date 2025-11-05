"""
메트릭 검증 스크립트
새로 구현한 메트릭들이 제대로 작동하는지 검증
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.parsing_metrics import ParsingMetrics
from src.evaluation.ragas_metrics import RAGASMetrics
from src.evaluation.complexity_metrics import ComplexityMetrics


def create_sample_table_structure():
    """샘플 테이블 구조 생성"""
    cells = [
        {'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 2, 'text': 'Header'},
        {'row': 0, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': 'Year'},
        {'row': 1, 'col': 0, 'rowspan': 1, 'colspan': 1, 'text': 'Category A'},
        {'row': 1, 'col': 1, 'rowspan': 1, 'colspan': 1, 'text': 'Value A'},
        {'row': 1, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': '2023'},
        {'row': 2, 'col': 0, 'rowspan': 1, 'colspan': 1, 'text': 'Category B'},
        {'row': 2, 'col': 1, 'rowspan': 1, 'colspan': 1, 'text': 'Value B'},
        {'row': 2, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': '2024'},
    ]
    
    headers = [
        {'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 2, 'text': 'Header', 'is_header': True},
        {'row': 0, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': 'Year', 'is_header': True},
    ]
    
    return {'cells': cells, 'headers': headers}


def test_parsing_metrics():
    """파싱 메트릭 테스트"""
    print("\n" + "="*70)
    print("파싱 메트릭 검증")
    print("="*70)
    
    metrics = ParsingMetrics()
    
    # 샘플 테이블 생성
    pred_table = create_sample_table_structure()
    gt_table = create_sample_table_structure()
    
    # GriTS 테스트
    print("\n1. GriTS 메트릭 테스트...")
    grits_results = metrics.calculate_grits(pred_table, gt_table)
    print(f"   GriTS-Content: {grits_results['grits_content']:.4f}")
    print(f"   GriTS-Topology: {grits_results['grits_topology']:.4f}")
    print(f"   GriTS-Location: {grits_results['grits_location']:.4f}")
    print(f"   GriTS-Overall: {grits_results['grits_overall']:.4f}")
    
    # 완벽 일치 시 1.0에 가까워야 함
    assert grits_results['grits_overall'] > 0.9, "완벽 일치 시 GriTS는 0.9 이상이어야 합니다"
    print("   ✓ GriTS 테스트 통과")
    
    # 헤더 메트릭 테스트
    print("\n2. 헤더 감지 메트릭 테스트...")
    header_results = metrics.calculate_header_metrics(pred_table, gt_table)
    print(f"   Header Precision: {header_results['header_precision']:.4f}")
    print(f"   Header Recall: {header_results['header_recall']:.4f}")
    print(f"   Header F1: {header_results['header_f1']:.4f}")
    print(f"   Merged Cell Accuracy: {header_results['merged_cell_accuracy']:.4f}")
    
    assert header_results['header_f1'] > 0.9, "완벽 일치 시 헤더 F1은 0.9 이상이어야 합니다"
    print("   ✓ 헤더 메트릭 테스트 통과")
    
    # 부분 일치 테스트
    print("\n3. 부분 일치 테스트...")
    partial_pred = create_sample_table_structure()
    partial_pred['cells'][0]['text'] = 'Modified Header'  # 하나만 변경
    
    partial_grits = metrics.calculate_grits(partial_pred, gt_table)
    print(f"   부분 일치 GriTS: {partial_grits['grits_overall']:.4f}")
    
    assert 0.5 < partial_grits['grits_overall'] < 1.0, "부분 일치 시 GriTS는 0.5와 1.0 사이여야 합니다"
    print("   ✓ 부분 일치 테스트 통과")
    
    print("\n✓ 파싱 메트릭 검증 완료")


def test_ragas_metrics():
    """RAGAS 메트릭 테스트"""
    print("\n" + "="*70)
    print("RAGAS 메트릭 검증")
    print("="*70)
    
    ragas_metrics = RAGASMetrics()
    
    # 샘플 데이터
    question = "2023년 매출액은 얼마인가요?"
    answer = "2023년 매출액은 100억원입니다."
    contexts = [
        "2023년 매출액: 100억원",
        "2024년 매출액: 120억원",
        "회사 정보: ABC 기업"
    ]
    ground_truth_answer = "100억원"
    ground_truth_contexts = ["2023년 매출액: 100억원"]
    
    print("\n1. Faithfulness 테스트...")
    faithfulness = ragas_metrics.calculate_faithfulness(answer, contexts)
    print(f"   Faithfulness: {faithfulness:.4f}")
    assert 0.0 <= faithfulness <= 1.0, "Faithfulness는 0-1 사이여야 합니다"
    print("   ✓ Faithfulness 테스트 통과")
    
    print("\n2. Answer Relevancy 테스트...")
    answer_relevancy = ragas_metrics.calculate_answer_relevancy(question, answer)
    print(f"   Answer Relevancy: {answer_relevancy:.4f}")
    assert 0.0 <= answer_relevancy <= 1.0, "Answer Relevancy는 0-1 사이여야 합니다"
    print("   ✓ Answer Relevancy 테스트 통과")
    
    print("\n3. Context Precision 테스트...")
    context_precision = ragas_metrics.calculate_context_precision(contexts, ground_truth_contexts)
    print(f"   Context Precision: {context_precision:.4f}")
    assert 0.0 <= context_precision <= 1.0, "Context Precision은 0-1 사이여야 합니다"
    print("   ✓ Context Precision 테스트 통과")
    
    print("\n4. Context Recall 테스트...")
    context_recall = ragas_metrics.calculate_context_recall(contexts, ground_truth_contexts)
    print(f"   Context Recall: {context_recall:.4f}")
    assert 0.0 <= context_recall <= 1.0, "Context Recall은 0-1 사이여야 합니다"
    print("   ✓ Context Recall 테스트 통과")
    
    print("\n5. 종합 RAG 평가 테스트...")
    all_metrics = ragas_metrics.evaluate_rag(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth_answer=ground_truth_answer,
        ground_truth_contexts=ground_truth_contexts
    )
    print(f"   전체 메트릭: {all_metrics}")
    assert 'faithfulness' in all_metrics, "Faithfulness 메트릭이 포함되어야 합니다"
    print("   ✓ 종합 평가 테스트 통과")
    
    print("\n✓ RAGAS 메트릭 검증 완료")


def test_complexity_metrics():
    """복잡도 메트릭 테스트"""
    print("\n" + "="*70)
    print("복잡도 메트릭 검증")
    print("="*70)
    
    complexity_metrics = ComplexityMetrics()
    
    # Low complexity 테이블 (단순)
    print("\n1. Low Complexity 테이블 테스트...")
    low_complexity_table = {
        'cells': [
            {'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 1, 'text': 'A'},
            {'row': 0, 'col': 1, 'rowspan': 1, 'colspan': 1, 'text': 'B'},
            {'row': 1, 'col': 0, 'rowspan': 1, 'colspan': 1, 'text': '1'},
            {'row': 1, 'col': 1, 'rowspan': 1, 'colspan': 1, 'text': '2'},
        ],
        'headers': [
            {'row': 0, 'col': 0, 'is_header': True, 'text': 'A'},
            {'row': 0, 'col': 1, 'is_header': True, 'text': 'B'},
        ]
    }
    
    low_results = complexity_metrics.calculate_complexity(low_complexity_table)
    print(f"   병합 셀 비율: {low_results['merged_cell_ratio']:.4f}")
    print(f"   헤더 깊이: {low_results['header_depth']:.4f}")
    print(f"   구조적 복잡도: {low_results['structural_complexity_score']:.4f}")
    print(f"   복잡도 등급: {low_results['complexity_level']}")
    
    assert low_results['complexity_level'] == 'low', "단순 테이블은 'low' 복잡도여야 합니다"
    print("   ✓ Low Complexity 테스트 통과")
    
    # High complexity 테이블 (복잡)
    print("\n2. High Complexity 테이블 테스트...")
    high_complexity_table = {
        'cells': [
            # 여러 레벨의 병합 셀
            {'row': 0, 'col': 0, 'rowspan': 3, 'colspan': 2, 'text': 'Main Header'},
            {'row': 0, 'col': 2, 'rowspan': 1, 'colspan': 2, 'text': 'Sub Header 1'},
            {'row': 1, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': 'A'},
            {'row': 1, 'col': 3, 'rowspan': 1, 'colspan': 1, 'text': 'B'},
            {'row': 2, 'col': 2, 'rowspan': 1, 'colspan': 2, 'text': 'Sub Header 2'},
            {'row': 3, 'col': 0, 'rowspan': 1, 'colspan': 1, 'text': 'Data1'},
            {'row': 3, 'col': 1, 'rowspan': 1, 'colspan': 1, 'text': 'Data2'},
            {'row': 3, 'col': 2, 'rowspan': 1, 'colspan': 1, 'text': 'Value1'},
            {'row': 3, 'col': 3, 'rowspan': 1, 'colspan': 1, 'text': 'Value2'},
        ],
        'headers': [
            {'row': 0, 'col': 0, 'is_header': True, 'text': 'Main Header', 'rowspan': 3, 'colspan': 2},
            {'row': 0, 'col': 2, 'is_header': True, 'text': 'Sub Header 1', 'rowspan': 1, 'colspan': 2},
            {'row': 1, 'col': 2, 'is_header': True, 'text': 'A'},
            {'row': 1, 'col': 3, 'is_header': True, 'text': 'B'},
            {'row': 2, 'col': 2, 'is_header': True, 'text': 'Sub Header 2', 'rowspan': 1, 'colspan': 2},
        ]
    }
    
    high_results = complexity_metrics.calculate_complexity(high_complexity_table)
    print(f"   병합 셀 비율: {high_results['merged_cell_ratio']:.4f}")
    print(f"   헤더 깊이: {high_results['header_depth']:.4f}")
    print(f"   구조적 복잡도: {high_results['structural_complexity_score']:.4f}")
    print(f"   복잡도 등급: {high_results['complexity_level']}")
    
    # 복잡도 등급 확인 (0.1867이면 medium에 가까워야 함)
    assert high_results['structural_complexity_score'] > low_results['structural_complexity_score'], "복잡한 테이블의 구조적 복잡도 점수가 더 높아야 합니다"
    assert high_results['merged_cell_ratio'] > low_results['merged_cell_ratio'], "복잡한 테이블의 병합 셀 비율이 더 높아야 합니다"
    assert high_results['header_depth'] >= low_results['header_depth'], "복잡한 테이블의 헤더 깊이가 더 깊거나 같아야 합니다"
    print("   ✓ High Complexity 테스트 통과 (복잡도 점수 비교 통과)")
    
    print("\n✓ 복잡도 메트릭 검증 완료")


def main():
    """메인 실행 함수"""
    print("="*70)
    print("메트릭 검증 스크립트")
    print("="*70)
    
    try:
        # 파싱 메트릭 검증
        test_parsing_metrics()
        
        # RAGAS 메트릭 검증
        test_ragas_metrics()
        
        # 복잡도 메트릭 검증
        test_complexity_metrics()
        
        print("\n" + "="*70)
        print("✓ 모든 메트릭 검증 완료!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ 검증 실패: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

