"""
PDF에서 테이블 추출 유틸리티
RAG-Evaluation-Dataset-KO의 PDF 파일에서 테이블 추출
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
import tabula
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PDFTableExtractor:
    """PDF에서 테이블 추출 클래스"""
    
    def __init__(self, output_dir: str = "data/extracted_tables"):
        """
        Args:
            output_dir: 추출된 테이블 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_tables_from_pdf(self, pdf_path: str, 
                               table_id_prefix: str = "",
                               use_pdfplumber: bool = True,
                               use_tabula: bool = True) -> List[Dict]:
        """
        PDF 파일에서 모든 테이블 추출
        
        Args:
            pdf_path: PDF 파일 경로
            table_id_prefix: 테이블 ID 접두사
            use_pdfplumber: pdfplumber 사용 여부
            use_tabula: tabula 사용 여부
        
        Returns:
            추출된 테이블 리스트 (딕셔너리 형태)
        """
        tables = []
        pdf_name = Path(pdf_path).stem
        
        # pdfplumber로 테이블 추출
        if use_pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_tables = page.extract_tables()
                        for table_num, table in enumerate(page_tables, 1):
                            if table and len(table) > 0:
                                try:
                                    # 첫 번째 행을 헤더로 사용
                                    df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                                    # 빈 행 제거
                                    df = df.dropna(how='all').dropna(axis=1, how='all')
                                    if not df.empty:
                                        table_id = f"{table_id_prefix}_{pdf_name}_p{page_num}_t{table_num}"
                                        tables.append({
                                            'table_id': table_id,
                                            'table_data': df,
                                            'source': pdf_path,
                                            'page': page_num,
                                            'table_num': table_num,
                                            'method': 'pdfplumber'
                                        })
                                except Exception as e:
                                    print(f"  경고: pdfplumber로 테이블 추출 실패 (페이지 {page_num}, 테이블 {table_num}): {e}")
                                    continue
            except Exception as e:
                print(f"  경고: pdfplumber로 PDF 읽기 실패 ({pdf_path}): {e}")
        
        # tabula로 테이블 추출 (pdfplumber가 실패한 경우 백업)
        if use_tabula and len(tables) == 0:
            try:
                tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
                for table_num, df in enumerate(tabula_tables, 1):
                    if df is not None and not df.empty:
                        # 빈 행/열 제거
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        if not df.empty:
                            table_id = f"{table_id_prefix}_{pdf_name}_t{table_num}"
                            tables.append({
                                'table_id': table_id,
                                'table_data': df,
                                'source': pdf_path,
                                'page': 0,  # tabula는 페이지 정보를 직접 제공하지 않음
                                'table_num': table_num,
                                'method': 'tabula'
                            })
            except Exception as e:
                print(f"  경고: tabula로 PDF 읽기 실패 ({pdf_path}): {e}")
        
        return tables
    
    def extract_all_from_dataset(self, 
                                 documents_csv: str = "RAG-Evaluation-Dataset-KO/documents.csv",
                                 pdf_base_dir: str = "RAG-Evaluation-Dataset-KO",
                                 save_to_excel: bool = True) -> List[Dict]:
        """
        데이터셋의 모든 PDF에서 테이블 추출
        
        Args:
            documents_csv: 문서 메타데이터 CSV 파일 경로
            pdf_base_dir: PDF 파일이 있는 기본 디렉토리
            save_to_excel: Excel 파일로 저장 여부
        
        Returns:
            추출된 모든 테이블 리스트
        """
        # 문서 메타데이터 로드
        if not os.path.exists(documents_csv):
            print(f"경고: 문서 CSV 파일을 찾을 수 없습니다: {documents_csv}")
            return []
        
        df_docs = pd.read_csv(documents_csv)
        all_tables = []
        
        print(f"총 {len(df_docs)}개 문서에서 테이블 추출 시작...")
        
        for idx, row in tqdm(df_docs.iterrows(), total=len(df_docs), desc="PDF 처리"):
            domain = row['domain']
            file_name = row['file_name']
            pdf_path = os.path.join(pdf_base_dir, domain, file_name)
            
            # PDF 파일 존재 확인
            if not os.path.exists(pdf_path):
                # 다른 경로 시도
                alt_paths = [
                    os.path.join(pdf_base_dir, file_name),
                    os.path.join("data", domain, file_name),
                    os.path.join("data", file_name)
                ]
                pdf_path = None
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        pdf_path = alt_path
                        break
                
                if pdf_path is None:
                    print(f"  경고: PDF 파일을 찾을 수 없습니다: {file_name}")
                    continue
            
            # 테이블 추출
            tables = self.extract_tables_from_pdf(
                pdf_path, 
                table_id_prefix=f"{domain}_{idx}"
            )
            
            # Excel로 저장
            if save_to_excel and tables:
                domain_dir = self.output_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)
                
                for table_info in tables:
                    excel_path = domain_dir / f"{table_info['table_id']}.xlsx"
                    try:
                        table_info['table_data'].to_excel(excel_path, index=False)
                    except Exception as e:
                        print(f"  경고: Excel 저장 실패 ({table_info['table_id']}): {e}")
            
            all_tables.extend(tables)
        
        print(f"\n총 {len(all_tables)}개 테이블 추출 완료")
        return all_tables
    
    def get_all_tables_as_dataframes(self, 
                                     documents_csv: str = "RAG-Evaluation-Dataset-KO/documents.csv",
                                     pdf_base_dir: str = "RAG-Evaluation-Dataset-KO") -> List[pd.DataFrame]:
        """
        추출된 모든 테이블을 DataFrame 리스트로 반환
        
        Returns:
            DataFrame 리스트
        """
        tables_info = self.extract_all_from_dataset(documents_csv, pdf_base_dir, save_to_excel=False)
        return [info['table_data'] for info in tables_info]


def main():
    """메인 실행 함수"""
    extractor = PDFTableExtractor(output_dir="data/extracted_tables")
    
    # 전체 데이터셋에서 테이블 추출
    print("=== PDF 테이블 추출 시작 ===")
    all_tables = extractor.extract_all_from_dataset(
        documents_csv="RAG-Evaluation-Dataset-KO/documents.csv",
        pdf_base_dir="RAG-Evaluation-Dataset-KO",
        save_to_excel=True
    )
    
    print(f"\n추출 완료: {len(all_tables)}개 테이블")
    print(f"저장 위치: {extractor.output_dir}")


if __name__ == "__main__":
    main()

