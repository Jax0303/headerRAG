"""
RAG-Evaluation-Dataset-KO의 PDF 파일 다운로드 유틸리티
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


def download_pdfs_from_dataset(documents_csv: str = "RAG-Evaluation-Dataset-KO/documents.csv",
                               output_base_dir: str = "RAG-Evaluation-Dataset-KO"):
    """
    documents.csv에서 PDF URL을 읽어서 다운로드
    
    Args:
        documents_csv: 문서 메타데이터 CSV 파일 경로
        output_base_dir: PDF 저장 기본 디렉토리
    """
    df = pd.read_csv(documents_csv)
    output_base = Path(output_base_dir)
    
    print(f"총 {len(df)}개 문서 다운로드 시작...")
    
    downloaded = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="PDF 다운로드"):
        domain = row['domain']
        file_name = row['file_name']
        url = row['url']
        
        # 도메인별 디렉토리 생성
        domain_dir = output_base / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = domain_dir / file_name
        
        # 이미 다운로드된 파일은 스킵
        if pdf_path.exists():
            downloaded += 1
            continue
        
        try:
            # PDF 다운로드
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded += 1
            time.sleep(0.5)  # 서버 부하 방지
            
        except Exception as e:
            print(f"\n경고: 다운로드 실패 ({file_name}): {e}")
            failed += 1
            continue
    
    print(f"\n다운로드 완료: {downloaded}개 성공, {failed}개 실패")
    return downloaded, failed


if __name__ == "__main__":
    download_pdfs_from_dataset()

