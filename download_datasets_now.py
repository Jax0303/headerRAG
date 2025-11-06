#!/usr/bin/env python
"""
데이터셋 자동 다운로드 스크립트
"""
from utils.download_datasets import DatasetDownloader

def main():
    downloader = DatasetDownloader()
    
    print("="*70)
    print("데이터셋 자동 다운로드 시작")
    print("="*70)
    
    # 1. RAG-Evaluation-Dataset-KO 다운로드 (가장 중요)
    print("\n" + "="*70)
    print("1. RAG-Evaluation-Dataset-KO 다운로드")
    print("="*70)
    try:
        rag_dir = downloader.download_rag_eval_ko()
        print(f"✅ RAG-Evaluation-Dataset-KO 다운로드 완료: {rag_dir}")
    except Exception as e:
        print(f"❌ RAG-Evaluation-Dataset-KO 다운로드 실패: {e}")
    
    # 2. PubTables-1M 샘플 다운로드 (선택사항)
    print("\n" + "="*70)
    print("2. PubTables-1M 샘플 다운로드 (1000개)")
    print("="*70)
    try:
        pubtables_dir = downloader.download_pubtables1m_hf(num_samples=1000)
        print(f"✅ PubTables-1M 샘플 다운로드 완료: {pubtables_dir}")
    except Exception as e:
        print(f"⚠️  PubTables-1M 다운로드 실패 (선택사항): {e}")
    
    print("\n" + "="*70)
    print("다운로드 완료!")
    print("="*70)

if __name__ == "__main__":
    main()



