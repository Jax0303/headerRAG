#!/usr/bin/env python
"""
데이터셋 자동 다운로드 스크립트
"""
import sys
from utils.download_external_datasets import ExternalDatasetDownloader

def main():
    downloader = ExternalDatasetDownloader()
    
    print("="*70)
    print("데이터셋 자동 다운로드 시작")
    print("="*70)
    
    # 1. RAG-Evaluation-Dataset-KO 다운로드 (가장 중요)
    print("\n" + "="*70)
    print("1. RAG-Evaluation-Dataset-KO 다운로드")
    print("="*70)
    try:
        from utils.download_datasets import DatasetDownloader
        rag_downloader = DatasetDownloader()
        rag_dir = rag_downloader.download_rag_eval_ko()
        print(f"✅ RAG-Evaluation-Dataset-KO 다운로드 완료: {rag_dir}")
    except Exception as e:
        print(f"❌ RAG-Evaluation-Dataset-KO 다운로드 실패: {e}")
    
    # 2. PubTables-1M 샘플 다운로드 (선택사항)
    print("\n" + "="*70)
    print("2. PubTables-1M 다운로드")
    print("="*70)
    
    # 사용자 선택
    print("\n다운로드 방법 선택:")
    print("1. 샘플만 다운로드 (1000개, 빠름, 권장)")
    print("2. 전체 다운로드 (git-lfs 사용, 매우 느림, 수십 GB)")
    print("3. 건너뛰기")
    
    choice = input("\n선택 (1/2/3, 기본값: 1): ").strip() or "1"
    
    if choice == "1":
        try:
            pubtables_dir = downloader.download_pubtables1m_hf(num_samples=1000, use_git_lfs=False)
            print(f"✅ PubTables-1M 샘플 다운로드 완료: {pubtables_dir}")
        except Exception as e:
            print(f"⚠️  PubTables-1M 샘플 다운로드 실패: {e}")
    elif choice == "2":
        print("\n⚠️  경고: 전체 데이터셋은 수십 GB이며 다운로드에 시간이 오래 걸립니다.")
        confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                pubtables_dir = downloader.download_pubtables1m_hf(use_git_lfs=True)
                print(f"✅ PubTables-1M 전체 다운로드 완료: {pubtables_dir}")
            except Exception as e:
                print(f"❌ PubTables-1M 전체 다운로드 실패: {e}")
        else:
            print("다운로드 취소됨")
    else:
        print("PubTables-1M 다운로드 건너뛰기")
    
    print("\n" + "="*70)
    print("다운로드 완료!")
    print("="*70)

if __name__ == "__main__":
    main()




