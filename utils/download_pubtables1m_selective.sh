#!/bin/bash
# PubTables-1M 선택적 다운로드 스크립트
# 특정 파일만 다운로드하여 시간과 공간 절약

cd /root/headerRAG-1/data/pubtables1m/pubtables-1m || exit 1

echo "PubTables-1M 선택적 다운로드"
echo "================================"
echo ""
echo "다운로드할 파일 선택:"
echo "1. Structure Annotations만 (표 구조 정보, 추천)"
echo "2. Detection Annotations만 (표 감지 정보)"
echo "3. Filelists만 (파일 목록, 작음)"
echo "4. 모든 파일 (매우 큼, 시간 오래 걸림)"
echo ""

read -p "선택 (1-4, 기본값: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo "Structure Annotations 다운로드 중..."
        git lfs pull --include="PubTables-1M-Structure_*.tar.gz" --exclude=""
        ;;
    2)
        echo "Detection Annotations 다운로드 중..."
        git lfs pull --include="PubTables-1M-Detection_*.tar.gz" --exclude=""
        ;;
    3)
        echo "Filelists 다운로드 중..."
        git lfs pull --include="*Filelists*.tar.gz" --exclude=""
        ;;
    4)
        echo "모든 파일 다운로드 중... (시간이 오래 걸립니다)"
        git lfs pull
        ;;
    *)
        echo "잘못된 선택"
        exit 1
        ;;
esac

echo ""
echo "다운로드 완료!"

