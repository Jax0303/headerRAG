#!/bin/bash
# PubTables-1M 다운로드 진행 상황 확인 스크립트

cd /root/headerRAG-1/data/pubtables1m || exit 1

echo "=" | head -c 70; echo ""
echo "PubTables-1M 다운로드 진행 상황"
echo "=" | head -c 70; echo ""

# 프로세스 확인
echo -n "다운로드 프로세스: "
if pgrep -f "git lfs pull" > /dev/null; then
    echo "✅ 실행 중"
else
    echo "❌ 실행 중이지 않음"
fi

# 디렉토리 크기
echo -n "현재 디렉토리 크기: "
du -sh pubtables-1m 2>/dev/null | cut -f1

# 다운로드된 파일 확인
echo ""
echo "다운로드된 파일 (실제 크기, 100B 이상만 표시):"
cd pubtables-1m
ls -lh *.tar.gz 2>/dev/null | while read line; do
    size=$(echo "$line" | awk '{print $5}')
    name=$(echo "$line" | awk '{print $9}')
    # 100B 이상인 파일만 표시 (포인터 파일 제외)
    if [[ "$size" =~ [0-9]+[KMGT] ]] || [[ "$size" =~ [0-9]+\.[0-9]+[KMGT] ]]; then
        echo "$name: $size"
    fi
done | head -10

# 로그 확인
echo ""
echo "최근 로그 (마지막 10줄):"
if [ -f ../lfs_download.log ]; then
    tail -10 ../lfs_download.log
else
    echo "로그 파일이 없습니다."
fi

echo ""
echo "진행 상황 모니터링: tail -f data/pubtables1m/lfs_download.log"

