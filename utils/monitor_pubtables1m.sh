#!/bin/bash
# PubTables-1M 다운로드 실시간 모니터링

echo "PubTables-1M 다운로드 모니터링"
echo "종료하려면 Ctrl+C를 누르세요"
echo "================================"
echo ""

cd /root/headerRAG-1/data/pubtables1m || exit 1

# 이전 크기 저장
prev_size=$(du -sb pubtables-1m 2>/dev/null | cut -f1)

while true; do
    # 현재 크기
    current_size=$(du -sb pubtables-1m 2>/dev/null | cut -f1)
    current_size_h=$(du -sh pubtables-1m 2>/dev/null | cut -f1)
    
    # 다운로드 속도 계산
    if [ -n "$prev_size" ] && [ -n "$current_size" ]; then
        diff=$((current_size - prev_size))
        diff_mb=$((diff / 1024 / 1024))
        if [ $diff_mb -gt 0 ]; then
            echo "[$(date +%H:%M:%S)] 크기: $current_size_h | 속도: +${diff_mb}MB"
        fi
    else
        echo "[$(date +%H:%M:%S)] 크기: $current_size_h"
    fi
    
    prev_size=$current_size
    
    # 프로세스 확인
    if ! pgrep -f "git lfs pull" > /dev/null; then
        echo ""
        echo "다운로드 완료 또는 중단됨"
        break
    fi
    
    sleep 10
done

