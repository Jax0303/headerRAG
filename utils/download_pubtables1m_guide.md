# PubTables-1M 다운로드 문제 해결 가이드

## 문제: `git lfs pull`이 멈춤

### 원인
- 전체 데이터셋이 매우 큼 (수십 GB)
- 네트워크 속도에 따라 시간이 오래 걸림
- 18개의 대용량 tar.gz 파일 다운로드 필요

### 해결 방법

#### 방법 1: 선택적 다운로드 (권장) ⭐

필요한 파일만 선택적으로 다운로드:

```bash
cd data/pubtables1m/pubtables-1m

# 1. Filelists만 (가장 작음, 빠름)
git lfs pull --include="*Filelists*.tar.gz"

# 2. Structure Annotations만 (표 구조 정보, 실험에 유용)
git lfs pull --include="PubTables-1M-Structure_*.tar.gz"

# 3. Detection Annotations만 (표 감지 정보)
git lfs pull --include="PubTables-1M-Detection_*.tar.gz"

# 4. 특정 파일만
git lfs pull --include="PubTables-1M-Structure_Annotations_Train.tar.gz"
```

#### 방법 2: 백그라운드로 실행

```bash
cd data/pubtables1m/pubtables-1m

# 백그라운드로 실행
nohup git lfs pull > lfs_download.log 2>&1 &

# 진행 상황 확인
tail -f lfs_download.log

# 프로세스 확인
ps aux | grep "git lfs"
```

#### 방법 3: 타임아웃 설정 및 재시도

```bash
cd data/pubtables1m/pubtables-1m

# 타임아웃 설정 (예: 1시간)
timeout 3600 git lfs pull

# 실패한 파일만 다시 다운로드
git lfs fetch --all
```

#### 방법 4: 샘플만 사용 (가장 빠름)

전체 다운로드 대신 샘플만 사용:

```python
from utils.download_external_datasets import ExternalDatasetDownloader

downloader = ExternalDatasetDownloader()
# 샘플만 다운로드 (1000개)
pubtables_dir = downloader.download_pubtables1m_hf(num_samples=1000)
```

## 파일 목록

다운로드 가능한 파일들:

1. **Structure Annotations** (표 구조 정보)
   - `PubTables-1M-Structure_Annotations_Train.tar.gz`
   - `PubTables-1M-Structure_Annotations_Test.tar.gz`
   - `PubTables-1M-Structure_Annotations_Val.tar.gz`
   - `PubTables-1M-Structure_Filelists.tar.gz`
   - `PubTables-1M-Structure_Images_Test.tar.gz`

2. **Detection Annotations** (표 감지 정보)
   - `PubTables-1M-Detection_Annotations_Train.tar.gz`
   - `PubTables-1M-Detection_Annotations_Test.tar.gz`
   - `PubTables-1M-Detection_Annotations_Val.tar.gz`
   - `PubTables-1M-Detection_Filelists.tar.gz`
   - `PubTables-1M-Detection_Images_*.tar.gz`

3. **기타**
   - `PubTables-1M-PDF_Annotations.tar.gz`
   - `PubTables-1M-Detection_Page_Words.tar.gz`

## 진행 상황 확인

```bash
# 다운로드된 파일 확인
cd data/pubtables1m/pubtables-1m
ls -lh *.tar.gz

# 실제 파일 크기 확인 (포인터가 아닌)
git lfs ls-files | grep -v "^-"

# 디렉토리 크기 확인
du -sh .
```

## 권장 사항

실험 목적이라면:
1. **Filelists만 다운로드** (가장 작고 빠름)
2. 또는 **샘플만 사용** (datasets 라이브러리 사용)
3. 필요할 때만 추가 파일 다운로드

전체 데이터셋이 필요한 경우:
- 백그라운드로 실행
- 네트워크가 안정적인 시간에 실행
- 충분한 디스크 공간 확보 (수십 GB)

