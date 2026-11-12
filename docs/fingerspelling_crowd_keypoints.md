# CROWD 키포인트 확보 현황 및 CTC 입력 가능성

확인일: 2026-11-13 (Asia/Seoul). 브랜치: `feat/2-crowd-keypoint-adapter`.
관련 이슈: [Refs #2](https://github.com/yj2trigger/sign-language-recognition/issues/2).

## 결론

지정 경로와 하위 디렉터리에는 형태소 라벨 ZIP만 있다. ZIP 내부도 형태소 JSON
2,000개뿐이다. 키포인트·영상은 확보되지 않았다. 따라서 실제 키포인트 규격을
확인하거나 `(T,53,2)` xy / `(T,53)` mask로 변환할 근거가 없다.
`crowd_keypoints.py`, `crowd_keypoint_adapter.py`는 만들지 않았다.
이번 변경은 읽기 전용 목록 검사와 조사 보고서이며, 작동하는 입력 어댑터나
Issue #2 전체 완료를 주장하지 않는다.

## 사전 조사 및 기존 구현

- 시작 시 작업 트리는 깨끗했고 요청 브랜치에 있었다. HEAD, main, origin/main은
  `c627cf77eb55564fb1a85ece63628d5bba67bb9a`였다. 브랜치 전환·merge는 하지 않았다.
- GitHub main의 [landmarks.py](https://github.com/yj2trigger/sign-language-recognition/blob/main/src/landmarks.py)
  blob `379bab4a5309964bcb106cc51daa69300475f8b6`와
  [build_dataset.py](https://github.com/yj2trigger/sign-language-recognition/blob/main/src/build_dataset.py)
  blob `00e586287b225f61121e65cdb1eacef4e692f7d9`를 조회했고 로컬 blob과 같았다.
  셸의 원격 접속은 실패했으나 연결된 GitHub 도구로 파일을 확인했다.
- Issue #2는 MediaPipe 53점, 15fps 입력과 미학습 수행자 CER 평가를 요구한다.
  이슈의 세로 영상·60fps 설명은 계획이며 이번에 실물로 검증한 값은 아니다.
- landmarks.py는 pose `[0,2,5,7,8,11,12,13,14,15,16]` 11점,
  왼손 21점, 오른손 21점 순서다. pose visibility 0.5 이상을 관측으로 표시한다.
  process_frame은 16:9 padding 이후 이미지 정규화 xy를 만든다.
  외부 좌표를 연결하려면 원래 영상 크기와 padding·좌우 방향도 확인해야 한다.
- build_dataset.py는 WORD/REAL 파일명과 라벨 경로를 전제로 한다. CROWD ZIP을
  그대로 넣을 수 없다. --full-fps는 t, src_fps, phase 등을 보존한다.
- [PR #12](https://github.com/yj2trigger/sign-language-recognition/pull/12)의
  `expected_video_name`, `parse_crowd_label`, `read_crowd_zip`, `compare_video_names`가
  라벨 검증·영상명 대응·누락 및 중복 검출을 이미 제공한다.
  `morpheme/<그룹>/NIA_SL_FS<번호>_CROWD<그룹>_F_morpheme.json`에서 예상 영상명을
  만들고 metaData.name과 URL basename을 확인한다. FS 번호만으로 정답을 합치지 않는다.
  실제 조사에서는 이 브랜치의 파서를 git show로 메모리에 읽어 재사용했다.
  파일을 복사하거나 현재 브랜치에 선행 PR을 적용하지 않았다.
- [PR #16](https://github.com/yj2trigger/sign-language-recognition/pull/16)은
  sequence_validation, sequence_preprocessing, sequence_features, ctc_batching을 제공한다.
  xy `(T,53,2)`, bool mask `(T,53)`, fps 또는 실제 timestamps를 받아 결측과 시간축을
  처리한다. 동일 기능을 추가하지 않았다.
- [PR #17](https://github.com/yj2trigger/sign-language-recognition/pull/17)의 모델은
  float32 `[B,T,F]`, input_lengths를 받는다. F는 설정한 input_size와 같아야 한다.
  raw xy/mask를 모델에 직접 넣는 계약이 아니며, #16의 feature 출력 및 배치와 연결한다.
  Packed BiGRU가 시간 길이를 보존하며 모든 입력 값은 padding까지 유한해야 한다.

## 실제 확보 데이터와 관찰한 구조

조사 루트: `D:\UNIVERSITY\UAI_project2\sign-language-data`.
유일한 파일(식별자·정답이 없는 배포 ZIP 이름):
`fingerspelling/crowd/수어 영상/2.Validation/[라벨]01_crowd_morpheme.zip`.
크기 952,783 bytes. 별도 JSON 0개, ZIP 1개, ZIP 내부 JSON 2,000개,
중복 member name 0개, 기타 파일 member 0개. 전체 압축 해제는 하지 않았다.

| 관찰 항목 | 실제 확인 결과 |
| --- | --- |
| JSON 최상위 | metaData, data (각 2,000개) |
| metaData 필드 | url, name, duration, exportedOn (각 2,000개) |
| duration 타입 | float 1,794개, int 206개; FPS로 해석하지 않음 |
| data | 파일당 segment 1개; start, end, attributes |
| 영상 연결 키 | metaData.name, URL basename, 라벨 member 이름; 파서로 일치 확인 |
| FPS | metaData.fps 2,000개 모두 없음 |
| 시간 | 라벨 구간 start/end 및 duration 존재; 키포인트 프레임 시각 없음 |
| 키포인트 형식·관절 수·인덱스·차원·좌표계 | 모두 미확인 |
| 프레임 수·프레임별 시간·좌표 이상 | 키포인트 부재로 검증 불가 |
| 수행자 ID | 공식 정의 미확보. CROWD 번호는 파일 그룹 키로만 취급 |
| 학습/평가 분리 | Validation 라벨만 있음. 독립 수행자 분할 가능성 확인 불가 |

파일별 이름·URL·정답·그룹 ID 값은 출력하거나 보고서에 저장하지 않았다.
다른 경로에 데이터가 없다고 단정하지 않으며 이 결과는 위 루트 전체 스캔에 한정한다.

## 실제 검증 결과

PR #12 파서를 그대로 사용해 전체 JSON을 순차적으로 읽었다.

| 항목 | 결과 |
| --- | --- |
| JSON 정상 파싱 및 영상명·정답 매핑 | 2,000 |
| 파싱/매핑 오류 | 0 |
| 파일명 불일치 | 0 |
| 동일 FS 그룹의 서로 다른 정답 | 3그룹; 자동 수정하지 않음 |
| 키포인트 파일 | 0 |
| 라벨과 대응되는 실제 키포인트/영상 파일 | 각각 0 |
| 키포인트의 라벨 누락 수 | 평가 대상 없음; 0건 통과로 해석하지 않음 |
| 관절·프레임 수 불일치, NaN/Inf, 결측 관절 | 평가 불가 |
| 키포인트 FPS/시간 정보 누락 건수 | 평가 불가; 키포인트 자체 없음 |
| 원본 ZIP 불변성 | 전체 읽기 전후 SHA-256 동일 |

## 호환성 판정

| 구분 | 이번 자료에 대한 판정 |
| --- | --- |
| 직접 사용 가능 | 입증되지 않음. 형태소 라벨은 xy/mask가 아님 |
| 명확한 변환 규칙으로 사용 가능 | 입증되지 않음. 실제 관절 대응표 없음 |
| 추가 명세 필요 | 관절 인덱스 의미, 좌우, 좌표계·크기·회전·padding, confidence/결측 정의, 시간축, 영상 연결 규칙, 공식 수행자 정의 |
| 현재 데이터만으로 변환 불가능 | 해당. 정답 문자열·구간으로 관절 좌표를 복원할 수 없음 |

53개 관절을 가졌다는 이유만으로 MediaPipe와 동일하다고 판단하면 안 된다.
확보 후 실제 대응 가능한 관절만 매핑해야 하며 관측되지 않은 정보를 생성해서는 안 된다.
대상 관절이 없는 경우 학습 입력 허용 정책을 명시하고 관측 마스크로 구분해야 한다.

## 추가한 기능과 재현 방법

`src/fingerspelling/crowd_dataset_validation.py`의 `inventory_dataset`은 디렉터리와 ZIP
중앙 목록만 읽는다. payload 파싱·CRC 검사·프레임 검증·라벨 매핑은 수행하지 않는다.
확장자 기반 `video_candidate`, `unclassified_data_candidate`는 내용 확인이 아니다.
임의 JSON이 키포인트라고 주장하지 않으며 결과의 keypoint_schema는 항상 not_inspected다.
`morpheme_json`도 경로/파일명 분류일 뿐이다. 이번 실데이터 내용 검증은 별도로 #12를 재사용했다.

`InventoryReport.summary`는 집계만 반환하고 `InventoryError`는 고정 오류 코드만 제공한다.
심볼릭 링크와 junction은 거부하며 전체 파일/ZIP member 수 제한을 제공한다.
ZIP 중앙 목록은 zipfile에서 먼저 메모리로 읽으므로 member 제한이 중앙 목록 메모리의
상한을 보장하지 않는다. 원본 경로는 신뢰 가능한 정적인 로컬 데이터 디렉터리를 전제로 한다.
ZIP 내부의 중첩 ZIP은 재귀 조사하지 않으므로 other로만 집계한다.

```powershell
$taskPython = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $taskPython src/fingerspelling/crowd_dataset_validation.py 'D:\UNIVERSITY\UAI_project2\sign-language-data'
& $taskPython -m pytest -q tests/test_crowd_dataset_validation.py
```

스캔 실패 시 부분 집계를 반환하지 않고 CLI는 exit 2와 비식별 오류 코드를 출력한다.
신규 런타임 의존성은 없고 Python 3.12 이상을 사용한다.

## 검증 범위와 후속 작업

합성 테스트는 빈 디렉터리, 후보 분류, 대소문자 확장자, ZIP 내 중복, ZIP 간 중복 범위,
경로 이탈 member의 비추출, 원본 해시 유지, 손상 ZIP, 없는 루트, 파일 루트,
잘못된 제한, 전체 제한 및 10,000-member 목록과 CLI 비노출을 확인한다.
좌표 형상·관절 인덱스·FPS·NaN/Inf 테스트는 변환 코드가 없어 해당하지 않는다.
이를 합성 키포인트 어댑터 검증으로 대체했다고 주장하지 않는다.

추가 확보할 자료는 실제 CROWD 키포인트 샘플과 공식 포맷/관절 대응 명세,
영상-키포인트-라벨 대응 명세, 시간/FPS 메타데이터, 공식 수행자 ID 정의 및
학습/평가 세트다. 원본 영상 다운로드는 실행하지 않았다.
자료 확보 후 작은 샘플의 구조를 조사하고 #12 매핑, #16 전처리, #17 모델을 연결할
신규 어댑터와 좌표·시간·누락·회귀 테스트를 구현한다. 이후 수행자 독립 CER를 평가한다.

PR은 **데이터 확보 현황 조사 및 읽기 전용 목록 검사** 범위로 제출 가능하다.
키포인트 변환 완료 PR로 제출하거나 Issue #2를 종료할 수 없다.
커밋·push·PR 생성·merge는 하지 않았다.

## 독립 리뷰와 변경 위치

별도 리뷰 에이전트가 일반 디렉터리의 상대경로 분류가 ZIP과 달라지는 결함 1건을
재현했다. 상대경로로 분류하도록 수정하고 회귀 테스트를 추가했다. 링크 거부,
디렉터리 읽기 실패, ZIP payload 비읽기 검증도 보강했다. 재리뷰에서 잔여 필수
수정 결함은 발견되지 않았다. 사람의 독립 리뷰는 미실행이다.
신규 합성 테스트 결과: **20 passed, 12 subtests passed** (Python 3.12.16).

모든 변경 파일은 신규이며 기존 추적 파일은 수정하지 않았다.

| 파일 | 실제 추가 행 | 공개 함수/클래스 위치 |
| --- | --- | --- |
| src/fingerspelling/crowd_dataset_validation.py | 1–118 | InventoryError 15–16, InventoryReport 32–49, inventory_dataset 52–101, main 104–114 |
| tests/test_crowd_dataset_validation.py | 1–162 | InventoryTests 18–158 |

PR #11–#17의 GitHub 변경 파일 목록과 비교해 동일 신규 경로가 없음을 확인했다.
__init__.py도 추가하지 않아 기존 패키지 정의와 중복되지 않는다. #12의 라벨 파서,
#16의 좌표·시간·배치 검증, #17 모델 구현을 수정하거나 복제하지 않았다.
현재 파일 수준 충돌은 없으나 선행 PR 통합 후 인터페이스 회귀 검증은 필요하다.
이 문서와 PR 초안은 각 파일의 1행부터 마지막 행까지 신규 추가다.
