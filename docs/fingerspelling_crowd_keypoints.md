# CROWD 실제 키포인트 파서와 CTC 입력 어댑터

## Commit 2 조사 및 구현

작업 브랜치는 `feat/2-crowd-keypoint-adapter`이며 Base는 main이다. 시작 시 Commit 1이
`ada2735`로 커밋되어 있었다. 미추적 PR 초안 `docs/fingerspelling_crowd_keypoints_pr.md`를
보존해 이번 결과로 갱신했다. 조사일은 2026-11-13 (Asia/Seoul)이다.
현재 결과는 이 절을 기준으로 읽으며 아래 Commit 1 기록은 당시 확보 현황을 보존한다.

### 확보 자료 및 ZIP 검사

실제 키포인트 경로는 예상 downloads 폴더의 하위 디렉터리에서 발견했다.

```text
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\downloads\수어 영상\2.Validation\[라벨]01_crowd_keypoint.zip
```

| 항목 | 확인값 |
| --- | --- |
| 키포인트 ZIP 크기 | 1,411,121,080 bytes |
| ZIP 파일 member 및 JSON 수 | 671,745 |
| 압축 해제 시 총 크기 | 6,461,753,903 bytes |
| 최대 JSON 크기 | 12,265 bytes |
| 최대 member 압축률 | 약 7.213 |
| 영상 ID 그룹 | 2,000 |
| 영상별 프레임 수 | 최소 12, 최대 974 |
| 프레임 번호 누락 및 중복 | 각각 0 |
| ZIP CRC/읽기 오류 | 0 |
| 원본 해시 | 전체 검사 전후 SHA-256 동일 |

형태소 ZIP은 기존 `crowd/수어 영상/2.Validation/[라벨]01_crowd_morpheme.zip`에 있으며
952,783 bytes, JSON 2,000개다. 전체 압축 해제를 수행하지 않고 zipfile로 순차 읽었다.
실제 영상은 조사 루트에 0개다. 영상 내용과 라벨의 일치 여부는 미검증이다.

### 실제 구조와 검증 결과

파일명 규칙은 다음과 같다. 자리수와 폴더·파일의 video ID 일치를 검증한다.
식별자 값은 산출물에 기록하지 않는다.

```text
keypoint/{CROWD 두 자리}/{VIDEO_BASE}/{VIDEO_BASE}_{프레임 열두 자리}_keypoints.json
VIDEO_BASE = NIA_SL_FS{FS 네 자리}_CROWD{CROWD 두 자리}_F
```

샘플 JSON 최상위는 version, people, camparam이며 version은 1.3이다.
people은 단일 객체로 person_id와 아래 배열을 가진다.

| 필드 | 관절 수 | 1D 배열 길이 | 해석 |
| --- | --- | --- | --- |
| pose_keypoints_2d | 25 | 75 | x,y,confidence triples |
| face_keypoints_2d | 70 | 210 | x,y,confidence triples |
| hand_left_keypoints_2d | 21 | 63 | x,y,confidence triples |
| hand_right_keypoints_2d | 21 | 63 | x,y,confidence triples |
| 각 그룹의 _3d 배열 | 0 | 0 | 샘플에서 비어 있음 |

AI-Hub 공개 페이지는 위 배열의 x,y,confidence와 30fps 분할 이미지를 설명한다.
OpenPose는 출력 scale 설정에 따라 좌표 단위가 달라질 수 있으므로 숫자 크기만으로
pixel 좌표라고 확정하지 않는다. 이미지 크기, 좌표 원점·방향·단위, BODY_25/Hand21
내보내기 의미, anatomical left/right 및 비반전 여부는 SourceSpec에 검증 근거와 함께
명시해야 한다. JSON person_id를 공식 수행자 ID로 사용하지 않는다.
[AI-Hub 데이터 설명](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103),
[OpenPose 출력 명세](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md)

| 전체 순차 검사 항목 | 결과 |
| --- | --- |
| 조사한 JSON | 671,745 |
| 엄격 파서 통과 프레임 | 529,700 |
| JSON 문법 정상 | 671,744 |
| JSON 문법 오류 | 1 |
| confidence [0,1] 위반 프레임 | 142,044 |
| 엄격 파서 기준 모든 프레임 통과 영상 | 446 |
| 엄격 파서 통과 프레임의 관절 형상 | 모두 (25,70,21,21) |
| 통과 프레임의 confidence=0 관절 | 4,749,826 / 72,568,900 (약 6.5453%) |
| 영상 ID와 형태소 매핑 | 2,000 / 2,000 |
| 키포인트의 라벨 누락 / 라벨의 키포인트 누락 | 0 / 0 |
| 검증된 실제 데이터 변환 성공 | 0 |
| source 명세 부재로 변환 차단 | 2,000 영상 |

별도 전체 구조 조사에서 문법 정상 671,744개 모두 (25,70,21,21)이었다.
최상위 필드는 version/people 각 671,744개, camparam 585,816개이며 그 밖의 필드는 없었다.
people 필드도 person_id와 위 2d/3d 배열뿐이어서 FPS, timestamp, 이미지 크기 필드가 없다.
camparam 하위에 비어 있지 않은 값은 0개였다. 문법 오류 파일1개는 필드를 조사할 수 없었다.
confidence 범위 위반이 있는 프레임은 body534, face120,410, 왼손15,312, 오른손11,545개다.
그룹 간 중복이 있으므로 이를 합산해 프레임 수로 해석하지 않는다.
범위 위반 confidence 값은 음수37,795개, 1 초과172,851개였다.
JSON 문법 오류와 이러한 품질 오류는 ZIP CRC 오류와 별도로 집계한다.

confidence 위반은 공식 명세에 따른 엄격 품질 판정이다. clip하거나 정답으로 자동
수정하지 않는다. 파서 오류는 프레임별 첫 오류 코드이며 하나의 프레임에 여러 문제가
있을 때 모든 문제를 중복 집계한 값은 아니다. 정상 JSON 문법 수와 엄격 파서 통과 수는
구분한다. valid_sequences는 모든 JSON이 엄격 파서를 통과했다는 뜻이며 프레임 번호
누락, 실제 영상 내용, 학습 적합성까지 보증하지 않는다.

confidence=0 비율은 검출 실패 후보 비율이다. 기준 confidence 0.5 미만까지 포함한
최종 mask 결측률과는 다르다. source 명세가 확보되면 실제 mask를 생성해 평가한다.
좌표에는 음수와 1을 넘는 값이 관찰되어 canvas 확인 전 범위 오류 여부를 확정하지 않는다.
PR #12 파서를 메모리에 불러 재사용했고 정답 불일치 3그룹을 그대로 유지했다.

### 관절 대응과 공통 표현

기존 landmarks.py는 pose 11점 `[0,2,5,7,8,11,12,13,14,15,16]`, 왼손 21점,
오른손 21점의 순서다. process_frame은 fit_16x9 padding 후 정규화한다.
pose 관측 기준은 visibility>=0.5이고 손은 검출 성공 시 전체 관절을 관측으로 표시한다.
face33은 별도 배열이며 기본53에 포함되지 않는다. full 추출의 t는 원본 프레임 시각이다.
AI-Hub의 confidence와 MediaPipe visibility가 통계적으로 같은 척도라는 가정은 적용하지 않는다.

공통 표현은 BODY_25/Hand21 의미가 확인된 입력을 위한 `(T,51,2)` xy와 `(T,51)`
bool mask, 실제 timestamps, 원본 frame_indices, private video_id다.

| 공통 인덱스 | 의미 | BODY_25 인덱스 | 기존53 슬롯 |
| --- | --- | --- | --- |
| 0 | 코 | 0 | 0 |
| 1,2 | 왼귀, 오른귀 | 18,17 | 3,4 |
| 3,4 | 왼어깨, 오른어깨 | 5,2 | 5,6 |
| 5,6 | 왼팔꿈치, 오른팔꿈치 | 6,3 | 7,8 |
| 7,8 | 왼손목, 오른손목 | 7,4 | 9,10 |
| 9–29 | 왼손 wrist 및 손가락 21점 | hand_left 0–20 | 11–31 |
| 30–50 | 오른손 wrist 및 손가락 21점 | hand_right 0–20 | 32–52 |

공통 손 순서는 wrist, thumb 4점, index 4점, middle 4점, ring 4점, pinky 4점이다.
MediaPipe pose 2/5는 눈 안쪽 점이며 OpenPose eye 중심과 동일한 위치로 대입하지 않는다.
얼굴70, neck, 하체 등 공통51에 없는 관절은 출력에서 제외한다. anatomical 명칭을
대응시켜도 두 검출기 사이의 실제 위치 오차·분포 차이는 영상으로 평가해야 한다.
[MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker),
[MediaPipe Hand](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

기존 #16은 53슬롯을 요구한다. `to_ctc53`은 기본적으로 inner_eyes_unavailable로
차단한다. `allow_missing_inner_eyes=True`를 선택하면 슬롯1/2를 mask=False와 0 sentinel로
기록한다. 이 값은 관측 좌표로 사용되지 않는다. 이 정책을 채택하면 #16/#17 코드를
수정하지 않고 연결할 수 있다. common51을 직접 학습하려면 #16의 고정53 형상 검사,
어깨 인덱스와 feature 폭, 모델 input_size·체크포인트 계약의 후속 변경이 필요하다.

### API 및 연결 방법

- `parse_keypoint_json`은 실제 스키마의 배열과 confidence·형상·수치를 검증한다.
- `CrowdKeypointArchive`는 ZIP 제한, 파일명·프레임 ID, CRC와 순차 읽기를 제공한다.
- `iter_frame_results`는 오류를 고정 코드로 반환해 다음 파일 검사를 계속한다.
- `SourceSpec`, `conversion_blockers`는 명세 누락을 명시적으로 차단한다.
- `adapt_crowd_sequence`는 확인된 source로 common51 좌표·mask·시간축을 생성한다.
- `adapt_mediapipe53`은 기존 정규화 좌표에서 동일한 common51을 선택한다.
- `adapt_labeled_sequence`는 #12 제공 exact-video label_lookup을 받아 private text와 연결한다.
- `audit_archive`는 전체 검증과 label ID coverage를 비식별 집계한다.

source.coordinates는 pixels_top_left 또는 normalized_top_left를 지원하며 x는 오른쪽,
y는 아래쪽이다. image_size는 그 좌표 canvas의 실제 width,height다. fit_16x9와 같은
round·정수 padding 규칙을 적용한다. 이미 반전된 영상이나 image-left/right 필드는
handedness_unverified로 차단한다. 손을 자동 교환하지 않는다.

confidence>0이며 threshold 이상인 관절만 mask=True다. 좌표0 자체는 결측이 아니다.
관측된 좌표가 명시한 canvas를 벗어나면 차단한다. 빈 그룹은 관측을 생성하지 않는다.
전체 결측 시퀀스도 진단용으로 표현할 수 있으며 #16의 학습 sample 정책에서 제외해야 한다.
빈 시퀀스, NaN/Inf, 자료형·shape 오류를 거부하고 원본 배열을 수정하지 않는다.

fps 또는 timestamps를 반드시 받는다. FPS를 지정하면 frame_index/fps로 시간을 만들며
번호가 건너뛴 구간의 경과 시간을 보존한다. timestamps를 제공하면 실제 시각을 우선한다.
자동 15fps resampling과 보간은 적용하지 않으며 #16의 시계열 전처리와 연결한다.
명시한 expected_frame_count가 파일 수와 다르면 차단한다.

#12/#16/#17의 미병합 코드를 복사하거나 런타임 의존성으로 추가하지 않았다.
label_lookup은 #12의 검증된 records에서 video_id로 조회하도록 호출자가 제공한다.
FS만으로 join하지 않는다. `to_ctc53`의 xy/mask/timestamps를 #16에 전달한 뒤
#16의 feature와 batch를 #17의 float32 [B,T,F], input_lengths 계약에 연결한다.

### 실행 및 안전 한계

```powershell
$taskPython = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $taskPython tools/audit_crowd_keypoints.py '<실제 키포인트 ZIP 경로>' --inventory-root '<crowd 루트>'
& $taskPython -m pytest -q tests/test_crowd_keypoints.py tests/test_crowd_keypoint_adapter.py tests/test_audit_crowd_keypoints.py tests/test_crowd_dataset_validation.py
& $taskPython -u -c "import torch,pytest; torch.set_num_threads(2); raise SystemExit(pytest.main(['-q']))"
```

sign-language Conda 환경의 Python 3.12.16, NumPy 1.26.4를 사용한다.
JSON당 65,536 bytes, 전체 8GiB, 1,000,000 member, 압축률100 제한을 기본 적용한다.
ZIP 중앙 목록은 zipfile이 먼저 메모리에 올리므로 해당 할당까지 제한하지 못한다.
중첩 ZIP과 스키마가 다른 경로는 member_name으로 거부한다. payload를 디스크에 추출하거나
실제 좌표·정답·ID를 저장하지 않는다. audit CLI는 파서·목록·파일 오류를 고정 코드로 출력한다.

### 남은 확인 사항

source canvas 크기, export 좌표계, 실제 관절 명세·handedness와 파일별 시간 근거가 필요하다.
공식 데이터셋의 30fps 설명을 개별 파일의 FPS 증거로 자동 대입하지 않았다.
JSON confidence 이상 처리 정책과 53슬롯의 구조적 눈 결측 학습 정책은 팀 검토가 필요하다.
공식 수행자 분리, 실제 영상-라벨 내용 대조, 실데이터 학습과 CER는 후속 작업이다.
Refs #2 범위의 파서·조건부 어댑터·검증 PR로 제출할 수 있다.

### Commit 2 최종 테스트 및 독립 리뷰

신규 테스트는 55개이며 Commit 1 목록 검사20개와 기존 회귀27개를 포함해
**102 passed, 2 warnings, 62 subtests passed**였다. 전체 실행 시간은 94.92초였다.
경고2개는 기존 google._upb의 Python 확장 타입 deprecation 경고다.
실데이터 검증은 ZIP 전체 읽기와 별도 구조 조사로 수행했으며 합성 변환 테스트와 구분했다.

별도 리뷰 에이전트가 all-missing 통계의 Infinity 직렬화, mixed boolean 시간/프레임 입력,
감사 CLI의 파일·목록 오류 처리, 극단 FPS overflow를 재현했다. 모두 수정하고 회귀 테스트를
추가했으며 재리뷰에서 잔여 필수 수정 결함은 발견되지 않았다. 사람의 리뷰는 미실행이다.

GitHub에서 Issue #2 및 PR #11/#12/#13/#15/#16/#17의 변경 파일 목록을 재확인했다.
신규 파일의 경로 중복은 없으며 Commit 1 코드와 기존 팀원 파일은 보존했다.
현재 main에 선행 PR들이 반영되지 않아 실제 CTC 통합은 인터페이스 수준으로 검증했다.
새 브랜치 생성, Commit, Push, PR 생성, Merge는 수행하지 않았다.

### Commit 2 변경 파일 및 함수 위치

아래 경로는 저장소 루트 D:/UNIVERSITY/UAI_project2/FILE2/sign-language-recognition 기준이다.

| 파일 | 변경 행 |
| --- | --- |
| src/fingerspelling/crowd_keypoints.py | 1–217 (신규) |
| src/fingerspelling/crowd_keypoint_adapter.py | 1–263 (신규) |
| tools/audit_crowd_keypoints.py | 1–115 (신규) |
| tests/test_crowd_keypoints.py | 1–218 (신규) |
| tests/test_crowd_keypoint_adapter.py | 1–226 (신규) |
| tests/test_audit_crowd_keypoints.py | 1–90 (신규) |
| docs/fingerspelling_crowd_keypoints.md | 1–237 (Commit 2 추가), 전체 1–400 |
| docs/fingerspelling_crowd_keypoints_pr.md | 1–120 (시작 시 미추적 초안 갱신) |

| 모듈 | 공개 함수·클래스 | 실제 행 |
| --- | --- | --- |
| src/fingerspelling/crowd_keypoints.py | KeypointError | 26–27 |
| src/fingerspelling/crowd_keypoints.py | KeypointFrame | 44–49 |
| src/fingerspelling/crowd_keypoints.py | parse_keypoint_json | 52–101 |
| src/fingerspelling/crowd_keypoints.py | SequenceIndex | 105–116 |
| src/fingerspelling/crowd_keypoints.py | CrowdKeypointArchive | 119–217 |
| src/fingerspelling/crowd_keypoint_adapter.py | SourceSpec | 23–35 |
| src/fingerspelling/crowd_keypoint_adapter.py | conversion_blockers | 38–57 |
| src/fingerspelling/crowd_keypoint_adapter.py | CommonLandmarks | 112–117 |
| src/fingerspelling/crowd_keypoint_adapter.py | adapt_crowd_sequence | 134–206 |
| src/fingerspelling/crowd_keypoint_adapter.py | adapt_mediapipe53 | 209–229 |
| src/fingerspelling/crowd_keypoint_adapter.py | to_ctc53 | 232–247 |
| src/fingerspelling/crowd_keypoint_adapter.py | adapt_labeled_sequence | 250–263 |
| tools/audit_crowd_keypoints.py | file_digest | 21–26 |
| tools/audit_crowd_keypoints.py | audit_archive | 29–96 |
| tools/audit_crowd_keypoints.py | main | 99–111 |

## Commit 1 기록

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
