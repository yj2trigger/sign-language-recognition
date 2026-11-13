# MediaPipe 지문자 CTC 구현 및 검증

2026-11-14. Issue #2, `feat/2-mediapipe-fingerspelling`.
기준 HEAD: `1aac6a14a5a26e09a94d1080cccfd6b8ac542023`.

## 범위와 기존 작업 보존

기존 OpenPose 실험과 새 MediaPipe 실험을 별도로 관리한다.
기존 미추적 파일 34개는 다음 위치에 복사한 뒤 각각 SHA-256을 대조했다.

`D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\issue2-predevelopment-backup\20261114-014156`

기존 팀원의 `landmarks.py`, `preprocess.py`, `realtime.py`는 수정하지 않았다.
기존 개발 단계에서는 Commit하지 않았다. 후속 선택적 Commit 지시에 따라
이번 문서와 필요한 코드·테스트만 커밋한다. Push·PR 생성·Merge·브랜치 이동은 하지 않는다.

## 재사용한 코드

이번 Commit에 포함하는 PR #11·#13·#15·#16·#17 모듈 12개는
기존 Git 커밋의 소스를 바이트 단위로 그대로 가져왔다.
실행 시 `git show`에 의존하지 않는 패키지다. 기존 실험용 동적 로더는 보존한다.
개발 당시 함께 복사했던 `ctc_batching.py`는 현재 실행 경로에 필요하지 않아
Commit에서 제외하고 로컬에 보존한다. 정확한 포함 목록은 `provenance.md`에 기록한다.

- `fingerspelling_video.py`: 실제 디코딩 PTS 검사 및 선택, `landmarks.process_frame()` 재사용.
- `fingerspelling_runtime.py`: 공통 특징·토큰 사전·모델 로더·공개 함수.
- `tools/run_mediapipe_ctc.py`: 영상별 CRC 추출, 재개 기록, 특징 검증, 학습·평가.
- `tests/test_mediapipe_fingerspelling.py`: 정상·경계·극단·재개·보존 검사.

## 환경

기존 `sign-language` Conda 환경을 유지했다.
Python 3.12.16, PyTorch 2.10.0+cu128, NumPy 1.26.4,
OpenCV 4.11.0, MediaPipe 0.10.21. `pip check`: 충돌 없음.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU, CUDA 사용 가능.
패키지 설치·업그레이드는 하지 않았다. 첫 Holistic complexity=0 호출에서
MediaPipe가 기존 Conda 폴더에 `pose_landmark_lite.tflite` 모델 자산을 자동 다운로드했다.

## 원본 및 라벨 연결

원본: `original-video/downloads/수어 영상/2.Validation/[원천]01_crowd_video.zip`.
22,443,361,135바이트, ZIP 엔트리 2,003개, MP4 2,000개.
전체 ZIP 압축 해제와 전체 엔트리 CRC 검사는 수행하지 않는다.
선택한 MP4를 완전히 읽을 때 해당 엔트리 CRC를 검사한다.

형태소 JSON 2,000개와 영상 ID를 연결했다. 한글 자모 타깃으로 인코딩 가능한
후보는 1,616개이며 숫자 정책이 필요한 382개, 기타 미지원 문자 2개를 제외했다.
이는 기존 OpenPose 특징 품질 검사까지 통과한 1,606개와 다른 모집단이다.

일부 MP4 이름의 `_F_.mp4`와 라벨의 `_F.mp4` 차이는 FS·CROWD·방향을
엄격한 정규식으로 추출해 유일한 경우만 연결한다. 원래 ZIP 경로와 별칭 여부를
각 표본에 보존하며, 같은 정규 ID가 둘 이상이면 중단한다.
라벨 원문을 교정하지 않는다. 모든 신규 표본의 의미 검증 상태는 `unverified`다.
기존 검토 CSV의 uncertain 상태도 그대로다.

## 시간축에서 발견한 문제와 보완

중단 전 두 출력 폴더에는 각각 9개와 18개의 특징이 존재했고 학습 체크포인트는 없었다.
초기 처리물은 그대로 보존했다. 평균 FPS를 사용한 시각은 한 실제 영상에서
디코딩 PTS와 최대 약 0.178초 차이가 났다. 프레임 간격은 0.03333·0.05·0.13333초로
가변적이었다. 따라서 그 방식의 VFR 특징을 새 학습에 채택하지 않았다.

새 추출기는 전체 영상의 단조·유한 PTS와 실제 디코딩 프레임 수를 검사한다.
15Hz tick마다 그 시각 이상의 첫 실제 프레임을 선택하고 중복 프레임을 제거한다.
두 번째 디코딩에서 선택 프레임의 PTS가 첫 검사와 같은지 확인한다.
라벨 `[start,end)` 구간은 실제 원본 PTS로 자른다. 좌우 반전은 하지 않는다.
OpenCV의 기본 영상 회전 처리를 사용하고 `process_frame()`의 16:9 패딩을 유지한다.

**VFR에서는 목표 15Hz 선택이며 실제 선택 시각은 완전히 균일하지 않을 수 있다.**
NPZ의 `t`와 진행 기록의 최대 간격 오차를 함께 보관한다.
PR #16 특징은 초 단위 velocity가 아닌 프레임 delta를 사용한다.
결측 보간 `max_gap=3`은 선택 프레임 3개를 뜻하며 정확히 0.2초를 보장하지 않는다.
완전히 규칙적인 15fps로의 좌표 보간은 이번 구현에 포함하지 않았다.

기존 CFR 특징은 실제 전체 PTS 검사, 원본·추출 코드·버전·특징 스키마 대조,
NPZ SHA 및 타깃 검사, 선택 시각 일치를 모두 통과한 경우만 재사용한다.

## 특징과 추론 계약

입력 `xy (T,53,2)`와 boolean `mask (T,53)`, 기본 15fps이다.
관절 순서는 상체 11개 + 왼손 21개 + 오른손 21개로 원본 순서를 유지한다.
어깨는 인덱스 5·6, 손목은 11·32다. MediaPipe가 제공한 좌우 의미를 그대로 사용한다.
이 계약이 원본 영상의 해부학적 좌우를 전문가가 확인했다는 뜻은 아니다.

PR #16 어깨 중심·너비 정규화, 16:9 화면비 보정, 짧은 결측 보간을 사용한다.
관절별 x/y, observed, interpolated, available, dx/dy, motion_valid의 8채널,
총 424차원이며 시간축을 64프레임으로 줄이지 않는다.
손목의 어깨 기준 이동과 원래 관절 유효성 마스크가 유지된다.
기존 OpenPose A138/B241 가중치는 이 424차원 입력에 연결하지 않는다.

토큰은 기존 Unicode 호환 자모 51개와 blank=0, 총 52클래스다.
이는 실제 지문자 37종의 공식 동작 토큰 정의를 검증한 결과가 아니다.
타깃의 인접 반복을 고려한 CTC 최소 길이를 검사한다.
예측은 기존 Greedy CTC 디코더와 한글 조합기를 사용한다.
조합이 불가능한 예측은 자모를 보존하며 누락 문자를 임의로 보충하지 않는다.

```python
# 기존 src/realtime.py처럼 src가 import 경로인 환경
from fingerspelling_runtime import FingerspellingReader
reader = FingerspellingReader(r"D:\...\pts-run-01\checkpoints\best.pt")
text = reader(xy, mask)

# 공개 함수: 시작 시 FINGERSPELLING_CHECKPOINT 환경변수 설정
from fingerspelling_runtime import read_fingerspelling
text = read_fingerspelling(xy, mask)
```

모델은 경로별 캐시로 재사용하므로 불변 체크포인트 경로를 지정한다.
빈 입력·한 프레임·양손 미검출은 빈 문자열로 처리한다.
잘못된 차원·마스크·NaN/Inf·극단 좌표·없는 모델·다른 vocabulary/스키마는 오류로 차단한다.
정규화 가능한 어깨가 없는 경우 특징 생성은 거부하며, 공개 Reader는 이를 빈 문자열로 처리한다.

## 학습·평가와 한계

이번 실행은 마감 내 처리 비용과 파이프라인을 검증하는 최대 64영상 탐색 실험이다.
실제 데이터가 있는 전체 2,000영상을 처리했다고 주장하지 않는다.
동일 FS 또는 NFC 정답 문자열로 연결된 표본을 같은 분할에 두어
Train/Validation/Test 간 중복을 방지한다. CROWD18/19는 확인된 수행자 ID가 아니다.
**수행자 독립 평가는 미완료다.** 라벨과 손동작의 의미적 일치도 검증되지 않았다.

기존 PR #17 BiGRU CTC, hidden=64, 2층, dropout=.2,
AdamW lr=.001, weight_decay=.01, batch=8, seed=20261114, 40epoch.
학습 전 실제 데이터에서 finite forward/backward를 확인한다.
체크포인트는 Validation loss로 선택하며 Test는 선택에 사용하지 않는다.
Train/Validation/Test의 자모 CER, 한글 CER, D/I/S, 완전 일치율을 구분한다.
추론 시간은 batch1의 이미 준비된 특징→모델→디코딩·조합 시간이다.
영상 추출 시간이나 실시간 웹캠 전체 지연을 뜻하지 않는다.

## 장애 대응

완료 표본의 SHA·배열·타깃·공통 전처리 재현을 검사한 뒤 재사용한다.
원본/코드/라이브러리/설정이 달라지면 같은 출력 폴더의 재개를 거부한다.
학습 재개 시 특징 해시·분할·사전·모델·epoch 설정도 대조한다.
원본 ZIP 보호 검사는 읽기 전용 접근과 size/mtime 대조이며 전체 파일 해시 검사가 아니다.

소유 기록이 없는 임시 MP4는 건드리지 않는다. 소유가 확인된 미완성 복사본은
고유 `.partial-*` 이름으로 보존하고 해당 멤버만 다시 추출한다.
정상 디코딩·라벨 연결·특징 저장·재읽기 검증·완료 기록 후에만 이번 작업의 MP4를 삭제한다.
실패한 MP4는 보존하고 다른 영상을 계속 처리한다.

## 재현 명령

PowerShell에서 기존 Conda Python을 사용한다. 같은 코드·설정이면 완료 단계부터 재개한다.
출력 폴더의 설정이 다르면 새 폴더를 지정하고 기존 결과를 보존한다.

```powershell
$py = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$crowd = 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
& $py tools/run_mediapipe_ctc.py `
  --videos "$crowd\original-video\downloads\수어 영상\2.Validation\[원천]01_crowd_video.zip" `
  --labels "$crowd\수어 영상\2.Validation\[라벨]01_crowd_morpheme.zip" `
  --output "$crowd\outputs\mediapipe-ctc-v1\pts-run-01" `
  --stage all --max-videos 64 --extraction-seconds 900 --epochs 40
& $py -m pytest -q -p no:cacheprovider `
  --basetemp "$crowd\outputs\mediapipe-ctc-v1\tests-new-unique-folder"
```

`.pytest_cache` 접근 경고에는 권한 변경이나 삭제 없이 pytest 캐시 비활성화를 사용한다.
같은 `--basetemp` 경로의 이전 파일을 pytest가 정리하므로 매 실행 고유 폴더를 사용한다.

## 검증 기록

신규 정상·경계·극단·CFR/VFR·별칭·중복·재개·원본 보존 테스트 49개 통과.
중간 전체 회귀 실행은 338개 통과, protobuf 관련 deprecation warning 2개였다.
최종 실행 수치 및 독립 리뷰 결과는 아래 실행 결과에 기록한다.

독립 리뷰에서 지적한 재개 원본 계약, 버전/landmarks 고정,
소유 partial 재시도, orphan 전체 배열 대조, NFC 동치 분할을 보완했다.
별도 검토자가 초기 신규 테스트 31개를 독립 실행하여 통과를 확인했고,
확대된 테스트 및 최종 실제 결과의 대조 결과는 아래 기록한다.

## 완료 조건

인터페이스와 전처리 코드의 구현 및 합성 검증은 완료했다.
실제 처리·학습·평가의 완료 여부는 최종 실행 결과에 한정한다.
라벨 의미·해부학적 좌우·수행자 독립 평가·실시간 성능은 미완료다.
Issue #2의 모든 완료 조건을 충족한 것으로 처리하지 않는다.

## 최종 실제 실행 결과

결과 위치:
`D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-v1\pts-run-01`.
`progress.json`, `training-config.json`, `history.json`, `evaluation.json`,
`forward-backward.json`, `validation-audit.json`, `resource-preflight.json`,
`integrity-all.json`, 특징 NPZ 59개와 `checkpoints/best.pt`, `checkpoints/last.pt`를 재읽었다.

64개 선정 중 59개 성공, 5개 제외. 제외 영상은 평균 FPS가 10.79~12.47로
현재 추출기의 15fps 이상 정책을 충족하지 않았다. 손상 영상으로 단정하지 않는다.
이 MP4들은 각 실행 폴더의 `temp/`에 보존했다.
다른 원본 영상 1,936개는 이번 탐색 실행에서 처리하지 않았다.
동작 의미 검증을 통과한 표본 수는 0개다.

기존 CFR 특징 6개를 검증 후 재사용했다. 이전 VFR 처리물도 보존했다.
성공 영상의 CRC 복사·디코딩·추출·검증 시간 합계는 595.51초다.
실패 영상 처리와 준비·학습 시간은 이 합계에서 제외한다.
특징 길이는 17~97프레임이며, 표본별 전체 관절 결측률의 단순 평균은 0.2053%다.
선택 프레임 간격은 최소 0.016667초, 중앙값 0.066667초, 최대 0.166667초다.
균일한 15fps 입력을 확보한 것으로 해석할 수 없다.

학습 전 디스크 여유 205,771,960,320바이트, GPU 여유 7,385,120,768바이트
(전체 8,546,484,224바이트)를 확인했다. 최종 특징 배열은 총 5,966,528바이트다.
특징 크기는 GPU 학습 전체 할당량이나 peak 메모리가 아니다.

RTX 5060 Laptop GPU에서 40epoch 학습했다. epoch별 시간 합계 72.74초,
Validation loss로 선택한 epoch는 40이다. 최종 Train loss 2.5605467588,
Validation loss 2.6625976563, 파라미터 158,452개, best checkpoint 1,941,186바이트다.
실제 forward/backward와 체크포인트 재로딩 일치 검사를 통과했다.

| 항목 | Train | Validation | Test |
|---|---:|---:|---:|
| 영상 수 | 39 | 12 | 8 |
| 기준 자모 토큰 수 | 336 | 120 | 63 |
| 삭제 | 260 | 96 | 47 |
| 삽입 | 0 | 0 | 0 |
| 치환 | 12 | 6 | 8 |
| 자모 CER | 0.809524 | 0.850000 | 0.873016 |
| 한글 CER | 0.776978 | 0.826087 | 0.840000 |
| 완전 일치율 | 0% | 0% | 0% |
| 특징→텍스트 평균 지연(ms) | 16.62 | 17.49 | 15.09 |

FS/동일 문자열 그룹 분할 결과이며 수행자 독립 평가가 아니다.
Validation·Test 예측은 각각 한 종류, Train 예측도 두 종류에 불과하다.
**출력 다양성 붕괴와 높은 삭제율이 있어 실제 사용에 필요한 성능을 확보하지 못했다.**
라벨 오류·데이터 규모·최적화 부족 중 하나를 원인으로 확정하지 않는다.
학습 초반 CER 1.0에서 Train 0.8095로 변했지만 실시간 인식 성공을 뜻하지 않는다.

최종 전체 회귀 테스트: **342 passed, 2 warnings, 135.03초**.
신규 테스트 49개 통과 로그도 별도로 존재한다. 경고는 기존 protobuf deprecation 2개다.
중간 수집 단계에서 발생한 테스트 괄호 오류는 수정하고 신규·전체 회귀를 재실행했다.

최종 독립 리뷰에서 59개 특징 재계산·SHA·분할·평가 순서,
40epoch history와 best/last의 모델 상태·스키마·사전, CER 전수 합산을 대조했다.
별도 CPU Reader가 실제 Test 표본에서 저장 예측을 재현했다.
필수 코드 결함은 발견하지 못했으며 출력 다양성 붕괴를 성능 한계로 지적했다.

보호 검증: 이전 학습 산출물 1,632개, 기존 미추적 파일 34개,
원본 키포인트·형태소 ZIP 2개, 기존 A/B 확률 캐시 2개의 SHA 일치.
원본 영상 ZIP은 읽기 전용 접근 및 시작·종료 size/mtime 일치 확인이며 전체 SHA/CRC 확인은 아니다.
기존 PR 모듈 13개와 원래 Git blob의 바이트 일치도 확인했다.
감사 도구가 처음 테스트용 형태소 ZIP을 고른 문제는 `outputs/`를 제외한
원본 폴더만 조사하도록 수정한 뒤 실제 원본 해시 일치를 확인했다.

학습 PowerShell 래퍼는 stderr 리다이렉션 상태에서 종료 코드 1을 반환했으나
최종 평가·재로딩·완료 상태와 `integrity-all.json` 기록까지 모두 존재한다.
실행 로그 `mediapipe-ctc-v1/pts-run-01-console.log`와 최종 산출물을 확인했으며,
전체 Python 처리 실패로 분류하지 않는다.

완료 여부:

- **구현·실제 검증 완료:** 입력 검사, PTS 기반 개별 영상 처리, 공통 특징,
  GPU CTC 학습·분할 평가, 체크포인트 로딩, 한글 반환, 회귀·독립 리뷰.
- **구현 완료·검증 미완료:** 전체 데이터로 확대한 학습, 운영 입력 안정성.
- **미완료:** 동작·라벨 의미 검증, 수행자 신원 확인과 수행자 독립 CER,
  균일 15fps 좌표 입력 검증, 실사용 모델 품질, 실시간 루프 통합.

라벨 의미·수행자 정보와 시간 간격 정책을 먼저 확정한 뒤, 확인된 표본을 늘려
학습·평가해야 한다. 현재 체크포인트를 기본 실사용 모델로 지정하지 않는다.

감사 재현(이미 존재하는 보고서는 덮어쓰지 않는다):

```powershell
& $py tools/audit_mediapipe_ctc.py `
  --run "$crowd\outputs\mediapipe-ctc-v1\pts-run-01" `
  --backup "$crowd\outputs\issue2-predevelopment-backup\20261114-014156"
```

직전 개발 종료 당시에는 기준 HEAD를 유지했고 새 파일은 22개였다.
후속 선택적 Commit에서는 사용하지 않는 `ctc_batching.py`를 제외한 21개를 선별한다.
기존 미추적 파일 전체를 일괄 stage하지 않는다.

## 선택적 Commit 전 데이터 재조사

`pts-run-01/features`의 직접 자식 NPZ 파일 59개, 파일명 기준 고유 영상 59개,
progress 완료 영상 59개, training-config 분할에 등장하는 고유 영상 59개가 일치한다.
모든 NPZ가 정상적으로 열리며 config·progress·현재 파일의 SHA-256이 전부 일치한다.
분할 참조 중 누락 파일 0개, 참조하지 않는 추가 NPZ 0개, 중복 영상 0개다.
**실제 Train은 39개이며 59개는 Train/Validation/Test를 합친 실험 표본 수다.**
이전에 관찰했다는 56개는 현재 이 폴더에서 재현하지 못했다.
과거 관찰의 경로·시점을 확인할 근거가 없어 당시 차이의 원인을 확정하지 않는다.
다른 개발 출력 폴더의 NPZ는 각각 9개와 18개이며 현재 실험 표본에 더해 세지 않는다.

공개 함수 직접 호출/설정 누락 테스트 2개를 추가했다.
기존 추적 코드 + 이번 선별 파일만 있는 독립 소스 폴더에서 신규 **51개 통과**.
실제 best 체크포인트를 공개 함수로 CPU 추론하여 저장 예측 일치를 확인했고,
같은 실제 표본의 특징 재계산·모델 캐시·체크포인트 SHA 불변성도 확인했다.
GPU 학습 결과는 기존 40epoch 기록을 재사용하며 새 GPU 학습은 하지 않았다.
관련 로그는 `mediapipe-ctc-v1/commit-validation-01/`에 보관한다.
Commit 전 전체 회귀를 다시 실행해 **344 passed, 2 warnings, 114.71초**를 확인했다.
선별 소스 테스트는 **51 passed, 2 warnings, 15.06초**다.
첫 선택적 Commit은 `d5b0035`이며 CTC 핵심 14파일·1,333줄 추가다.
12개 원본 모듈은 Staging된 Git blob까지 기존 PR 소스와 동일함을 대조했다.

선별 이유: 현재 실행 의존성이 있는 기존 PR 모듈 12개 및 package 초기화·출처,
MediaPipe runtime/video, run/audit 도구, 테스트, 두 문서만 포함한다.
감사 도구는 로컬 과거 실험·백업이 있을 때 사용하는 선택적 검증 도구이며,
인식기 실행에 과거 OpenPose 실험 파일이나 Git 이력을 요구하지 않는다.

### 기존 미추적 파일 분류

아래 `preserve`는 이번 PR에 불필요하지만 보존할 자료,
`separate`는 기존 실험 또는 별도 PR 범위다. 모두 삭제·수정·stage하지 않는다.

| File | Classification | Reason |
|---|---|---|
| `docs/crowd_full_fit.md` | preserve | Existing local report/review material |
| `docs/crowd_label_review.md` | preserve | Existing local report/review material |
| `docs/crowd_label_review_integration.md` | preserve | Existing local report/review material |
| `docs/crowd_label_semantics_validation.md` | preserve | Existing local report/review material |
| `docs/crowd_temporal_token_validation.md` | preserve | Existing local report/review material |
| `docs/ctc_deletion_diagnosis.md` | preserve | Existing local report/review material |
| `docs/fingerspelling_ab_experiment.md` | preserve | Existing local report/review material |
| `docs/fingerspelling_decoding_comparison.md` | preserve | Existing local report/review material |
| `docs/fingerspelling_error_analysis.md` | preserve | Existing local report/review material |
| `docs/label_review/crowd_label_correction_review_40.csv` | preserve | Existing local report/review material |
| `docs/label_review/crowd_label_overrides_verified.json` | preserve | Existing local report/review material |
| `docs/label_review/crowd_label_review_instructions.md` | preserve | Existing local report/review material |
| `src/crowd_label_overrides.py` | separate | OpenPose experiment or separate label/decoder review |
| `src/ctc_postprocessing.py` | separate | OpenPose experiment or separate label/decoder review |
| `src/fingerspelling_ab.py` | separate | OpenPose experiment or separate label/decoder review |
| `src/fingerspelling_partial_compose.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_crowd_label_overrides.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_crowd_label_review.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_crowd_temporal_labels.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_ctc_postprocessing.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_diagnose_ctc_deletions.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_fingerspelling_ab.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_fingerspelling_error_analysis.py` | separate | OpenPose experiment or separate label/decoder review |
| `tests/test_train_crowd_full.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/analyze_beam_recovery.py` | preserve | Unfinished cache analysis; outside MediaPipe scope |
| `tools/analyze_crowd_temporal_labels.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/analyze_fingerspelling_errors.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/benchmark_crowd_ab.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/build_crowd_label_review.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/compare_crowd_decoding.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/diagnose_ctc_deletions.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/run_crowd_ab.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/train_crowd_full.py` | separate | OpenPose experiment or separate label/decoder review |
| `tools/validate_crowd_label_reviews.py` | separate | OpenPose experiment or separate label/decoder review |
| `src/fingerspelling/ctc_batching.py` | separate | Existing PR #16 helper; not imported by this pipeline |
