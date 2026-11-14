# Issue #2 성능 진단 및 제한된 개선 실험

2026-11-14, `feat/2-mediapipe-fingerspelling`에서 진행했다.
새 브랜치·가상환경은 만들지 않았다. 모든 Python 명령은 기존 Conda
`D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe`로 실행했다.
Python 3.12.16 / PyTorch 2.10.0+cu128 / MediaPipe 0.10.21 /
NumPy 1.26.4 / OpenCV 4.11.0 환경을 유지했으며 패키지를 설치하지 않았다.

## 재사용과 보호

원본 MP4 2,000개 중 기존에 선택했던 64개·성공 특징 59개만 재사용했다.
특징 NPZ 59개, 고유 영상 59개, Train 39 / Validation 12 / Test 8이다.
이전 56개 관찰의 경로·시점은 확인되지 않았으며 현재 59개와 해시·ID 목록이 일치한다.
새로운 영상 추출이나 기존 특징·체크포인트의 덮어쓰기는 없었다.

기존 `pts-run-01`의 파일 79개를 실험 시작·종료에 SHA-256 대조해 일치했다.
입력 특징마다 기존 training-config 해시 및 공통 특징 함수 재계산을 검증했다.
새 결과는 저장소 외부의 다음 폴더에만 썼다.

`D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance\controlled-01`

학습 전 디스크 여유 205,753,032,704바이트, GPU 여유 7,385,120,768바이트를 확인했다.
GPU는 NVIDIA GeForce RTX 5060 Laptop GPU다. Python TEMP/TMP도 D 경로로 지정했다.
C드라이브를 프로젝트 저장소·검색·백업·실험 출력 대상으로 사용하지 않았다.

## 진단과 가설

기존 모델을 같은 특징에서 다시 평가해 Train 자모 CER 0.809524, Validation 0.85를 재현했다.
Train은 D260/I0/S12, 336자모 중 272오류다. 평균 예측 길이 1.9487,
blank argmax 비율 0.963109, 프레임 평균 blank 확률 0.850948이었다.
Validation 예측은 한 종류, Train도 두 종류여서 출력 다양성이 붕괴했다.

Train에는 51개 자모 중 24종, 336토큰만 나타난다. Validation에는 Train에 없는
토큰 2개(120토큰 중)가 있다. 작은 데이터·길이·빈도 편중·특징 스케일 모두 영향을
줄 수 있다. 높은 blank 비율만으로 원인을 blank라고 확정하지 않았다.
기존 PTS/결측/CTC 최소 길이 검사를 재사용했고, 좌표·라벨을 임의 교정하지 않았다.

각 후보는 seed 20261114, 동일 영상·타깃·분할, BiGRU hidden64/2층,
입력424, batch8, AdamW weight_decay .01, 40epoch를 유지했다.
Validation 자모 CER → Validation loss → 이른 epoch로 체크포인트를 선택했다.
후보 모델은 Validation 자모 CER → 한글 CER → 이름 순으로 선택했다.
이미 과거에 관찰한 Test는 후보·렌더러 선택에 사용하지 않았다.

1. dropout-zero: dropout .2→0, lr .001 유지. 작은 데이터에서 정규화 강도가 원인인지 확인.
2. lr-three: dropout0 그대로 lr .001→.003. 제한된 업데이트에서 최적화 속도 확인.
3. train-affine: 두 번째 설정에 Train 실제 프레임의 채널별 평균/표준편차를 추가.
   std 최솟값 .05로 최대 증폭을 20배로 제한한다. Validation/Test 통계는 사용하지 않는다.

## 실제 학습 결과

아래 CER·Loss는 각 선택 체크포인트에서의 값이다. 실행은 모두 40epoch까지 완료했다.
시간은 각 epoch의 학습·Validation loss·Train/Validation CER 계산 합계이며,
동시에 회귀 테스트가 실행돼 단독 GPU 벤치마크로 해석하지 않는다.

| 후보 | 선택 epoch | Train loss | Val loss | Train 자모 CER | Val 자모 CER | 학습·평가 시간(초) |
|---|---:|---:|---:|---:|---:|---:|
| 기존 dropout .2 / lr .001 | 40 | 2.560547 | 2.662598 | 0.809524 | 0.850000 | 72.74(이전 실행) |
| dropout-zero | 40 | 2.544110 | 2.654202 | 0.809524 | 0.850000 | 48.70 |
| lr-three | 24 | 2.496962 | 2.638856 | 0.809524 | 0.850000 | 48.73 |
| train-affine | 38 | 0.947116 | 2.730064 | 0.583333 | 0.758333 | 48.59 |

dropout 제거와 학습률 변경만으로는 Validation CER 개선이 없었다.
표준화 후보는 Val 오류가 102→91개로 감소했다. Train은 D196/I0/S0,
Validation은 D82/I0/S9다. Val 예측 종류는 1→8, 평균 길이 2→3.1667,
blank argmax는 0.969504→0.931385로 변했다. 삭제 감소와 출력 다양성 증가는
관찰되지만, 서로 상관된 작은 표본·한 seed 결과로 일반적 인과 효과를 확정하지 않는다.
표준화 후보의 Val loss가 기존보다 높다는 점도 숨기지 않는다.

## 한글 출력과 부분 조합

표준화 후보의 기존 strict 렌더러 한글 CER는 Train 0.863309 / Val 0.956522로 악화했다.
잘못된 자모가 하나라도 있으면 전체를 원시 자모로 반환하는 정책의 영향도 조사했다.
기존 로컬 부분 조합 구현에서 문법 함수만 재사용하고 Git 기반 동적 의존성을 제거했다.
`resync`는 유효 음절만 조합하고 나머지 자모를 순서대로 남긴다.
빠진 자모·ㅇ을 생성하거나 사전으로 정답을 추측하지 않는다.
모든 실제 예측에서 재분해한 자모가 원래 예측과 동일함을 검사했다.

모드 선택은 Validation 한글 CER만 사용했고 기본 `strict`는 바꾸지 않았다.

| 한글 CER | 기존 모델 strict | 표준화 strict | 표준화 resync |
|---|---:|---:|---:|
| Train | 0.776978 | 0.863309 | 0.705036 |
| Validation | 0.826087 | 0.956522 | 0.804348 |
| 기존에 관찰한 Test 진단 | 0.840000 | 1.000000 | 0.840000 |

Validation 차이는 46글자 중 오류 1개(38→37)에 불과하며 Test 한글 CER는 개선되지 않았다.
모든 조건에서 완전 일치율은 0%다. **실사용 수준의 모델을 확보했다고 판단하지 않는다.**

모델 선택 후 Test를 진단했다. 이후 렌더러 선택 계산에는 Validation만 사용했다.
Test가 이미 개발 중 관찰된 상태라는 사실은 달라지지 않는다. 자모 CER 0.825397,
D42/I0/S10/N63, 평균 예측 길이 2.625, blank argmax 0.938326이다.
기존 자모 CER 0.873016보다 낮지만 이미 개발 중 관찰한 Test이며 수행자 독립 평가가 아니다.

## 추론 호환성과 복구

표준화를 첫 Linear에 `W'=W/scale`, `b'=b-W'@mean`으로 합쳐 기존 424차원 입력을 유지했다.
Validation 전체에서 원래 표준화 모델과 합친 모델의 logits/argmax를 비교했다.
최대 절대 차이는 0.000012279, argmax는 같았다. CPU Reader 재로딩도 확인했다.
기존 public 함수에 실제 후보 체크포인트·resync를 지정해 CPU/GPU 예측 일치를 확인했다.

`training/last.pt`는 학습 재개용, `inference.pt`는 추론 전용이다.
추론 파일은 `deployment_only=True`, 빈 optimizer state로 잘못된 재학습 재개를 거부한다.
학습 재개는 config·스키마·사전·데이터 해시가 같은 경우만 허용한다.
독립 리뷰에서 발견한 last.pt 직후 중단/selected JSON 중단 문제를 보완했다.
현재 epoch의 Validation을 다시 평가하고 selected.pt를 기준으로 JSON을 복구한다.
복구 중 모델 구성은 학습 RNG를 소비하지 않는다. 완료 후보 재사용도 파일 해시를 검사한다.

실제 3후보 학습은 중단 없이 완료됐으며 복구 수정 전 소스 SHA는
`b01dc9cf61402df9e7e32488919bd473e3dcddfeaa4a7ea357634a47b3a782cc`다.
실행 원본은 외부 결과 폴더 `runner-executed.py`로 보존했다.
이후 복구 보완은 정상 실행의 학습 수식·기존 측정 결과를 바꾸지 않는다.
새 코드로 다시 실험할 때에는 새 출력 폴더를 지정한다.

## 라벨·수행자 근거와 완료 한계

[AI-Hub 공식 자료](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)는
크라우드소싱 언어제공자 21명과 형태소 `name`이 표제어라는 점을 설명한다.
하지만 로컬 CROWD18/19와 실제 개인의 대응이나 개별 영상의 철자 순서를 확정할 근거는 아니다.
실제 형태소 JSON 2,000개에는 metaData url/name/duration/exportedOn,
구간 start/end와 attributes name이 있으며 수행자 검증·자모별 전사 필드는 없다.
[설명서 링크](https://aihub.or.kr/file/down.do?dataMnlFileSn=7969&dataSetSn=103&fileSn=7969)와
[구축활용가이드 링크](https://aihub.or.kr/file/down.do?cnstcPrcuseFileSn=18941&dataSetSn=103&fileSn=18941)는
이번 웹 열람에서 내용이 반환되지 않아 확인했다고 주장하지 않는다.

확인된 수행자 수는 미확정이며, CROWD 두 그룹을 서로 다른 사람으로 임의 분류하지 않는다.
라벨 검토 상태와 수행자 독립 CER는 미완료다. 전문가의 실제 영상 대조 및 공식 수행자 매핑이 필요하다.
기존 원본 영상의 VFR에 따른 선택 간격 불균일과 max_gap의 시간 의미도 계속 남는다.
새 모델로 기본 체크포인트를 자동 변경하지 않았고 실시간 루프도 수정하지 않았다.
Issue #2 전체 완료를 선언하지 않는다.

## 재현

```powershell
$repo = 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$py = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$crowd = 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
Set-Location -LiteralPath $repo
# 새 출력 경로/임시 폴더를 지정하고 TEMP/TMP도 D에 설정
$env:TEMP = "$crowd\outputs\mediapipe-ctc-performance\controlled-01\temp"
$env:TMP = $env:TEMP
& $py tools/experiment_mediapipe_performance.py `
  --source "$crowd\outputs\mediapipe-ctc-v1\pts-run-01" `
  --output "$crowd\outputs\mediapipe-ctc-performance\controlled-new" --epochs 40
& $py -m pytest -q -p no:cacheprovider `
  --basetemp "$crowd\outputs\mediapipe-ctc-performance\tests-new-unique"
```

예측별 원문·자모·영상 ID·모델은 외부 결과에만 있고 Git에는 코드·집계 문서만 포함한다.
검토된 변경만 현재 브랜치에 Commit하며 Force Push·PR 생성·Merge는 하지 않는다.

## 최종 검증과 독립 리뷰

전체 회귀 **371 passed, 2 warnings, 104.56초**. 실제 명령은 위와 같으며
`--basetemp ...\mediapipe-ctc-performance\regression-final-02`를 사용했다.
로그는 `mediapipe-ctc-performance/regression-final-02.log`다.
경고 2개는 기존 protobuf deprecation이다. 새 테스트는 27개다.
중간 실행의 권한 오류는 코드 실패가 아니라 외부 D 테스트 폴더 생성 제한이었고,
허용된 경로 권한으로 다시 실행해 통과했다. C 경로로 우회하지 않았다.
독립 리뷰의 선택 체크포인트 재개 지적을 수정하고 실제 파일 중단 상황을 합성 테스트했다.
`last.pt` 직후 중단, 선택 JSON 불일치, 학습 RNG 보존, 추론 파일 재학습 거부를 포함한다.
기존 소스·체크포인트·원본 ZIP은 삭제하거나 교체하지 않았다.
최종 독립 리뷰에서 모든 후보·렌더러의 CER 전수 합산과 예측 자모 왕복,
371개 테스트 로그·복구 동작·문서 수치를 대조해 일치함을 확인했다.
기존 미추적 실험 파일을 제외하고 Commit 대상 소스만 복사한 폴더에서도
두 지문자 테스트 파일 **78 passed, 2 warnings, 14.80초**를 확인했다.
로그: `mediapipe-ctc-performance/clean-validation-01/tests.log`.
