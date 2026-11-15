# Issue #2 stage2 실제 GPU 실험 — 2026-11-15

기존 Validation CER **28/120=.233333 → 27/120=.225000**.
시간 증강으로 1개 오류 감소(절대 .008333, 상대3.571%)했다.
동일 seed 재학습에서 선택 가중치와 예측이 완전히 일치했다. .20 미만은 달성하지 못했다.
작은 Validation에서의 탐색적 개선이며 통계적 유의성·수행자 독립 일반화를 뜻하지 않는다.

## Why / 기존 작업 확인

[PR #57](https://github.com/yj2trigger/sign-language-recognition/pull/57)의 정확한 head
`62f7b6e3bbb16eeeae3ece1956085e878f5e1d6d`에서 신규 `feat/2-performance-stage2`를 만들었다.
독립 Worktree는 `.tmp-issue2/performance-stage2/worktree`다.
기존 보고서·학습기·특징·오류 분석을 읽고 기존 CTCTrainer/score/fine_tune/training_data를 재사용했다.
학습·정답·특징 파이프라인을 다시 구현하지 않았다. 기존 PR #11~#57은 수정하지 않았다.

원본 초기화는 `expanded-models-01/expanded-hand-local/inference.pt`,
SHA256 `0ec9ded244354bf00ede986e66d7619b6b19d6b413996f6129f6b1815329e3f5`다.
모든 seed와 학습 후보가 여기서 독립 시작한다. 이전 미세조정 최선 가중치에서 이어 학습하지 않는다.
기존 비교 모델은 `.tmp-issue2/performance-20261115/run-03/finetune-lr001/improved.pt`다.
이전 결과 파일을 읽고 실제 GPU 재평가로 개별 Validation 예측을 대조했다.

## What / 다중 seed 결과

lr=.001, 20epoch, fresh AdamW(weight_decay=.01, clip5), batch8,
dropout0, BiGRU hidden64/2층, 입력424/출력52(blank0), 손 상대좌표·delta를 유지했다.
Train123/Validation12/Test8 분할 그대로다. Test 배열은 로드하지 않았다.
모든 수치의 Validation 분모는 120자모이며 CER→Validation loss→이른 epoch로 선택했다.

| seed | Validation CER | D/I/S | 선택 epoch | 상태 |
|---|---:|---|---:|---|
| 20261114 | .233333 | 13/2/13 | 14 | PR57 결과 사용·실제 재평가 |
| 42 | .275000 | 17/2/14 | 13 | GPU 학습·CPU/GPU 일치 |
| 123 | .266667 | 16/2/14 | 18 | GPU 학습·CPU/GPU 일치 |
| 2026 | .241667 | 8/3/18 | 16 | GPU 학습·CPU/GPU 일치 |

4 seed 평균 **.254167**, 표본 표준편차(ddof1) **.019837**,
모집단 표준편차(ddof0) .017180이다. 추가3개만의 평균 .261111, 표본 표준편차 .017347이다.
이는 각 seed에서 선택한 최선 epoch의 기술통계이며 독립 Test 성능 분포가 아니다.
추가 seed 모두 기존 단일 최선보다 나빴다. seed 민감성이 있어 안정적 .233333 성능이라고 단정할 수 없다.

## How / 최대3개 후보의 단일변수 비교

기존 오류 분석에서 삭제가13개였고, 자모별 삭제에는 ㅅ4개·ㅏ3개·ㅍ2개가 있었다.
모델·데이터를 늘리지 않고 정규화/시간 변화/삭제 편향을 확인하는 세 후보를 사전 고정했다.
새 학습 후보는 seed20261114·원본 초기화를 고정해 dropout 또는 시간 증강만 변경했다.

| 후보 | 변경 변수 | CER | 기존 .233333 대비 | D/I/S | epoch |
|---|---|---:|---:|---|---:|
| dropout01 | dropout 0→.1 | .258333 | +.025000 | 13/2/16 | 16 |
| temporal | Train 시간 증강만 추가 | **.225000** | **-.008333** | 11/4/12 | 17 |
| blank-minus03 | temporal 최선의 blank bias −.3만 추가 | **.225000** | **-.008333** | 10/4/13 | 17 |

blank 후보는 parent temporal 대비 CER 변화0이다. 삭제1개 감소/치환1개 증가가 상쇄됐다.
전체 모델 선택은 CER→선택 epoch→체크포인트 경로 사전 고정 동률규칙이다.
동률에서 `blank-minus03/candidate.pt`가 선택됐다. bias 자체의 추가 개선을 주장하지 않는다.
시간 증강 원본 `temporal/improved.pt`도 같은 최선 CER이며 둘 다 보존했다.
dropout은 치환 오류가 늘어 실패했다. 시간 증강은 삭제2개·치환1개 감소/삽입2개 증가로 순1개 개선이다.
다중 seed 실험 결과를 보고도 추가 seed 검색·학습률 탐색·bias grid 확대는 하지 않았다.
새 seed60epoch + 두 학습후보40epoch + 동일설정 재현20epoch, 총120epoch로 종료했다.

시간 증강은 Train의 raw xy/mask를 배율 Uniform[.9,1.1]로 단조 nearest resample한다.
순서·양끝점·정답을 유지하고 반복 자모를 포함한 CTC 최소길이를 보장한다.
보간·어깨 정규화·손 상대좌표·delta를 원시 좌표에서 전부 다시 계산한다.
Validation은 원본 그대로이며 증강하지 않는다. 배율1의 전처리를 실제 Train123개에서 대조해
기존 특징과 bitwise 일치(max_abs0)를 확인했다.
이는 의미 보존을 의도한 제한적 속도 증강이며 전문가 영상 전사로 의미 보존을 입증한 것은 아니다.

## Where / 체크포인트·명령·설정

아래는 실제 실행한 세 phase 명령과 같은 인자다. 신규 output만 허용하며 phase 순서대로 실행한다.
Python/환경 설치·교체는 하지 않았다. 모든 결과와 임시 파일은 D드라이브다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$area="$repo\.tmp-issue2\performance-stage2"
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance'
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'
Set-Location "$area\worktree"
& $py -u tools/run_issue2_stage2.py --phase seeds --source "$data\data-expansion-01" --baseline "$data\expanded-models-01\expanded-hand-local\inference.pt" --reference "$repo\.tmp-issue2\performance-20261115\run-03\finetune-lr001\improved.pt" --output "$area\run-01"
& $py -u tools/run_issue2_stage2.py --phase candidates --source "$data\data-expansion-01" --baseline "$data\expanded-models-01\expanded-hand-local\inference.pt" --reference "$repo\.tmp-issue2\performance-20261115\run-03\finetune-lr001\improved.pt" --output "$area\run-01"
& $py -u tools/run_issue2_stage2.py --phase verify --source "$data\data-expansion-01" --baseline "$data\expanded-models-01\expanded-hand-local\inference.pt" --reference "$repo\.tmp-issue2\performance-20261115\run-03\finetune-lr001\improved.pt" --output "$area\run-01"
& $py -m pytest -q -p no:cacheprovider --basetemp "$area\pytest-01"
& $py "$area\audit_local.py"
```

선택 모델: `$area\run-01\blank-minus03\candidate.pt`.
동일 CER의 보정 없는 모델: `$area\run-01\temporal\improved.pt`.
재현 가중치는 `repeat-selected/improved.pt`, `repeat-selected/with-bias.pt`에 저장했다.
`manifest.json`에는 원본SHA·후보계획·구현SHA·경로·GPU/디스크 자원,
`seeds.json`에는 전체seed/통계, `selection.json`에는 후보/선택, `verification.json`에는 재현 결과가 있다.
각 학습의 `config/history/result.json`에는 seed·설정·초기화SHA·epoch·D/I/S·성공 상태가 있다.
`training/last.pt,best.pt`는 복구/Validation loss용, `improved.pt`는 원본 CER 개선 시 CER 선택용이다.
기존 학습기 의미를 보존하므로 `improved`는 .308333 원본 대비이고 stage2 목표 대비는 selection.json에 있다.
선택 모델 Train CER .010732(11/1025)로 Validation과 격차가 크다. 과적합 가능성이 남는다.
원본·NPZ·라벨·개별예측·학습결과JSON·checkpoint는 GitHub에 업로드하지 않는다.

## 보호·누수·CPU/GPU·독립 리뷰

시작 refs와 미추적35개 파일 SHA를 stage2 작업영역에 저장했다.
종료 시 35파일과 동결 데이터/기존run-03/원본checkpoint172항목 SHA 일치를 확인했다.
원본 영상/ZIP에 쓰지 않았고, 기존143NPZ·라벨·최선checkpoint를 보존했다.
일부 과거 temp는 알려진 권한 제한이 있어 모든 디렉터리를 재귀 감사했다고 주장하지 않는다.
Train/Validation/Test 메타데이터의 video/FS/NFC문자열, Train/Validation 동일NPZ/특징을 검사했다.
동일 수행자나 유사 영상 누수를 배제할 수행자ID는 미검증이다.
Train만 역전파하고 Validation은 eval/no_grad다. Test는 선택·평가 모두 사용하지 않았다.

전체 학습 후보와 blank 모델의 CPU/GPU 개별예측·D/I/S 일치를 확인했다.
시간 증강 모델·blank 보정 모델의 재학습 선택weights 모두 bitwise 일치,
개별 Validation 예측 일치, raw xy/mask 공개Reader12개, 독립 Levenshtein 계산을 확인했다.
전체 회귀 **162 passed, 2 warnings, 125.56초**. 경고는 기존 protobuf deprecation이다.
이는 독립Worktree의 전체 테스트이며 다른Worktree의 미추적 실험 테스트는 포함하지 않는다.
기존 공개API와 팀원1 웹캠 코드는 수정하지 않았다.

별도 Codex 리뷰 에이전트가 코드와 저장 결과를 읽기 전용으로 검사했다.
누수, float32/424차원, packed int64/CTC길이, seed초기화, dropout로드,
증강 delta재계산, CPU/GPU, CER재집계, checkpointSHA, 재현 결과에서 blocker 없음.
향후 방어 강화로 phase별 CLI경로와 manifest 대조, 재사용 학습기SHA의 phase간 고정을 제안받았다.
이번에는 같은CLI와 변경 없는 구현으로 전체phase를 실행했고 기록SHA·실제경로도 대조했다.
AI사용: Codex가 구현/실행/테스트/보고, 독립 Codex에이전트가 리뷰했다.
사람 전문가의 영상·라벨 의미 검증과 외부 인간 리뷰를 대신하지 않는다.

## 한계·다음 우선순위

.225는 1자모 개선으로 Validation 반복관찰/선택편향이 남는다. 증강 효과의 다중seed 검증은 미실시다.
기존4seed 평균 .254167은 단일최선 .233333보다 나빠 견고한 일반화 개선을 확인했다고 할 수 없다.
다음은 라벨 의미·수행자 매핑 검증, 유효한 독립평가 설계, 이후
사전 고정 다중seed 증강 비교와 삽입/삭제 분석을 우선한다.
.20미만·수행자 독립CER·실시간 연속입력·전체 원본영상 전사검증은 미달성이다. Issue #2를 닫지 않는다.
신규PR생성/Merge는 하지 않고 검증된 코드/테스트/집계문서만 stage2에 Commit/일반Push한다.
PR57과 미병합 Issue2 기반에 의존하므로 main 직접제출 전 선행구현 통합과 재검증이 필요하다.
