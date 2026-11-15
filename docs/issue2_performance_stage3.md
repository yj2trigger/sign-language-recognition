# Issue #2 Stage 3 실제 GPU 실험 — 2026-11-15

최저 Validation CER **27/120=.225000 → 23/120=.191667**.
동일 seed 재학습에서 가중치·예측이 완전히 일치했다. 단일 seed에서 .20 미만을 달성했다.
하지만 다른 3seed는 .258333/.266667/.233333이므로 모든 seed에서 이 목표를 달성한 것은 아니다.
평균 개선은 관찰했으나 변동성이 증가했다. 수행자 독립·통계적 유의성을 주장하지 않는다.

## Why / Git 이력과 의존성

시작 시 `git status`, `show-ref`, `log -1 --format='%H %P %s'`, `merge-base`,
`rev-list --left-right --count`, `worktree list`, 실제 `git ls-remote --heads origin`을 확인했다.
아래 3개 HEAD는 로컬·원격이 정확히 일치했다. Stage3는 로컬·원격 모두 없었다.

| 브랜치 | 로컬/원격 HEAD | 실제 부모 |
|---|---|---|
| feat/2-mediapipe-fingerspelling | 95d53b362c4ce117ce6bdb15ae012d710aec0edb | 651983003edb29389dfaaa6836e6d5ca715f7690 |
| feat/2-performance-20261115 | 62f7b6e3bbb16eeeae3ece1956085e878f5e1d6d | 95d53b362c4ce117ce6bdb15ae012d710aec0edb |
| feat/2-performance-stage2 | 1bd4705d64fc7492a170d43833a732f62d02821e | 62f7b6e3bbb16eeeae3ece1956085e878f5e1d6d |

분기 구조: `6519830 → 95d53b3 → 62f7b6e(PR57) → 1bd4705(Stage2) → Stage3 신규 커밋`.
mediapipe...1차는 merge-base95d53b3, ahead/behind0/1;
1차...Stage2는 merge-base62f7b6e, 0/1; mediapipe...Stage2는0/2; main...Stage2는0/7이었다.
신규 `feat/2-performance-stage3`는 정확히1bd4705에서 생성했고 새 커밋의 부모도1bd4705다.
독립 Worktree는 `.tmp-issue2/performance-stage3/worktree`다. 이력 변경 명령은 사용하지 않았다.
[PR57](https://github.com/yj2trigger/sign-language-recognition/pull/57)은 base95d53b3/head62f7b6e의 미병합 Draft다.
Stage2 브랜치의 PR 검색 결과는0건이었다. 신규 PR은 Stage2와 그 선행 미병합 구현에 의존한다.
기존 추적 파일의 사용자 수정은 없었다. 루트 미추적35개를 보호했고 팀원 웹캠 코드는 수정하지 않았다.
이번 기존 파일 변경은 학습기/시간증강기/합성테스트에 한정된다. main 통합 전 선행 PR과 재검증이 필요하다.

## What / 기존 시간 증강의 대응 seed 검증

모든 새 학습은 동일 원본 `expanded-models-01/expanded-hand-local/inference.pt`에서 독립 시작했다.
원본 SHA256은 `0ec9ded244354bf00ede986e66d7619b6b19d6b413996f6129f6b1815329e3f5`다.
lr.001/20epoch/fresh AdamW/clip5/batch8/dropout0, hidden64/2층 BiGRU,424차원/52클래스를 유지했다.
기존 시간 증강은 raw xy/mask Uniform[.9,1.1] 단조 재표본화 후 모든 특징·delta 재계산이다.
Train123/Validation12/Test8은 고정이고 Test 배열은 읽지 않았다. Validation 분모는120자모다.
선택은 CER→Validation loss→이른 epoch다. 시간은20epoch 학습·증강·평가·체크포인트 IO의 합계다.

| 설정 | seed | Train CER | Validation CER | D/I/S | epoch | 시간(초) | 상태 |
|---|---:|---:|---:|---|---:|---:|---|
| 기존 시간증강 wd.01 | 20261114 | .013659 | .225000 | 11/4/12 | 17 | 37.52 | 기존 결과 재평가 |
| 동일 | 42 | .028293 | .266667 | 14/1/17 | 11 | 32.67 | GPU 성공 |
| 동일 | 123 | .026341 | .266667 | 12/3/17 | 12 | 32.35 | GPU 성공 |
| 동일 | 2026 | .019512 | .266667 | 11/2/19 | 12 | 32.76 | GPU 성공 |

기존 미증강 CER는 seed20261114/42/123/2026 순서로 .233333/.275000/.266667/.241667이다.
증강 대응 차이는 −.008333/−.008333/0/+.025000: 2개 개선·1개 동률·1개 악화다.

| 설정(4seed) | 평균 CER | 표본 표준편차(ddof1) |
|---|---:|---:|
| 기존 미증강 | .254167 | .019837 |
| 기존 시간증강 wd.01 | .256250 | .020833 |
| 시간증강 wd.03 | .237500 | .033679 |

시간 증강만으로 안정적인 개선은 확인하지 못했다. 이 값은 각 seed의 선택 epoch를 요약한 기술통계다.
안정성 seed는 추가 설정 선택 pool에 넣지 않고 사전 고정 seed20261114에서 후보를 비교했다.
안정성 seed의 최저값도.266667이므로 더 좋은 관측 모델을 누락한 경우는 없다.

## How / 오류 분석과 최대3개 후보의 근거

기존 .225 모델의 삭제11개는 ㅣ/ㅏ/ㅓ/ㅕ/ㅅ 각2개, ㅇ1개였다.
치환12개는 각각1회로 반복적인 특정 치환 쌍을 확정할 표본이 부족하다.
예: ㅣ→ㅐ, ㅝ→ㅏ/ㅜ, ㅅ→ㅆ/ㅠ/ㅐ. 원문 정답과 개별예측은 비공개 `analysis-local.json`에만 저장했다.
짧은6클립 CER15/58=.258621, 긴6클립12/62=.193548이었다.
높은 motion 대리값6클립18/64=.281250, 낮은6클립9/56=.160714였다.
이는 실제 자모 구간이나 검증된 손 속도가 아니다. 클립 길이·특징 변화량 대리값의 관찰이다.
인접 반복 자모를 포함한2클립 모두 삭제1개가 있었고, 같은 토큰의 blank 분리 재출현은 그중1클립이었다.
CTC blank/반복 관계의 증상이며 독립 시간 정답이 없어 실제 반복 자모 삭제 원인을 확정하지 않는다.
blank가 nonblank보다0~.3 높았던 프레임 비율 평균은 .001225였다.
nonblank290프레임 중8개에서 상위2 logits 차이가.5미만이었다. ㅗ/ㅡ는2프레임,
ㅐ/ㅣ·ㅜ/ㅝ·ㅂ/ㅍ 등은1프레임이었다. 이는 모델의 음향적 모호성이지 검증된 수형 혼동은 아니다.
평균 가용 손관절 비율은1.0이지만 보간을 포함하므로 실제 관측·수형 분리 품질이 완벽하다는 뜻은 아니다.
Train .013659와 Validation .225의 큰 격차로 과적합 가능성도 남았다.

후보계획은 분석 후 `candidate-plan.json`에 고정했고 이후 확대하지 않았다.
A: 짧고 높은 motion 클립의 오류와 seed2026 악화 때문에 속도 변형을 약하게 [.95,1.05]로 제한했다.
B: Train/Val 격차에 대응해 wd만 .01→.03으로 높였다. 초기화·증강·lr 등은 고정했다.
C: 기존 blank−.3의 무이득과 작은 logit margin 구간을 고려해 bias 탐색을 반복하지 않고,
기존 사전 없는 CTC prefix beam 폭10/정확 CTC 확률 재순위만 사용했다. 정답은 디코더 입력이 아니다.

| 후보(seed20261114) | Validation CER | 기존 .225 대비 | D/I/S | epoch | 시간(초) | 상태 |
|---|---:|---:|---|---:|---:|---|
| A 약한 시간증강 | .275000 | +.050000 | 19/1/13 | 7 | 32.81 | 악화 |
| B 시간증강+wd.03 | **.191667** | **−.033333** | **2/6/15** | 13 | 31.62 | 개선·재현 |
| C 기존 모델 beam10 | .216667 | −.008333 | 8/5/13 | 17 | .59 | 디코딩만, CPU/GPU 재확인 |

최선은4자모 개선, 상대14.815% 감소다. 삭제9개 감소와 삽입2/치환3개 증가가 합쳐진 결과다.
새 구조·dropout.1 재실험·bias grid·정답 사후보정은 사용하지 않았다. 열세 후보로 자동 교체하지 않는다.
동률 시 기존 모델을 유지한다. 추가 seed는 새 후보가 아니라 선택 설정의 대응 안정성 검증이다.

| 선택 wd.03 | Train CER | Validation CER | D/I/S | epoch | 시간(초) | 상태 |
|---|---:|---:|---|---:|---:|---|
| seed20261114 | .024390 | .191667 | 2/6/15 | 13 | 31.62 | GPU 성공·동일seed재현 |
| seed42 | .017561 | .258333 | 11/4/16 | 16 | 31.18 | GPU 성공 |
| seed123 | .017561 | .266667 | 14/2/16 | 16 | 31.89 | GPU 성공 |
| seed2026 | .015610 | .233333 | 11/3/14 | 16 | 31.69 | GPU 성공 |

기존 시간증강과 대응해 3seed 개선·1동률, 평균 .018750 개선이다. 그러나 표준편차는 증가했다.
다른3seed는 .225보다도 높아 .191667을 안정적으로 기대할 수 없다. 단일seed 성과와 구분한다.
총180epoch(시간증강seed60+학습후보40+최선seed60+재현20) 후 종료했다.

## Where / 정확한 체크포인트와 SHA256

최선 경로: `D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\performance-stage3\run-01\decay03\improved.pt`.
기존 .225와 .233333 모델은 그대로 보존했다. 아래 경로는 stage3/run-01 기준이다.

| 파일 | SHA256 |
|---|---|
| temporal-seed-42/improved.pt | 7c53d852a3266bfb12c4727a480751f79ff549c9f4fa5598d52836e511bf0c5c |
| temporal-seed-123/improved.pt | 196820681f90470943168d8e827667be6d97a062721868ecef980fb85c123f56 |
| temporal-seed-2026/improved.pt | c3460cf046cce7d2e36253041d5c794de9e647265010d63a8ace6303ab1e0327 |
| weaker-temporal/improved.pt | ba8bf5b9eb308945160f83c13528ebbe51ed774374c1ff336338cf5f4466a914 |
| decay03/improved.pt | bd3a010ff4f2eb95393ce7503939f006b154f41acf1500fc47818236e9f3113a |
| selected-seed-42/improved.pt | 5643f6678e3c9212a54fd30b897d95a34477e103573429ec394211b14ea97457 |
| selected-seed-123/improved.pt | 44dd70ae9f14526765784088403d5fc860ab9aaab5a0d1286be67e88e6d09a8f |
| selected-seed-2026/improved.pt | 1ece94a54ac0ee23b08f8dea4b038f358563f400d2f80ceb7d7dbf3c1d81da5b |

기존 시간증강 seed20261114/beam의 공통 checkpoint SHA는
`903c409d7f93cb7dad45249dde4f036f6ed31d9a5a33b80c53f093fcaad639a9`다.
`manifest/seeds/selection/stability/verification.json`, 학습별 `config/history/result.json`에
정확한 설정·seed·분할·시간·성공상태·초기화/결과SHA를 보존했다. 원문 포함 결과는 모두 로컬 전용이다.

## 실제 실행 명령

아래 인자로 각 phase를 개별 실행했다. 새실험 재현 시 output은 존재하지 않는 새 경로로 바꾼다.
analyze 후 아래 JSON을 UTF-8/BOM없이 `$area/run-01/candidate-plan.json`에 저장했다.
`{"A":{"strength":[0.95,1.05]},"B":{"weight_decay":0.03},"C":{"beam_width":10}}`

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$area="$repo\.tmp-issue2\performance-stage3"
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance'
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'
Set-Location "$area\worktree"
$inputs=@('--source',"$data\data-expansion-01",'--baseline',"$data\expanded-models-01\expanded-hand-local\inference.pt",'--reference',"$repo\.tmp-issue2\performance-stage2\run-01\temporal\improved.pt",'--stage2',"$repo\.tmp-issue2\performance-stage2\run-01",'--output',"$area\run-01")
& $py -u tools/run_issue2_stage3.py --phase seeds @inputs
& $py -u tools/run_issue2_stage3.py --phase analyze @inputs
& $py -u tools/run_issue2_stage3.py --phase candidates @inputs
& $py -u tools/run_issue2_stage3.py --phase stability @inputs
& $py -u tools/run_issue2_stage3.py --phase verify @inputs
& $py tools/inspect_stage3_acoustics.py --source "$data\data-expansion-01" --checkpoint "$repo\.tmp-issue2\performance-stage2\run-01\temporal\improved.pt" --output "$area\run-01\acoustics-local.json"
& $py -m pytest -q -p no:cacheprovider --basetemp "$area\pytest-final"
& $py "$area\audit_local.py"
& $py "$area\protect_local.py" before
& $py "$area\protect_local.py" after
```

보호 before는 실제로 학습 전에 실행했다. Python3.12.16/torch2.10.0+cu128/RTX5060 Laptop,
기존 Conda만 사용했고 모든 출력은D다. 환경 설치·교체는 하지 않았다. main phase 시작에서 자원도 검사했다.

## 검증·독립 리뷰·보호

Train만 역전파/증강하고 Test는 메타데이터 그룹검사만 사용했다. video/FS/NFC문자열 중복,
Train/Val 동일NPZ/특징, float32/424차원, int64정답/반복CTC최소길이, schema를 기존 구현으로 검사했다.
phase별 실제 CLI경로·학습기 포함 구현SHA·원본SHA를 고정하고 변경을 거부한다.
각 학습모델의 CPU/GPU 개별예측 일치, beam CPU/GPU 일치, 기존/신규 checkpoint 로드를 확인했다.
선택 모델 재학습은 weights bitwise/개별예측 일치, 공개 raw xy/mask Reader12개도 일치했다.
기존 API·웹캠은 변경하지 않았다. beam은 실험용 CLI이며 공개 Reader 기본 greedy를 바꾸지 않았다.
이전 미추적 decoder와 alignment를 새로 개발하지 않고 원문 그대로 추적범위에 재사용했다.
decoder SHA동일/align함수AST동일이다. 기존 원본 미추적 파일은 변경하지 않았다.
초기 가져온 테스트의 미추적 모듈 의존성으로6실패/192통과했다. 5개 사례의 의존성을 기존 추적API로
교체하고, 이번 범위 밖의 다른 미추적 실행기 전용 n-best1개 테스트는 가져오지 않았다.
decoder exhaustive-path 검증은 유지했다. 수정 후 관련42통과, 전체 **197 passed/2 warnings,106.33초**다.
경고는 기존 protobuf deprecation이다. PowerShell 보조ps1은 실행정책으로 실패해 직접 Python으로 실행했고 정책은 바꾸지 않았다.
별도 Codex 에이전트가 누수/길이/schema/초기화/단일변수/CPU·GPU/측정/실제결과SHA를 독립 검토해 blocker 없음.
원본 MP4/ZIP/NPZ/PT/JSON과 기존 미추적35개 등10,366항목(약25.1GB)의 시작/종료 SHA256가 일치했다.
명시 보호범위는 crowd 아래 해당 확장자와 stage1/run-03,stage2/run-01,미추적35개다. 모든 과거temp를 감사했다는 뜻은 아니다.
기존 Git refs도 시작 목록과 대조한다. 기존 브랜치·PR #11~#57·팀원 작업은 수정하지 않는다.

## 한계와 다음 우선순위

Validation12개/120자모를 여러 차례 관찰·선택해 낙관편향이 있다. seed들은 독립 새데이터가 아니다.
3seed 대응개선도 통계적 유의성·수행자 독립·실용적 일반화를 입증하지 않는다. 일부 seed의 절대 개선은1자모다.
원본 영상 의미/자모별시각/수행자ID는 독립 검증되지 않았다. 실제 수형 혼동과 빠른 동작 원인 확정도 미완료다.
추가 grid 확대보다 라벨·수행자 검증과 유효한 새 평가 설계가 우선이다. 이후 사전고정 다중seed와
늘어난 삽입/치환 오류를 검토한다. 모든seed .20미만·실시간 연속입력·독립CER는 미완료다.
AI사용: Codex가 구현·실제GPU학습·검증·문서, 별도 Codex에이전트가 코드/결과 리뷰했다. 인간 전문가 검증의 대체가 아니다.
이번에는 신규브랜치 Commit/일반Push만 수행하며 PR생성·수정·Merge·Issue종료는 하지 않는다. Refs #2.
