# Issue #2 실제 성능 개선 실험 — 2026-11-15

Validation 자모 CER **37/120=.308333 → 28/120=.233333**. 절대 .075 감소,
상대 24.324% 감소다. 동일 seed 재학습에서 선택 epoch와 가중치 텐서가 완전히 일치했다.
이는 기존 영상/문자열 그룹 분할의 탐색적 결과이며 수행자 독립 성능이 아니다.

## Why / 기존 구현과 비교 기준

기존 `95d53b3`의 143개 NPZ, Train123/Validation12/Test8을 그대로 사용했다.
기존 최선은 `expanded-models-01/expanded-hand-local/inference.pt`, epoch39,
BiGRU hidden64/2층, 입력424, 출력52(blank0), dropout0, batch8,
AdamW(lr.003, weight_decay.01), seed20261114, 총40epoch다.
첫 Linear에 Train 통계 표준화가 접혀 있으며 손목 전역 위치·손 상대좌표·delta를 쓴다.
Baseline의 실제 재평가는 .308333, D21/I3/S13/N120으로 기존 보고값을 재현했다.

[Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2)와
PR #11~#41의 구현·본문을 확인했다. #11~#20은 자모/라벨/CTC/학습 기반,
#21~#23은 독립 근거·실시간·평가 검증, #24~#30은 MediaPipe/조건화/추론,
#31~#34는 확장·손 표현·실측, #35~#41은 검토·계약·의존성 문서다.
#14는 별도 숫자 보조헤드다. 기존 기능을 복제하거나 기존 PR을 수정하지 않았다.
현재 브랜치는 기존 미병합 구현에 의존하므로 main에 바로 병합 가능한 독립 패치가 아니다.

## What / 제한된 실험 결과

| 실험 | 변경 사항 | Validation 자모 CER | 기존 대비 CER | D/I/S | 상태 |
|---|---|---:|---:|---|---|
| Baseline | 기존 보고값 .3083, 실제 체크포인트 재평가 | .308333 | 기준 | 21/3/13 | 재현 |
| 저비용1 | blank logit bias -.3, 학습 없음 | .300000 | -.008333 | 21/2/13 | 성공, 가중치 배포 안 함 |
| 저비용2 | blank logit bias +.3, 학습 없음 | .325000 | +.016667 | 24/2/13 | 악화 |
| 실험1 | 기존 가중치 미세조정 lr.0003, 20epoch | .275000 | -.033333 | 19/1/13 | epoch12 선택, 반복 재현 |
| 실험2 | 기존 가중치 미세조정 lr.001, 20epoch | **.233333** | **-.075000** | **13/2/13** | epoch14 선택, 반복 재현 |

모든 미세조정은 fresh AdamW, weight_decay.01, gradient_clip5, batch8,
dropout0, seed20261114다. 각 후보는 같은 원본 최선 가중치에서 독립 시작한다.
모델/특징 크기와 데이터 양을 고정했다. Validation CER→loss→이른 epoch 순으로 선택한다.
Test는 배열·정답을 로드하지 않고, 기존 분할 메타데이터만 누수 검사에 사용했다.
Train만 optimizer에 전달되며 Validation은 no_grad 평가다. 고정20epoch 예산을 끝까지 실행했다.
새 후보 탐색은 2개/40epoch, 동일 설정 재현40epoch로 총80epoch 후 종료했다.

## How / 근거와 해석

선택 모델의 Train CER는 .046829(48/1025), Validation은 .233333이다.
개선 9개 오류 중 삭제8개·삽입1개 감소, 치환13개는 유지됐다.
따라서 삭제 감소가 직접 관측된 개선 근거다. 낮은 학습률의 추가 최적화가
기존 가중치를 개선했다는 결과이며 특징 변경이나 데이터 증가의 인과 효과를 주장하지 않는다.
blank bias -.3의 개선은 1개뿐이며 최선 미세조정보다 작다.
동일 seed 반복은 재현성 근거이나 다중 seed 안정성·통계적 유의성 근거가 아니다.
Validation은 12개/120자모로 작고 여러 기존 실험에서 관찰됐으므로 선택 편향이 남는다.

손 상대좌표+기존 전역 손목, 프레임 delta, 좌우21점 분리 표현은 기존 구현을 유지했다.
hidden/층수 확대는 작은 데이터의 과적합과 비용 때문에 이번 묶음에서 보류했다.
dropout/weight decay/증강 변경은 효과 혼합을 피하고 학습률만 비교하려고 보류했다.
기존 미추적 beam/어휘 디코더는 보호하고 재개발하지 않았다. 이번에는 사전 없는 blank 보정만 실행했다.
Train 자모 빈도를 `baseline.json`에 저장했다. CTC 임의 class weighting은 적용하지 않았다.
다음 우선순위는 검증된 라벨·수행자 매핑 확보, 다중 seed, 증강/정규화 단일변수 비교다.
데이터를 더 많이 학습하면 반드시 좋아진다고 가정하지 않는다.

## Where / 실행 명령과 비공개 결과

저장소 루트 `D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition`에서
신규 `feat/2-performance-20261115`, `.tmp-issue2/performance-20261115/worktree`를 생성했다.
다음 명령은 해당 Worktree에서 실행했다. 기존 출력은 덮어쓰지 않으며 새 output만 허용한다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$area='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\performance-20261115'
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance'
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'
& $py -u tools/run_issue2_performance_batch.py --source "$data\data-expansion-01" --checkpoint "$data\expanded-models-01\expanded-hand-local\inference.pt" --output "$area\run-02"
& $py -u tools/run_issue2_performance_batch.py --source "$data\data-expansion-01" --checkpoint "$data\expanded-models-01\expanded-hand-local\inference.pt" --output "$area\run-03"
& $py -m pytest -q -p no:cacheprovider --basetemp "$area\pytest-01"
& $py -m pytest -q -p no:cacheprovider tests/test_issue2_performance_batch.py tests/test_mediapipe_performance.py tests/test_fingerspelling_hand_features.py --basetemp "$area\pytest-final"
```

`run-02`는 첫 완료 묶음, `run-03`은 보호/검증·메타데이터 보완 후 최종 코드 재현이다.
최선 파일은 `$area\run-03\finetune-lr001\improved.pt`다.
`run-03/{baseline,decoding,summary}.json`, 각 후보 `{config,history,result}.json`에
명령 인자에 해당하는 경로·seed·설정·스키마·해시·epoch·D/I/S·실행 상태가 기록된다.
각 `training/last.pt,best.pt`는 학습 복구용이며 Validation CER 최선 배포 파일과 구분한다.
개선 checkpoint만 `improved.pt`로 저장한다. 기존 모델은 덮어쓰지 않았다.
원시 좌표 API·독립 편집거리·반복 가중치 비교는 `$area\audit_local.py`를 지정 Python으로 실행했다.
`verification-local.json`에 실제 예측 사례와 자모별 편집 정렬 오류가 있다.
이 파일들은 원문/영상ID를 포함하므로 Git에 넣지 않았다. 정렬 동률에 따라 오류 유형 분해는 달라질 수 있다.

## 보호·환경·실패 기록

시작 상태와 refs는 `$area/status-before.txt`, `refs-before.txt`,
미추적35개 해시는 `protected-untracked.json`에 기록했다. 35개 모두 변경 없음이다.
기존 143NPZ를 포함한 동결 데이터·체크포인트157항목 SHA가 종료 후 일치했다.
원본 영상·ZIP과 팀원1 웹캠 코드는 수정하지 않았다. 기존 브랜치·커밋·PR도 수정하지 않았다.
기존 일부 임시 테스트 디렉터리는 접근 거부되어 전체 재귀 목록을 완전 감사했다고 주장하지 않는다.
초기 별도 run-01은 출력 리다이렉션 상태에서 진행 확인이 안 돼 중단했고 모델 결과가 없다.
로컬 감사의 중복 keyword와 UTF-8 읽기 오류는 수정했다. 실험 수치는 성공한 run-02/03만 썼다.
GPU RTX5060 Laptop, torch2.10.0+cu128, Python3.12.16, CPU CTC loss/고정 RNG다.
시작 여유 공간 약204.6GB, GPU 가용 약7.38GB였다. 환경 설치/교체는 하지 않았다.
deepcopy 후 GRU contiguous 경고가 발생했지만 실행·재현·CPU/GPU 일치를 확인했다.

## 독립 리뷰·검증·한계

별도 리뷰 에이전트가 누수, 차원/자료형, 반복 포함 CTC 길이, GPU/CPU,
체크포인트 계약, API 회귀, micro CER를 읽기 전용으로 검토했다. 최종 blocker 없음.
지적에 따라 출력 경로 겹침 사전 거부, hand-local 스키마 일치, Test 포함 메타데이터
그룹 검사, CPU/GPU 개별 영상 예측·D/I/S 일치를 보완했다.
동일 ID/FS/NFC문자열 및 Train/Validation 완전 동일 특징 중복이 없음을 검사했다.
유사 영상·동일 수행자 누수까지 검증한 것은 아니다.
재현2회 가중치 bitwise 일치, 공개 raw xy/mask Reader12개 일치,
독립 Levenshtein 거리 재계산 통과. 전체 회귀155 passed/2 warnings,
최종 변경 후 관련 회귀44 passed/2 warnings다. 경고는 기존 protobuf deprecation이다.
원본 main/다른 Worktree의 미추적 테스트까지 모두 실행했다는 의미는 아니다.
기존 API와 웹캠 통합 파일은 수정하지 않았다.
수행자 독립 CER, 전체 원본 영상·라벨 의미 전문가 검증, 다중 seed,
실시간 연속 입력 성능은 미해결이다. Issue #2를 닫지 않는다.
AI 사용: Codex가 구현·실제 학습 실행·검증·문서 작성, 별도 Codex 에이전트가 코드 리뷰했다.
사람 전문가 라벨 확인과 외부 인간 리뷰를 대신하지 않는다.
