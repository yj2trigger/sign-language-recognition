# Issue #2 기존 PR 의존성·병합 시뮬레이션 검증

Refs #2

2026-11-14에 GitHub API와 원격 ref를 다시 조회했다. 원래 17개 원격 브랜치와 보호 Commit 95d53b3를 보존했다.
이 문서는 이전 보고서에서 통합 용도와 실제 import 의존성을 혼합했던 #17/#18 설명을 정정한다.

## 현재 기존 PR

| PR | 상태 / mergeable | Base → Head | Commit | 파일 | + | - |
|---|---|---|---:|---:|---:|---:|
| [#11](https://github.com/yj2trigger/sign-language-recognition/pull/11) | open / True | `main` → `feat/2-jamo-utils` | 1 | 3 | 220 | 0 |
| [#12](https://github.com/yj2trigger/sign-language-recognition/pull/12) | open / True | `main` → `feat/2-label-preparation` | 2 | 9 | 1029 | 0 |
| [#13](https://github.com/yj2trigger/sign-language-recognition/pull/13) | open / True | `feat/2-label-preparation` → `feat/2-ctc-targets` | 1 | 6 | 483 | 0 |
| [#15](https://github.com/yj2trigger/sign-language-recognition/pull/15) | open / True | `main` → `feat/2-ctc-inference-evaluation` | 1 | 7 | 1032 | 0 |
| [#16](https://github.com/yj2trigger/sign-language-recognition/pull/16) | open / True | `main` → `feat/2-ctc-data-pipeline` | 1 | 9 | 1073 | 0 |
| [#17](https://github.com/yj2trigger/sign-language-recognition/pull/17) | open / True | `main` → `feat/2-fingerspelling-model` | 1 | 9 | 1045 | 0 |
| [#18](https://github.com/yj2trigger/sign-language-recognition/pull/18) | open / True | `main` → `feat/2-crowd-keypoint-adapter` | 2 | 9 | 1809 | 0 |
| [#19](https://github.com/yj2trigger/sign-language-recognition/pull/19) | open / True | `feat/2-crowd-keypoint-adapter` → `feat/2-crowd-source-validation` | 1 | 9 | 891 | 7 |
| [#20](https://github.com/yj2trigger/sign-language-recognition/pull/20) | open / True | `main` → `research/2-fingerspelling-training-integration` | 1 | 1 | 278 | 0 |

기존 PR #12/#13/#15/#16/#17/#18/#19의 추가량은 400줄을 넘는다. 기존 PR 변경은 소유자 승인 범위에서 별도 결정한다.

## 검증한 의존성

- Git ancestry에서 #12 Head가 #13의 선행 Commit이고 #18 Head가 #19의 선행 Commit이다.
- #13 TargetEncoder의 기본 분해는 #11 jamo를 지연 import한다. #11이 없을 때 missing_pr11_dependency, 추가하면 실제 인코딩 성공을 확인했다.
- #15 디코딩/CER, #16 시계열·배치, #17 모델·학습·checkpoint, #18 키포인트·어댑터는 각각 자기 PR 소스만 export한 격리 프로세스에서 import에 성공했다.
- #17에 #13/#15/#16을 먼저 Merge해야 한다는 강제 순서는 없다. #18도 외부 타깃·전처리를 호출자가 공급하는 구조다.
- #20은 분석 문서다. 다른 PR의 API import를 막는 선행 관계가 없다.
- MediaPipe runtime은 #11/#13/#15/#16/#17을 사용한다. #13의 Git Base 관계 때문에 #12가 함께 고려된다.
- OpenPose #18/#19와 설계 #20의 병합을 MediaPipe의 코드상 필수 조건으로 추가하지 않는다. 기존 비교 기준에는 이 PR들도 들어 있다.
- 확정된 간선은 #12→#13, 기본 인코딩의 #11→#13, #18→#19다. 이 그래프에는 순환이 없다.

기존 #13의 Base는 여전히 feat/2-label-preparation이고 #19의 Base는 feat/2-crowd-keypoint-adapter다.
선행 PR이 main에 병합돼도 기존 Base가 자동으로 main이 된다고 가정하지 않는다. 소유자가 승인한 Base 조정 또는 실제 main 기반 새 제출 경로가 필요하다.
Squash를 사용했다면 파일 내용 일치만으로 원본 Head가 main의 조상이라고 판단하지 않고 merge-base와 diff를 다시 검증한다.

## 기존 PR 시뮬레이션

시작 원격 main은 `c627cf77eb55564fb1a85ece63628d5bba67bb9a`다. 검증 순서 11→12→13→15→16→17→18→19→20은 가능한 위상 순서 하나다.
Git merge-tree의 실제 3-way 결과를 새 로컬 Commit object로 기록하고 detached Worktree에서 검사했다.
독립 PR들 사이에 이 실행 순서를 필수 의존성으로 부여하지 않는다.
충돌은 #11 뒤 #12 적용의 src/fingerspelling/__init__.py add/add 1건이었다.
두 원본 파일의 AST가 docstring 한 개뿐임을 검사하고, 시뮬레이션에서 #12 설명문을 선택했다. 두 원본 blob과 충돌 기록을 보존했다.
실제 Merge 시 이 설명문 선택은 리뷰어가 확인해야 한다. 이 선택에 따라 비교 기준과 tree가 일치했다.
기존 PR 통합 테스트는 `390 passed, 2 warnings, 254 subtests passed in 102.68s (0:01:42)`다.
첫 전체 실행에서는 기존 #17 CUDA checkpoint 테스트의 os.replace가 WinError 5로 한 번 실패했다. 같은 소스·새 임시 경로의 전체 재실행은 통과했다. 원인은 확정하지 않았고 실패 로그를 보존했다.
시뮬레이션 Commit `a1746d52fea6c75606ffffead0ccc91e2dca51af`, tree `6360ca58fc0042eb267cb96fba81dfc58d70a196`다.
비교 기준 Commit `1a0fa9cb90effe095109c8a466cb2bd00162cd31`는 main 한 개만 부모로 갖는다. 시뮬레이션은 원본 PR Head들을 부모 그래프에 포함한다.
두 결과의 tree diff는 0개이며 공통 조상은 `c627cf77eb55564fb1a85ece63628d5bba67bb9a`다.
따라서 내용 일치와 공식 PR 병합 ancestry는 구분해서 사용해야 한다. 비교 기준 자체를 main PR로 제출하지 않는다.

## 비교 기준 58개 파일의 실제 출처

| 파일 | 동일 blob을 제공한 기존 PR |
|---|---|
| `.gitignore` | #12 |
| `README.md` | #12 |
| `docs/crowd_source_validation.md` | #19 |
| `docs/ctc_targets.md` | #13 |
| `docs/fingerspelling_crowd_keypoints.md` | #19 |
| `docs/fingerspelling_data_pipeline.md` | #16 |
| `docs/fingerspelling_inference_evaluation.md` | #15 |
| `docs/fingerspelling_model.md` | #17 |
| `docs/fingerspelling_training_integration_plan.md` | #20 |
| `docs/label_preparation.md` | #12 |
| `docs/label_preparation_validation.md` | #12 |
| `src/fingerspelling/__init__.py` | #12 |
| `src/fingerspelling/cer_metrics.py` | #15 |
| `src/fingerspelling/crowd_dataset_validation.py` | #18 |
| `src/fingerspelling/crowd_keypoint_adapter.py` | #19 |
| `src/fingerspelling/crowd_keypoints.py` | #19 |
| `src/fingerspelling/crowd_labels.py` | #12 |
| `src/fingerspelling/crowd_source_validation.py` | #19 |
| `src/fingerspelling/ctc_batching.py` | #16 |
| `src/fingerspelling/ctc_checkpoint.py` | #17 |
| `src/fingerspelling/ctc_decode.py` | #15 |
| `src/fingerspelling/ctc_inference.py` | #17 |
| `src/fingerspelling/ctc_model.py` | #17 |
| `src/fingerspelling/ctc_targets.py` | #13 |
| `src/fingerspelling/ctc_training.py` | #17 |
| `src/fingerspelling/evaluation.py` | #15 |
| `src/fingerspelling/jamo.py` | #11 |
| `src/fingerspelling/sequence_features.py` | #16 |
| `src/fingerspelling/sequence_preprocessing.py` | #16 |
| `src/fingerspelling/sequence_validation.py` | #16 |
| `src/fingerspelling/tokens.py` | #13 |
| `src/label_preparation.py` | #12 |
| `tests/test_audit_crowd_keypoints.py` | #18 |
| `tests/test_cer_metrics.py` | #15 |
| `tests/test_crowd_dataset_validation.py` | #18 |
| `tests/test_crowd_keypoint_adapter.py` | #18 |
| `tests/test_crowd_keypoints.py` | #18 |
| `tests/test_crowd_labels.py` | #12 |
| `tests/test_crowd_source_validation.py` | #19 |
| `tests/test_ctc_batching.py` | #16 |
| `tests/test_ctc_checkpoint.py` | #17 |
| `tests/test_ctc_decode.py` | #15 |
| `tests/test_ctc_inference.py` | #17 |
| `tests/test_ctc_model.py` | #17 |
| `tests/test_ctc_targets.py` | #13 |
| `tests/test_ctc_training.py` | #17 |
| `tests/test_fetch_crowd_docs.py` | #19 |
| `tests/test_fingerspelling_evaluation.py` | #15 |
| `tests/test_fingerspelling_tokens.py` | #13 |
| `tests/test_jamo.py` | #11 |
| `tests/test_label_preparation.py` | #12 |
| `tests/test_sequence_features.py` | #16 |
| `tests/test_sequence_preprocessing.py` | #16 |
| `tests/test_sequence_validation.py` | #16 |
| `tools/audit_crowd_keypoints.py` | #18 |
| `tools/audit_ctc_targets.py` | #13 |
| `tools/fetch_crowd_docs.py` | #19 |
| `tools/validate_crowd_source.py` | #19 |

58개 파일 모두 기존 PR blob과 일치했다. 비교 기준에서만 생긴 실행 코드는 0개다.
중복 경로는 #11/#12 초기화 설명문과 #18/#19의 어댑터·키포인트·설명 문서다. 후행 #19 변경은 실제 ancestry를 따라 적용됐다.

## MediaPipe 16단계 연결

기존 PR 시뮬레이션 결과 위에 이미 Push된 16개 Head를 순서대로 3-way 적용했다.
첫 runtime 적용에서 패키지 초기화 설명문 add/add가 추가로 발생했다. 실행 AST가 없는 것을 확인하고 원래 MediaPipe 설명문을 선택했다.
나머지 단계에서 실행 코드 충돌은 발생하지 않았다. 각 단계 관련 회귀를 실제 실행했다.
16개 Base/Head의 merge-base도 별도로 계산했다. contracts의 Base runtime 팁 f45fb7d는 Head의 조상이 아니고 공통 조상은 8a0f704다.
이 차이는 이전 작업의 runtime 후속 문서 Commit에서 생겼으며 Base 쪽 변경은 docs/pr_split/01-2-mp-runtime.md 한 파일이다. 3-way 적용에서 해당 문서가 보존되고 실행 코드 충돌은 없었다.
나머지 15개는 현재 Base 팁이 Head의 조상이다. 문서 분기만을 이유로 16개 코드 브랜치를 다시 만들지 않았다.
최종 전체 회귀는 `513 passed, 2 warnings, 254 subtests passed in 114.71s (0:01:54)`다.
기존 확장 checkpoint와 실제 69프레임 캐시 1개를 읽어 원본·시뮬레이션 코드의 CPU/GPU/공개 API 출력 일치도 확인했다.
최종 시뮬레이션 tree는 기존 docs/2-mediapipe-pr-stack 최종 tree와 완전히 일치했다.
보호 Commit 95d53b3의 main 대비 40개 경로도 blob 바이트가 일치했다.
기존 미추적 35개와 특징·checkpoint 149개 파일의 SHA가 동일했다. 데이터 학습·영상 전처리 CLI는 실행하지 않았다.
상세 원본 API 응답·58개 blob SHA·모든 merge-base·충돌 양쪽 내용·테스트 명령과 로그는 D 저장소 .tmp-issue2/dependency-final에 보존했다.
