# Issue #2 PR 분리 및 Git 보존 결과

2026-11-14에 GitHub API와 실제 Git ref를 확인했다. Refs #2.
원래 브랜치 `feat/2-mediapipe-fingerspelling`와 `95d53b362c4ce117ce6bdb15ae012d710aec0edb`를 보존했다.
모든 작업·Worktree·로그는 D드라이브에 있으며 지정 Conda Python만 사용했다.

## 기존 PR 현황

| PR | 상태 | Base → Head | 파일 | 추가 | 삭제 | 선행 관계 |
|---|---|---|---:|---:|---:|---|
| [#11](https://github.com/yj2trigger/sign-language-recognition/pull/11) | open, merged=False | `main` → `feat/2-jamo-utils` | 3 | 220 | 0 | 없음 |
| [#12](https://github.com/yj2trigger/sign-language-recognition/pull/12) | open, merged=False | `main` → `feat/2-label-preparation` | 9 | 1029 | 0 | 없음, #11 관련 |
| [#13](https://github.com/yj2trigger/sign-language-recognition/pull/13) | open, merged=False | `feat/2-label-preparation` → `feat/2-ctc-targets` | 6 | 483 | 0 | #12 Base, #11 자모 모듈 필요 |
| [#15](https://github.com/yj2trigger/sign-language-recognition/pull/15) | open, merged=False | `main` → `feat/2-ctc-inference-evaluation` | 7 | 1032 | 0 | 독립 디코딩/CER |
| [#16](https://github.com/yj2trigger/sign-language-recognition/pull/16) | open, merged=False | `main` → `feat/2-ctc-data-pipeline` | 9 | 1073 | 0 | 독립 시계열 처리 |
| [#17](https://github.com/yj2trigger/sign-language-recognition/pull/17) | open, merged=False | `main` → `feat/2-fingerspelling-model` | 9 | 1045 | 0 | #13/#15/#16 런타임 연계 |
| [#18](https://github.com/yj2trigger/sign-language-recognition/pull/18) | open, merged=False | `main` → `feat/2-crowd-keypoint-adapter` | 9 | 1809 | 0 | #12/#13/#16 연계 |
| [#19](https://github.com/yj2trigger/sign-language-recognition/pull/19) | open, merged=False | `feat/2-crowd-keypoint-adapter` → `feat/2-crowd-source-validation` | 9 | 891 | 7 | #18 Base |
| [#20](https://github.com/yj2trigger/sign-language-recognition/pull/20) | open, merged=False | `main` → `research/2-fingerspelling-training-integration` | 1 | 278 | 0 | #11~#19 분석 문서 |

기존 PR별 실제 변경 파일은 다음과 같다.
- #11 `src/fingerspelling/__init__.py`, `src/fingerspelling/jamo.py`, `tests/test_jamo.py`
- #12 `.gitignore`, `README.md`, `docs/label_preparation.md`, `docs/label_preparation_validation.md`, `src/fingerspelling/__init__.py`, `src/fingerspelling/crowd_labels.py`, `src/label_preparation.py`, `tests/test_crowd_labels.py`, `tests/test_label_preparation.py`
- #13 `docs/ctc_targets.md`, `src/fingerspelling/ctc_targets.py`, `src/fingerspelling/tokens.py`, `tests/test_ctc_targets.py`, `tests/test_fingerspelling_tokens.py`, `tools/audit_ctc_targets.py`
- #15 `docs/fingerspelling_inference_evaluation.md`, `src/fingerspelling/cer_metrics.py`, `src/fingerspelling/ctc_decode.py`, `src/fingerspelling/evaluation.py`, `tests/test_cer_metrics.py`, `tests/test_ctc_decode.py`, `tests/test_fingerspelling_evaluation.py`
- #16 `docs/fingerspelling_data_pipeline.md`, `src/fingerspelling/ctc_batching.py`, `src/fingerspelling/sequence_features.py`, `src/fingerspelling/sequence_preprocessing.py`, `src/fingerspelling/sequence_validation.py`, `tests/test_ctc_batching.py`, `tests/test_sequence_features.py`, `tests/test_sequence_preprocessing.py`, `tests/test_sequence_validation.py`
- #17 `docs/fingerspelling_model.md`, `src/fingerspelling/ctc_checkpoint.py`, `src/fingerspelling/ctc_inference.py`, `src/fingerspelling/ctc_model.py`, `src/fingerspelling/ctc_training.py`, `tests/test_ctc_checkpoint.py`, `tests/test_ctc_inference.py`, `tests/test_ctc_model.py`, `tests/test_ctc_training.py`
- #18 `docs/fingerspelling_crowd_keypoints.md`, `src/fingerspelling/crowd_dataset_validation.py`, `src/fingerspelling/crowd_keypoint_adapter.py`, `src/fingerspelling/crowd_keypoints.py`, `tests/test_audit_crowd_keypoints.py`, `tests/test_crowd_dataset_validation.py`, `tests/test_crowd_keypoint_adapter.py`, `tests/test_crowd_keypoints.py`, `tools/audit_crowd_keypoints.py`
- #19 `docs/crowd_source_validation.md`, `docs/fingerspelling_crowd_keypoints.md`, `src/fingerspelling/crowd_keypoint_adapter.py`, `src/fingerspelling/crowd_keypoints.py`, `src/fingerspelling/crowd_source_validation.py`, `tests/test_crowd_source_validation.py`, `tests/test_fetch_crowd_docs.py`, `tools/fetch_crowd_docs.py`, `tools/validate_crowd_source.py`
- #20 `docs/fingerspelling_training_integration_plan.md`

이 PR들은 조회 시 모두 open이었다. 각 PR의 Base/Head SHA와 파일 목록은 작업 로그의 `github-pr-audit.json`에 있다.
기존 PR끼리 `src/fingerspelling/__init__.py`가 겹치며 #19는 #18의 두 어댑터 파일과 설명 문서를 수정한다.
초기화 파일의 서로 다른 설명문은 통합 기준에서 #12 문구를 사용하고 첫 새 PR에서 원래 MediaPipe 설명문을 복원했다.
이 변경은 설명문 한 줄이며 기존 자모·CTC 구현을 새 기능으로 제출하지 않는다.

원래 MediaPipe 브랜치에 복사돼 있던 자모·타깃·디코딩·CER·시계열·모델의 12개 모듈은 기존 PR blob과 동일하다.
이 구현들은 비교 기준에서 상속하므로 새 PR의 변경 파일에 중복 추가되지 않는다.

## 기준 브랜치와 의존성

`integration/2-existing-pr-baseline`는 9개 기존 PR의 파일을 모은 비교·테스트 기준 전용 브랜치다.
독립된 기존 PR들을 함께 실행하기 위해 선택한 구조이며 이 기준 브랜치 자체에는 새 PR을 제안하지 않는다.
Git merge/rebase/reset으로 기존 Commit을 변경하지 않고 새 선형 Commit의 트리에 기존 PR blob을 복사했다.
58개 파일/+7852줄은 기존 PR의 재사용 기준 코드이며 아래 새 PR의 실제 Base diff에서 제외된다.
각 새 PR은 바로 이전 새 브랜치를 Base로 사용한다. 기준이 없는 main과 비교하면 기존 코드가 다시 포함된다.
**선행 PR의 main 병합 후 Base 이름만 main으로 바꿔도 3-dot merge-base가 자동 정리되지는 않는다.**
실제 main 제출 시점에는 병합된 main에서 새 분리 브랜치를 다시 만들거나 승인된 이력 정리 절차가 필요하다.
현재 산출물은 기준 브랜치와 비교해 크기를 검증한 검토용 스택이다. 기존 PR Merge와 Base 재조정은 후속 단계다.

## 브랜치별 결과

| PR 예정 번호 | Base | Head | 변경 파일 | 추가 | 삭제 | 전체 테스트 | Push |
|---|---|---|---:|---:|---:|---|---|
| 기준 전용 | `main` | `integration/2-existing-pr-baseline` | 58 | 7852 | 0 | 390 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `integration/2-existing-pr-baseline` | `feat/2-mp-runtime` | 6 | 352 | 1 | 420 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-runtime` | `feat/2-mp-dataset-contracts` | 3 | 283 | 0 | 437 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-dataset-contracts` | `feat/2-mp-dataset-preparation` | 3 | 228 | 26 | 441 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-dataset-preparation` | `feat/2-mp-ctc-training` | 3 | 258 | 8 | 441 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-ctc-training` | `feat/2-mp-pipeline-guide` | 3 | 379 | 0 | 441 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-pipeline-guide` | `feat/2-mp-ctc-conditioning` | 3 | 299 | 0 | 460 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-ctc-conditioning` | `feat/2-mp-model-evaluation` | 3 | 327 | 5 | 460 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-model-evaluation` | `feat/2-mp-partial-rendering` | 6 | 294 | 18 | 468 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-partial-rendering` | `feat/2-mp-data-expansion` | 4 | 391 | 5 | 482 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-data-expansion` | `feat/2-mp-hand-features` | 6 | 218 | 3 | 494 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-hand-features` | `feat/2-mp-video-label-review` | 3 | 221 | 0 | 499 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-video-label-review` | `feat/2-mp-expansion-report` | 4 | 212 | 0 | 499 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-expansion-report` | `feat/2-mp-review-drafts` | 3 | 303 | 0 | 499 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-review-drafts` | `feat/2-mp-keypoint-experiment` | 3 | 346 | 0 | 513 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-keypoint-experiment` | `feat/2-mp-keypoint-report` | 2 | 335 | 0 | 513 passed | 원격 SHA 확인 기록 참조 |
| 미생성 | `feat/2-mp-keypoint-report` | `docs/2-mediapipe-pr-stack` | 3 | 147 | 5 | 513 passed | 원격 SHA 확인 기록 참조 |

새 PR마다 실제 Base와의 `git diff Base...Head --numstat`를 집계했다. 제안 브랜치는 모두 10파일/추가400줄 이하다.
소스 함수 경계로 데이터 계약 API, 전처리 API, CLI 학습 실행을 단계화했고 각 단계 전체 테스트를 실행했다.
중간 단계는 아직 추가하지 않은 CLI 대신 해당 API와 테스트를 검토 진입점으로 제공한다.
기존 OpenPose 검증 도구·회귀와 보고서는 MediaPipe 스택 뒤의 두 별도 검토 단위로 보존했다.

## Commit과 PR 본문 경로

| 순서 | Head | Commit SHA | PR Markdown |
|---|---|---|---|
| 0 | `integration/2-existing-pr-baseline` | `1a0fa9cb90effe095109c8a466cb2bd00162cd31` | 기준 전용, PR 없음 |
| 1 | `feat/2-mp-runtime` | `f45fb7d0ee9bf4dd6caa4aa1752b20d4c17af8de` | docs/pr_split/01-2-mp-runtime.md |
| 2 | `feat/2-mp-dataset-contracts` | `ee22ef44c5efa3090376713fc079729ba202b0a6` | docs/pr_split/02-2-mp-dataset-contracts.md |
| 3 | `feat/2-mp-dataset-preparation` | `444b293c5284e4cca0621e2d4ebdfe8375b9602a` | docs/pr_split/03-2-mp-dataset-preparation.md |
| 4 | `feat/2-mp-ctc-training` | `4211eae14afd020fc5975e77c4691fe4b16d1b86` | docs/pr_split/04-2-mp-ctc-training.md |
| 5 | `feat/2-mp-pipeline-guide` | `9a466a6e827d75030e6917c761ee73a409e1fc47` | docs/pr_split/05-2-mp-pipeline-guide.md |
| 6 | `feat/2-mp-ctc-conditioning` | `d09b466591e2b9a81ba6b5518a2fa13e8c745419` | docs/pr_split/06-2-mp-ctc-conditioning.md |
| 7 | `feat/2-mp-model-evaluation` | `df0cd99594dac5f6af46d7c5811e9901c82b6e17` | docs/pr_split/07-2-mp-model-evaluation.md |
| 8 | `feat/2-mp-partial-rendering` | `7c31f23cd23e765a846276dcda9b0c6cb6caa748` | docs/pr_split/08-2-mp-partial-rendering.md |
| 9 | `feat/2-mp-data-expansion` | `6b66f29f9b2f2ce7c36af926ee636e8d75e77693` | docs/pr_split/09-2-mp-data-expansion.md |
| 10 | `feat/2-mp-hand-features` | `cc17262ed1583cb0aa1daffef2279a4f0bb5672a` | docs/pr_split/10-2-mp-hand-features.md |
| 11 | `feat/2-mp-video-label-review` | `d68231cda0ef274ae76e135ca4bf903a7079fe66` | docs/pr_split/11-2-mp-video-label-review.md |
| 12 | `feat/2-mp-expansion-report` | `1bd151f89844ddd0a872f71a1dc5026cd48239ad` | docs/pr_split/12-2-mp-expansion-report.md |
| 13 | `feat/2-mp-review-drafts` | `4308ab32ef67414601809fcbf98f6dc47ce09c98` | docs/pr_split/13-2-mp-review-drafts.md |
| 14 | `feat/2-mp-keypoint-experiment` | `07c556084015a22284eced315046d78e0d617640` | docs/pr_split/14-2-mp-keypoint-experiment.md |
| 15 | `feat/2-mp-keypoint-report` | `a4e07e038c7ed1bb9a50f9d2be0441650e638e9d` | docs/pr_split/15-2-mp-keypoint-report.md |
| 16 | `docs/2-mediapipe-pr-stack` | 이 보고서 브랜치의 HEAD | `docs/pr_split/16-2-mediapipe-pr-stack.md` |

PR 생성 및 권장 검토·Merge 순서는 기존 PR의 의존성을 먼저 해소한 뒤 위 스택 순서를 따른다.
#11과 #12를 먼저 검토하고 #13, #15/#16, #17, #18→#19 및 #20을 기존 소유자 절차로 처리한다.
실제 PR 번호는 생성 후 부여한다. 각 제목·본문은 해당 브랜치의 `docs/pr_split/` Markdown에 있다.

## 검증과 보존

최종 통합 상태의 전체 회귀는 `513 passed, 2 warnings, 254 subtests passed in 115.65s (0:01:55)`다.
모든 브랜치 테스트는 지정 Conda Python에 torch threads=2를 설정한 pytest 전체 실행이다.
명령은 `python.exe -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" -q -p no:cacheprovider --basetemp <D 고유 경로>`다.
각 실행 로그는 원래 D 저장소의 `.tmp-issue2/git-split/*-tests*.log`에 보존했다.
첫 기준 실행에서 내가 만든 테스트 임시 폴더의 부모 경로 누락으로 14개 fixture 오류가 났다. 부모를 생성한 재실행은 통과했다.
기존 원본 브랜치의 main 대비 40개 변경 경로가 최종 스택에서 Git blob 바이트까지 일치함을 확인했다.
추가된 기존 PR 테스트·문서와 각 PR 초안을 제외하면 원래 구현·설정의 동작 변경은 없다.
원래 미추적 파일과 기존 143개 특징·체크포인트는 전후 SHA로 검사했다. 상세 목록은 `protection-before.json`, `equivalence.json`에 있다.
원본 ZIP·학습 결과를 쓰는 전처리/학습 CLI는 실행하지 않았다. 회귀 테스트의 합성 모델 최적화만 테스트 폴더에서 실행됐다.
GPU 데이터 재학습·CER 재평가는 수행하지 않았으며 기존 수치는 실험 보고서를 유지한다.
확장 손 표현 Val 자모 CER .308333, 기존 Test 진단 .269841, resync 한글 Val .695652/Test진단 .64다.
수행자 독립 평가·라벨 전사 의미 검증은 미완료다.

## Push 확인

원격 이름을 먼저 확인하고 이번 작업이 만든 새 ref만 일반 Push한다. 결과 SHA는 `publish-result.json`에 기록한다.
원래 원격 브랜치 SHA도 다시 비교한다. 원격에 다른 SHA의 동일 이름이 있으면 해당 Push를 중단한다.
기존 PR 수정·생성·Merge와 Force Push는 이번 작업에 포함하지 않는다.
