# PR 제목

feat: MediaPipe 53점 기반 지문자 CTC 파이프라인과 추론 인터페이스 추가

# PR 본문 초안

## 변경 이유와 동작

Issue #2의 `xy (T,53,2)`, `mask (T,53)` 입력을 한국어 자모 CTC 인식기에 연결한다.
기존 단어 인식과 OpenPose 실험 코드를 보존하고 별도 MediaPipe 모델을 학습한다.
`read_fingerspelling(xy, mask) -> str`와 체크포인트를 한 번 로딩하는 호출 객체를 제공한다.

Refs #2. 수행자 독립 평가와 영상·라벨 의미 검증이 남아 있어 Issue를 닫지 않는다.

## 구현

- PR #11·#13·#15·#16·#17의 기존 모듈 중 필요한 12개를 그대로 재사용한다.
- 원본 ZIP에서 MP4를 하나씩 CRC 검사하고, 실제 디코딩 PTS로 목표 15Hz 프레임을 선택한다.
- 기존 `landmarks.process_frame()`으로 53점을 추출하고 라벨의 실제 시간 구간으로 자른다.
- 학습·추론이 동일한 PR #16의 424차원 특징 함수를 사용하며 시간축을 보존한다.
- ID 별칭·중복, CTC 길이, 결측·비유한 좌표, 사전·체크포인트 불일치를 검사한다.
- 완료 특징의 해시·배열 재검증과 설정 검증으로 안전하게 재개한다.
- 부분 MP4는 소유가 확인된 경우에도 보존한 뒤 재복사한다. 검증 완료한 작업 임시 MP4만 정리한다.

## 변경 범위

새 코드: `src/fingerspelling/`, `src/fingerspelling_runtime.py`, `src/fingerspelling_video.py`,
`tools/run_mediapipe_ctc.py`, `tools/audit_mediapipe_ctc.py`.
테스트: `tests/test_mediapipe_fingerspelling.py`.
문서: `docs/mediapipe_fingerspelling.md`, 이 초안.

기존 미추적 OpenPose 코드·문서·검토 라벨은 이 PR에 일괄 포함하지 않는다.
최초 두 Commit의 21개와 후속 추가 7개, 총 28개 파일만 변경 범위에 포함한다.
경로를 명시해 선별 Commit한다. 기존 OpenPose 파일과
현재 실행에서 참조하지 않는 `ctc_batching.py`는 미추적 상태로 보존한다.
원본 이름·영상·라벨·NPZ·체크포인트·개별 예측은 외부 D드라이브에 유지한다.
`src/realtime.py` 등 팀원 코드와 단어 인식 동작은 변경하지 않았다.

로컬 저장소에서 명시적인 PR 파일 수·줄 수 제한은 찾지 못했다.
선행 PR의 소스는 출처에 기록한 Git blob과 바이트가 같지만 현재 기준 HEAD에는 없다.
같은 기능을 다시 구현하지 않고 실행에 필요한 모듈을 통합한다.
추후 선행 PR 병합 시 경로 중복을 재검토해야 한다. 이번 Commit은 두 단위로 분리한다.

1. 기존 CTC 패키지 통합 및 출처 기록: 14파일.
2. MediaPipe 추출·학습·추론·테스트·문서: 7파일.

## 실제 검증과 결과

실행 결과는 `docs/mediapipe_fingerspelling.md`의 최종 실행 결과를 따른다.
실제 64영상 시도 중 59개 특징 생성, Train 39 / Validation 12 / Test 8로
RTX 5060에서 40epoch 학습했다. 자모 CER는 각각 0.809524 / 0.850000 / 0.873016,
완전 일치율은 모두 0%다. Validation·Test 예측이 각각 한 종류에 머무르므로
당시 체크포인트는 실사용 모델로 권장하지 않는다. 수행자 독립 결과도 아니다.
개발 단계 전체 회귀 342개, 신규 테스트 49개 통과. 독립 리뷰에서 특징·체크포인트·CER와
CPU 재로딩 예측 일치를 확인했다. 기존 파일 34개와 학습 산출물 1,632개의 해시가 유지됐다.
신규 합성 테스트에서 정상·한 손 결측·양손 미검출·NaN/Inf·극단 좌표·빈 입력,
CTC 반복·최소 길이·한글 왕복·사전 및 차원 불일치·복구·별칭·누수를 검증했다.
실제 데이터 학습·평가 수치는 합성 테스트와 구분한다.

## 남은 한계

- 최대 64영상의 탐색 실행으로 전체 2,000영상을 학습한 결과가 아니다.
- 형태소 표제어와 실제 철자 동작의 의미적 일치는 아직 검증되지 않았다.
- CROWD18/19를 수행자 ID로 확정하지 않았으며 수행자 독립 CER는 미완료다.
- VFR은 목표 15Hz로 선택하지만 실제 프레임 간격은 균일하지 않을 수 있다.
- 정확한 37종 동작 토큰과 Unicode 자모 51종의 대응 검증은 후속 과제다.
- 조합 불가능한 예측 자모는 보존한다. 실시간 웹캠 성능은 측정하지 않았다.
- 기존 OpenPose Train CER와 새 MediaPipe 분할 CER로 모델 우열을 판단하지 않는다.

## 재현

기존 Conda `sign-language` Python으로 `docs/mediapipe_fingerspelling.md`의 명령을 실행한다.
데이터와 체크포인트는 공개 저장소에 포함하지 않는다.

# Commit 전 추가 검증

공개 함수 직접 호출 테스트 2개를 추가했다. 기존 추적 코드와 선별한 21개 파일만
복사한 독립 소스 폴더에서 신규 테스트 **51 passed, 2 warnings, 15.06초**.
제외한 OpenPose 미추적 코드와 `ctc_batching.py` 없이도 통과했다.
실제 best 체크포인트로 공개 함수를 CPU에서 호출해 기존 Test 예측과 일치했고,
특징 재계산·로더 캐시·체크포인트 해시 불변성을 확인했다. GPU 재학습은 하지 않았다.

현재 파일 재조사 결과 `pts-run-01/features`의 NPZ 59개, 고유 영상 59개,
분할에 참조된 고유 영상 59개가 일치하며, 누락·추가 파일은 0개다.
실제 최적화에 사용한 Train은 **39개**이며 나머지는 Validation 12 / Test 8이다.
이전 56개 관찰은 현재 재현하지 못했고 해당 관찰의 경로·시점은 미확인이다.
전체 59개를 모두 Train에 투입했다고 표현하지 않는다.

실제 추가 테스트 명령:

```powershell
$py = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$validation = 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-v1\commit-validation-01'
# 선별 소스 폴더에서 실행한 테스트
Push-Location "$validation\tree"
& $py -m pytest tests/test_mediapipe_fingerspelling.py -q -p no:cacheprovider --basetemp "$validation\pytest-temp"
Pop-Location
# 기존 작업 폴더에서 실행한 전체 회귀 테스트
& $py -m pytest -q -p no:cacheprovider --basetemp "$validation\workspace-test-temp"
```

위 경로는 실제 실행 기록이다. 재실행 시 pytest가 이전 임시 파일을 정리하지 않도록
반드시 새로운 `--basetemp` 폴더를 사용한다. 결과 로그는 같은 validation 폴더에 있다.

이번 Commit 전 전체 회귀 결과: **344 passed, 2 warnings, 114.71초**.
두 경고는 기존 protobuf deprecation이다. 테스트 실패는 없다.

# Commit 구성

`[feat/2-mediapipe-fingerspelling] feat: integrate CTC dependencies`

첫 Commit: `d5b0035` (14파일, 1,333줄 추가). Staged 소스 12개를 기존 Git blob과
바이트 대조했고, `git diff --cached --check`를 통과했다.

`[feat/2-mediapipe-fingerspelling] feat: add MediaPipe CTC pipeline`

최초 두 Commit은 당시 허용 범위에서 로컬에 기록했다. 최신 지시에 따라 현재 브랜치에서
추가 성능 개선 Commit·Push를 수행하며, Force Push·PR 생성·Merge는 하지 않는다.

## 파일별 실제 변경과 줄 범위

### 후속 성능 개선 및 팀원 A 전달

- 기존 59개 특징을 읽기 전용으로 사용하고 동일 Train39/Val12/Test8을 유지했다.
- dropout 제거, 학습률 변경, Train 전용 특징 표준화의 3후보를 각40epoch 비교했다.
- 선택 후보의 자모 CER: Train0.583333 / Validation0.758333 / 기존 Test진단0.825397.
- 원래 strict 한글 CER는 악화했고, 자모를 보존하는 opt-in resync를 Val로 비교했다.
  resync 한글 CER: Train0.705036 / Val0.804348 / 기존 Test진단0.84, 완전 일치율0%.
  Val 차이는 기존보다 오류1개/46글자 감소이며 Test 한글 CER는 개선되지 않았다.
- 표준화를 기존 Linear에 합쳐 기존 424차원 입력과 체크포인트 로더를 유지했다.
  기본 strict 렌더러와 기존 모델 경로는 자동 변경하지 않는다.
- 체크포인트 선택 재개/메타데이터 복구와 RNG 보존을 보완하고,
  전처리 원본 SHA 계약이 Windows 줄바꿈 변환으로 바뀌지 않도록 3파일만 LF로 고정했다.
- 최종 전체 회귀371개 통과(2개 기존 경고,104.56초). 실제 CPU/GPU public 함수 예측 일치.
- Commit 대상만 있는 독립 소스 폴더에서도 두 테스트 파일78개 통과(14.80초).
- 기존 자료·별도 PR 파일35개는 보존했다. 새 브랜치·환경·패키지 설치는 없다.
- 수행자 독립 CER·라벨 의미 검증·실사용 성능은 여전히 미완료다.
  Test는 과거 개발에 관찰됐으며 새로운 독립 평가로 주장하지 않는다.

추가 실행 명령과 모든 후보 결과는 `docs/mediapipe_performance.md`,
호출 예제와 비공개 모델 확보·SHA 검증은 `docs/fingerspelling_handoff.md`에 기록했다.
후보 추론 체크포인트는 private D 폴더에서 확보해야 하며 GitHub에 넣지 않는다.
제3 Commit은 위 성능 진단·복구·옵션·문서에 필요한 파일만 포함한다.

비교 기준은 개발 전 HEAD `1aac6a14a5a26e09a94d1080cccfd6b8ac542023`이다.
아래 파일은 모두 새 파일이므로 변경 전에는 줄이 없고, 변경 후 범위는 파일 전체다.
기존 팀원의 추적 파일은 수정하지 않았다.

<!-- selected-file-ranges -->

| File | Before | After | Change |
|---|---|---|---|
| `.gitattributes` | absent | 1-4 | Preserve LF for three hashed preprocessing sources |
| `docs/fingerspelling_handoff.md` | absent | 1-76 | Team A interface and private checkpoint acquisition with SHA verification |
| `docs/issue2_mediapipe_pr_draft.md` | absent | 1-180 | PR body, file scope, tests and line ranges |
| `docs/mediapipe_fingerspelling.md` | absent | 1-344 | Environment, measured results, counts, limitations and reproduction |
| `docs/mediapipe_performance.md` | absent | 1-166 | Measured experiments, limitations, integrity and reproduction |
| `src/fingerspelling/__init__.py` | absent | 1-1 | Package initialization |
| `src/fingerspelling/cer_metrics.py` | absent | 1-101 | Existing PR #15 CER and deterministic D/I/S |
| `src/fingerspelling/ctc_checkpoint.py` | absent | 1-174 | Existing PR #17 atomic restricted-loader checkpoints |
| `src/fingerspelling/ctc_decode.py` | absent | 1-163 | Existing PR #15 greedy CTC decoding and rendering |
| `src/fingerspelling/ctc_inference.py` | absent | 1-46 | Existing PR #17 checkpoint-backed prediction |
| `src/fingerspelling/ctc_model.py` | absent | 1-110 | Existing PR #17 length-preserving BiGRU CTC |
| `src/fingerspelling/ctc_targets.py` | absent | 1-134 | Existing PR #13 target encoding and repeated-token length checks |
| `src/fingerspelling/ctc_training.py` | absent | 1-191 | Existing PR #17 CTC loss, trainer and resume |
| `src/fingerspelling/jamo.py` | absent | 1-104 | Existing PR #11 Hangul decomposition/composition |
| `src/fingerspelling/partial_compose.py` | absent | 1-87 | Reuse token-preserving grammar without dynamic Git dependency |
| `src/fingerspelling/provenance.md` | absent | 1-16 | Source revisions and exclusion of unused batching helper |
| `src/fingerspelling/sequence_features.py` | absent | 1-51 | Existing PR #16 424-channel features preserving time |
| `src/fingerspelling/sequence_preprocessing.py` | absent | 1-102 | Existing PR #16 masks, bounded gaps and shoulder normalization |
| `src/fingerspelling/sequence_validation.py` | absent | 1-99 | Existing PR #16 shape, finite coordinate and timing validation |
| `src/fingerspelling/tokens.py` | absent | 1-47 | Existing PR #13 immutable vocabulary |
| `src/fingerspelling_runtime.py` | absent | 1-100 | Shared features, cached reader and public read_fingerspelling interface |
| `src/fingerspelling_video.py` | absent | 1-58 | Decoded PTS checks and 15Hz-target frame selection |
| `tests/test_mediapipe_fingerspelling.py` | absent | 1-310 | 51 synthetic contract, error, timing and recovery tests |
| `tests/test_mediapipe_performance.py` | absent | 1-163 | 27 standardization, recovery, rendering and evaluation tests |
| `tools/audit_mediapipe_ctc.py` | absent | 1-85 | Optional local archive/artifact/hash and feature audit |
| `tools/compare_mediapipe_rendering.py` | absent | 1-57 | Validation-only rendering choice and cached prediction diagnostics |
| `tools/experiment_mediapipe_performance.py` | absent | 1-282 | Train-only conditioning, bounded tuning, affine export and selection recovery |
| `tools/run_mediapipe_ctc.py` | absent | 1-455 | Per-MP4 CRC extraction, recovery, grouping, GPU training and evaluation |
