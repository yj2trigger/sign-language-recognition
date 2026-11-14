# 지문자 CTC 추론·CER 평가

## 범위와 사전 조사 (2026-11-12 KST)

- 저장소: `yj2trigger/sign-language-recognition`, Issue [#2](https://github.com/yj2trigger/sign-language-recognition/issues/2).
- 브랜치: `feat/2-ctc-inference-evaluation`; base `main` 및 원격 main: `c627cf77eb55564fb1a85ece63628d5bba67bb9a`.
- main에는 단어 분류용 `dataset.py`, `models.py`, `train.py`, `eval_saved.py`, `realtime.py`가 있다. 기존 평가는 분류 정확도·F1 중심이며 CTC 디코더와 CER 구현은 없다. 기존 수행자 fold 코드는 단어 데이터용이다.
- [PR #11](https://github.com/yj2trigger/sign-language-recognition/pull/11): `8f23314`, 자모 분리·조합. 일반 모드는 음절 경계 모호성이 있으며 별도 경계 보존 옵션이 있다.
- [PR #12](https://github.com/yj2trigger/sign-language-recognition/pull/12): `2f10cda`, 형태소 파서·CROWD 영상/정답 매핑·분할 검증. 기존 파서와 CROWD 로직은 재구현하지 않는다. 해당 PR의 `CROWDxx` signer 값은 공식 수행자 신원을 보장하지 않는다.
- [PR #13](https://github.com/yj2trigger/sign-language-recognition/pull/13): `b37954a`, 외부 TokenVocabulary·학습 정답 생성·길이 검사. base는 `feat/2-label-preparation`. 토큰 정의·인코더·진단 도구는 재구현하지 않는다.
- 다른 열린 [PR #14](https://github.com/yj2trigger/sign-language-recognition/pull/14)는 숫자 보조 학습 헤드이며 `models.py`, `train.py`, 숫자 도구·테스트·실험 결과를 변경한다. 이번 파일과 겹치지 않는다.
- #11·#12·#13은 조사 시점 모두 open/unmerged. GitHub 파일 목록과 실제 로컬 Git 객체를 확인했다.
- 공식 37종 목록·고정 ID·복합 자모와 실제 손동작 대응·숫자/영문/공백의 학습 표현은 미확정이다. 관측된 37종이나 Unicode 자모를 공식 목록으로 채택하지 않는다.

## 파일과 공개 인터페이스

프로젝트 루트: `D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition`.

| 상대 경로 | 공개 함수·클래스 |
|---|---|
| `src/fingerspelling/ctc_decode.py` | `DecodeError`, `greedy_decode`, `greedy_decode_batch`, `TokenRenderer` (`from_vocabulary`, `tokens`, `render`, `blank_id`, `num_classes`) |
| `src/fingerspelling/cer_metrics.py` | `MetricError`, `TextPolicy` (`apply`), `ErrorCounts` (`errors`, `cer`, 덧셈), `edit_counts`, `character_error_rate` |
| `src/fingerspelling/evaluation.py` | `EvaluationError`, `Reference`, `VideoResult`, `EvaluationReport` (`mean_video_cer`), `validate_evaluation_data`, `evaluate_texts`, `evaluate_scores` |
| `tests/test_ctc_decode.py` | CTC·토큰 렌더링 검증 |
| `tests/test_cer_metrics.py` | Unicode·편집거리·오류별 집계 검증 |
| `tests/test_fingerspelling_evaluation.py` | 누수·평가 통합·실제 PR 계약 검증 |
| `docs/fingerspelling_inference_evaluation.md` | 설계·실행·검증·한계 |
| `docs/fingerspelling_inference_evaluation_pr.md` | GitHub에 붙여넣을 PR 제목과 본문 |

모든 파일은 신규 추가다. 기존 파일·금지 파일은 변경하지 않는다. `fingerspelling/__init__.py`도 만들지 않고 Python namespace package로 사용한다. 선행 PR의 정규 패키지가 합쳐져도 같은 디렉터리 모듈로 동작한다. 기존 미추적 `.vscode/`는 사용자 파일로 유지했다.

## 점수·토큰 계약

`greedy_decode(scores, blank_id=..., num_classes=..., length=None)` 입력은 `[T,C]` 실수 점수다. logits/probability/log-probability 모두 argmax만 사용한다. 배치 함수는 명시적 `[B,T,C]`이며 ragged T도 허용한다. `[T,B,C]` 자동 추정은 하지 않는다.

1. 프레임별 최댓값 ID를 선택한다. 완전 동률이면 가장 작은 ID다.
2. 연속한 같은 ID를 한 번만 출력한다.
3. blank를 제거한다. `a,a,blank,a`는 `a,a`로 남는다.

반환은 불변 tuple ID 시퀀스이며 배치는 tuple의 tuple이다. `length`는 모델 출력 프레임 수다. 양의 클래스 수·blank 범위·매 프레임의 정확한 C·정수 유효 길이를 검증한다. 음수·bool 길이·초과 길이는 거부한다. 유효 길이 뒤 padding도 폭과 수치 유효성을 검사한다. NaN과 양/음 무한대는 모두 거부하므로 `-inf` 마스킹 모델은 호출 전 유한 점수 정책을 정해야 한다. 빈 T와 빈 배치도 계약 검증 후 허용한다.

표준 라이브러리만 필요하다. NumPy 배열은 선택적으로 지원하나 import하지 않는다. PyTorch tensor는 직접 받는 계약이 아니다. 추후 adapter가 `detach().cpu().tolist()` 등으로 변환해야 한다. 입력은 수정하지 않는다.

`TokenRenderer`는 외부 `token -> ID` mapping을 복사해 보존한다. token 순서나 공식 ID를 생성하지 않는다. PR #13처럼 sparse ID를 허용하며 `num_classes=max(blank, IDs)+1`이다. sparse 구멍 ID가 예측되면 `unregistered_token_id`로 실패한다. blank는 문자로 내보내지 않으며 렌더러는 반복 토큰을 다시 축약하지 않는다.

composer 미지정 시 문자 토큰을 그대로 join한다. 자모 모델은 composer를 반드시 명시하여 완성형 문자 CER을 계산해야 한다. composer에는 새로운 `list[str]`를 전달한다. 실패를 임의의 문자열로 바꾸거나 삭제하지 않는다. ValueError/TypeError/KeyError는 원문 없는 `composition_rejected`로 전환한다. 그 밖의 callback 실행 오류는 숨기지 않는다.

## CER 정책

CER은 Unicode code point 기준 `(S+D+I)/N`이다. 한글은 기본 NFC 후 완성형 음절이 문자 단위가 된다. grapheme cluster 단위 지표가 아니다.

- 기본 `TextPolicy()`는 정답·예측을 모두 NFC 정규화한다. 원본 입력은 유지한다.
- `TextPolicy(normalize_nfc=False)`는 정규화 없이 비교한다.
- 숫자·공백·탭·구두점·영문 대소문자를 그대로 센다. strip, 공백 삭제, 숫자 읽기 변환, NFKC 전각 변환은 없다.
- 호환 자모 `ㄱㅏ`는 NFC만으로 `가`가 되지 않는다. 자모 조합은 #11 callback 책임이다.
- 정답/예측 모두 빈 문자열이면 CER 0. 정답만 비고 예측이 있으면 CER `+inf`. 예측만 비면 CER 1. CER은 1보다 클 수 있다.
- corpus CER은 모든 오류 수와 N을 합산한 뒤 나눈다. 영상별 CER의 단순 평균은 `mean_video_cer`로 구분한다. 빈 정답 영상의 삽입도 corpus 분자에 포함한다. 전체 N=0일 때 같은 0/+inf 규칙을 적용한다.
- 비어 있는 평가 집합은 corpus counts=0, CER=0, 평균=None이며 신원 검증 상태는 unverified다. 성공적인 실제 평가의 증거가 아니다.
- 최소 편집거리 정렬이 여러 개면 diagonal(일치/치환), 삭제, 삽입 순으로 동률을 처리한다. S/D/I는 이 정책에 따른 재현 가능한 한 최적 정렬의 집계다.
- `ErrorCounts`는 음수·bool·불가능한 S+D>N을 거부한다. 잘못된 문자열 타입 및 surrogate도 거부한다.

CER DP는 O(N*M) 시간, 예측 길이 M에 대해 O(M) 메모리다. 실제 DP 표 전체나 편집 경로는 저장하지 않는다. 디코더는 점수를 프레임별 순회한다. 평가기는 영상별 결과·ID 매핑을 보존하므로 항목 수 및 예측 출력 길이에 비례하는 메모리가 필요하다. 매우 긴 문장에는 O(N*M) 시간이 한계이며, 이 구현에 실시간 성능을 주장하지 않는다.

## 데이터 검증과 신원

`Reference(video_id, text, performer_id=None)`를 사용한다. 빈 정답/예측 문자열은 허용하되 None은 누락이다. 영상 ID 중복, 예측 ID 중복, 정답과 예측 ID 집합 불일치, 잘못된 record/type을 거부한다. 중복을 탐지해야 하면 예측을 `(video_id, value)` pair iterable로 전달한다. 이미 dict로 덮어쓴 중복은 복원할 수 없다.

`training_performer_ids`와 평가 수행자가 겹치거나 `training_video_ids`와 영상이 겹치면 항상 오류다. 수행자 정보 미제공은 None이며 임의 추정하지 않는다. `None` 수행자는 결과에서 별도 그룹으로 집계한다.

`identities_verified=True`는 호출자가 외부 근거로 학습·평가 ID의 동일한 실제 수행자 namespace를 검증했다는 선언이다. 이 코드가 신원을 인증한 결과가 아니다. CROWDxx를 자동 추출하지 않는다. 검증 선언·모든 평가 수행자 ID·학습 수행자 inventory가 있어야 verified다. 학습 수행자 None은 알 수 없음, `[]`는 학습 수행자가 없음을 명시한다. 빈 평가 집합은 verified가 될 수 없다. 학습 영상 목록은 선택적 추가 검증이며 미제공이면 영상 중복 여부는 확인할 수 없다.

`require_verified=True`이면 미검증 평가를 거부한다. 기본은 CER 계산을 허용하되 `identity_status='unverified'`와 원인을 보고한다. 수행자 누수·검증 정책은 점수 소비와 composer 호출 전에 검사한다. 실제 모델 평가에서는 검증된 inventory와 `require_verified=True`를 사용해야 한다.

보고서에는 원본 정답/예측 문자열을 넣지 않으며 repr에서도 영상/수행자 ID를 숨긴다. 그러나 결과 속성에는 집계에 필요한 ID가 있으므로 `by_performer` 등을 외부에 저장할 때는 호출자가 비식별화해야 한다. JSON 직렬화 도구는 제공하지 않는다. `+inf`와 MappingProxyType은 저장 형식에 맞게 명시적으로 변환해야 한다.

## 독립 실행 예제

프로젝트 루트에서 `src`를 PYTHONPATH에 추가한다.

```python
from fingerspelling.ctc_decode import TokenRenderer
from fingerspelling.evaluation import Reference, evaluate_scores

# 합성 문자 목록이며 공식 지문자 목록이 아니다.
renderer = TokenRenderer({'가': 1}, blank_id=0)
report = evaluate_scores(
    [Reference('synthetic-video', '가')],
    {'synthetic-video': [[0, 3], [0, 2], [3, 0]]},
    renderer=renderer,
)
assert report.corpus.cer == 0
assert report.identity_status == 'unverified'
```

## PR #11·#13 및 실시간 시스템 연결

선행 PR 반영 후 다음 계약으로 연결한다. 미병합 코드를 이 브랜치에 복사하지 않는다.

```python
from fingerspelling.tokens import TokenVocabulary       # PR #13
from fingerspelling.jamo import compose_hangul            # PR #11
from fingerspelling.ctc_decode import TokenRenderer

# approved_token_ids와 approved_blank_id는 확정된 외부 설정이다.
vocabulary = TokenVocabulary(approved_token_ids, blank_id=approved_blank_id)
renderer = TokenRenderer.from_vocabulary(vocabulary, compose=compose_hangul)
```

경계 보존 토큰을 학습한 경우에만 동일한 boundary 정책의 composer를 전달한다. 일반 조합 모드의 음절 경계 모호성은 해결되지 않는다. 예측 자모가 불완전하면 오류로 나타나므로 실제 모델 통합 시 부분 발화의 조합 정책을 별도 결정해야 한다.

Issue #2의 향후 `read_fingerspelling(xy, mask) -> str` adapter는 다음을 담당한다.

1. 15fps landmark `xy(T,53,2)`와 `mask(T,53)`를 학습 때와 같은 전처리로 변환.
2. 모델 checkpoint·토큰 ID·blank·출력 C의 동일성 검증.
3. 모델 출력 축을 `[B,T,C]`로 변환하고 subsampling 후 유효 길이를 전달.
4. `greedy_decode_batch` → `renderer.render` → 최종 문자열 반환.
5. streaming chunk 경계의 반복/blank 상태·발화 종료·부분 문자열/조합 실패 정책을 관리.

현재 디코더는 완성된 발화 기준 stateless다. chunk별 결과를 단순 연결하면 경계 반복이 틀릴 수 있다. 실시간 adapter·모델·전처리·chunk 상태는 이번 구현 범위가 아니며 `realtime.py`를 수정하지 않았다.

## 검증과 독립 리뷰

기존 Conda 환경을 그대로 사용했다. 새 환경 생성/패키지 설치/PyTorch 설치 없음.

```powershell
$python = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $python -m pytest -q tests/test_ctc_decode.py tests/test_cer_metrics.py tests/test_fingerspelling_evaluation.py
& $python -m pytest -q tests/test_preprocess.py tests/test_select_classes.py tests/test_extract_full.py
& $python -m pytest -q --ignore=tests/test_realtime.py --ignore=tests/test_train_pipeline.py
```

환경 Python 3.12.16. 최종 실행 가능 전체: **109 passed**, 2개 기존 protobuf DeprecationWarning. 신규 **91 passed**, 기존 선택 **18 passed**. 기존 실제 라벨 테스트는 파일 부재 시 자체 return하므로 pytest pass 개수가 실제 AI-Hub 검증을 뜻하지 않는다.

전체 `pytest -q`도 시도했으나 `test_realtime.py`, `test_train_pipeline.py` 두 모듈이 `ModuleNotFoundError: torch`로 수집 실패했다. 해당 기존 기능의 전체 회귀는 미완료다.

검증 범위: blank/반복/blank로 분리된 동일 ID, 동률, sparse unknown ID, 잘못된 수치/shape, ragged 배치·길이, 입력 불변성, 20,000 프레임, 빈 정답/예측, CER>1, NFC/NFD, 숫자/공백, corpus와 평균 차이, 신원 미검증, 수행자·영상 누수, 5,000개 평가 항목 및 전체 파이프라인.

CTC는 길이 0~6, 클래스 3의 1,093개 경로를 itertools.groupby oracle과 비교했다. 편집거리는 길이 0~4 이진 문자열 31종의 961개 쌍을 독립 full-matrix 거리와 비교했다. 상수/기대값은 구현 출력에서 만들지 않았다.

PR 계약 테스트는 검토한 고정 Git SHA의 jamo.py/tokens.py를 메모리에서만 실행하여 11,172 현대 음절과 합성 파이프라인을 검증했다. 해당 객체가 없는 clean clone에서는 이 선택적 테스트가 skip된다. 이는 merge 완료나 공식 37종 검증을 의미하지 않는다.

별도 리뷰 에이전트가 정확성·중복/충돌·예외·누수·성능/메모리·확장성·테스트를 검토했다. 초기 85 tests를 독립 실행했으며 누수 검증이 디코딩 뒤에 이루어지는 문제를 재현했다. `_prepare`를 추론 전에 실행하도록 수정하고 사전 거부 4건 및 inventory generator 1회 소비 회귀 1건을 추가했다. 실제 PR 메모리 연결 테스트도 추가했다. 사람의 독립 리뷰는 아직 없다.

## 제출 상태와 남은 일

이 범위는 리뷰용 PR 제출 가능하다. Issue #2 전체 완료나 merge 준비 완료를 뜻하지 않는다. `Refs #2`만 사용한다.

- 공식 37종/ID·숫자·영문·공백·경계 정책 확정.
- 실제 모델 학습, 출력 길이/축/checkpoint 계약 확인.
- 공식 수행자 신원·학습/평가 inventory 검증, 미학습 수행자 실제 영상 CER 측정.
- 실제 영상/라벨 정합 검증 및 15fps 모델 전처리 연결.
- 실시간 adapter·streaming·부분 발화 조합 정책 구현.
- 선행 PR merge 후 통합 회귀, PyTorch 환경에서 기존 전체 테스트, GitHub CI 및 사람 리뷰.

실제 AI-Hub 영상·개인 이름·라벨 원문은 추가하지 않았다. 실제 모델 CER·실시간 지연·정확도를 측정했다고 주장하지 않는다. 커밋·push·PR 생성·merge는 수행하지 않았다.
