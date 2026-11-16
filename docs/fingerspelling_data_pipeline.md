# 지문자 CTC 시계열 데이터 파이프라인

작업 브랜치: `feat/2-ctc-data-pipeline`. 확인일: 2026-11-12 (Asia/Seoul).
저장소 루트: `D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition`.
변경은 신규 파일뿐이며 커밋, push, PR 생성, merge는 수행하지 않았다.

## 사전 조사

- 시작 시 작업 트리는 깨끗했다. 로컬 HEAD 및 origin/main은 `c627cf7`이었다.
- GitHub의 최신 main 루트, src, tests 목록과 landmarks.py를 읽어 로컬 구조와 계약을 대조했다. fetch/merge로 작업 트리를 변경하지 않았다.
- [Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2): 53점 xy/mask, 15fps 추출, 외부 자모 타깃에 대한 CTC 학습이 접점이다. 본 변경은 학습 데이터 준비만 담당하며 이슈 전체 완료를 주장하지 않는다.
- landmarks.py: 상체 `[0,2,5,7,8,11,12,13,14,15,16]`의 11점 다음 왼손 21점, 오른손 21점. 어깨는 배열 인덱스 5/6, 손목은 11/32. 검출 실패는 mask=False이며 pose 좌표는 visibility가 낮아도 저장될 수 있다. 좌표 0 자체는 결측을 의미하지 않는다.
- build_dataset.py: clip NPZ는 xy,mask,fps,gloss,word_id,signer,complexity,z,wh; full NPZ는 face,face_mask,t,src_fps,phase,full,seg_start,seg_end 등을 추가한다. full의 t는 실제 원본 프레임 시각(초)이다. nominal fps와 실제 간격은 다를 수 있다.
- 로컬 youtube_test NPZ 하나의 키/shape/dtype만 조사했다. xy=(40,53,2) float32, mask=(40,53) bool, z,wh,fps,word_id,signer,gloss 구조였다. 내용이나 라벨은 문서에 공개하지 않았다. 학습 검증에는 사용하지 않았다.
- preprocess.py에는 어깨 정규화, 결측 보간 및 값 유지, 고정 길이 resample이 있다. dataset.to_arrays는 고정 길이 단어 분류용이다. 새 파이프라인은 실제 길이, 결측 provenance와 불규칙 시간 간격이 필요하여 이 경로를 호출하지 않는다. 기존 경로는 수정하지 않는다.
- 기존 테스트는 합성 데이터와 직접 실행하는 assert 함수 방식이다. 신규 테스트는 추가 의존성 없이 unittest로 실행하며 pytest에서도 수집 가능하다. 저장소 main에 .github 및 PR 템플릿은 없었다.

GitHub API로 확인한 변경 경로:

| PR | 파일 |
| --- | --- |
| [#11](https://github.com/yj2trigger/sign-language-recognition/pull/11) | src/fingerspelling/{__init__,jamo}.py, tests/test_jamo.py |
| [#12](https://github.com/yj2trigger/sign-language-recognition/pull/12) | .gitignore, README.md, docs/label_preparation{,_validation}.md, src/fingerspelling/{__init__,crowd_labels}.py, src/label_preparation.py, tests/test_{crowd_labels,label_preparation}.py |
| [#13](https://github.com/yj2trigger/sign-language-recognition/pull/13) | docs/ctc_targets.md, src/fingerspelling/{ctc_targets,tokens}.py, tests/test_{ctc_targets,fingerspelling_tokens}.py, tools/audit_ctc_targets.py |
| [#14](https://github.com/yj2trigger/sign-language-recognition/pull/14) | src/models.py, src/train.py, tests/test_number_aux.py, tools/number_labels.py, runs/e15-numaux-l0.3/tcn-f{0,1,2}-s0.json, runs/e15-numaux-l1.0/tcn-f{0,1,2,3,4,5}-s0.json, runs/e15-numaux-l2.0/tcn-f{0,1,2}-s0.json |
| [#15](https://github.com/yj2trigger/sign-language-recognition/pull/15) | docs/fingerspelling_inference_evaluation.md, src/fingerspelling/{cer_metrics,ctc_decode,evaluation}.py, tests/test_{cer_metrics,ctc_decode,fingerspelling_evaluation}.py |

이 변경과 정확히 같은 경로는 없다. `__init__.py`는 추가하지 않고 namespace package로 사용한다. 나중에 #11/#12의 일반 패키지와도 함께 사용할 수 있다. 제출 직전 PR 변경 목록을 다시 확인해야 한다.

## API 및 계약

| 신규 모듈 | 공개 함수 / 클래스 |
| --- | --- |
| sequence_validation.py | SequenceError, ValidationReport, validate_sequence, sequence_times |
| sequence_preprocessing.py | PreparedSequence, missing_runs, prepare_sequence |
| sequence_features.py | FeatureSequence, build_sequence_features |
| ctc_batching.py | CTCSample, CTCBatch, validate_targets, ctc_minimum_length, transform_input_lengths, prepare_ctc_sample, iter_ctc_samples, collate_ctc_batch, iter_ctc_batches |

입력은 비어 있지 않은 `(T,53,2)` 실수/정수 배열과 엄격한 boolean `(T,53)` mask다. 문자열·복소수·숫자 mask는 거부한다. NaN, ±Inf, float32로 표현 불가능한 값은 mask=False 슬롯을 포함해 기본 거부한다. `invalid_policy="mask"`는 해당 슬롯을 관측에서 제외하고 출력 좌표를 0으로 만든다. 화면 밖 좌표는 MediaPipe에서 가능하므로 기본 clip하지 않는다. 필요할 때 `coordinate_bounds=(0,1)` 같은 명시적 품질 정책으로 거부하거나 마스킹한다.

원본 배열을 변경하지 않는다. observed는 원래 mask와 좌표 유효성의 교집합이다. missing_rate는 보간 전 결측 슬롯 수/(T*53)다. 전체 결측은 기본 거부한다. `allow_all_missing=True`로 전처리 진단 결과를 받을 수 있지만 학습 sample은 사용 가능한 좌표가 하나도 없으면 거부한다.

`max_gap=0`이 기본이다. 보간을 켜면 양쪽 실제 관측값이 존재하고 길이가 max_gap 이하인 구간만 실제 시각으로 선형 보간한다. 가장자리, 긴 구간, 한 번도 검출되지 않은 관절은 0 및 available=False다. `missing_runs(~가 아닌 valid mask)`는 반열린 [start,stop) 구간을 반환한다. 미해결 구간은 `missing_runs(prepared.available[:, joint])`로 확인한다. 보간은 미래 관측을 쓰므로 오프라인 학습용이다. 실시간 causal 전처리와의 일치는 후속 과제다. 시간축 전체 지속 시간이 float64에서 표현되지 않는 극단 입력은 잘못된 가중치로 보간하는 대신 명시적으로 거부한다.

정규화:

- `none`: 이미지 좌표 보존(유효하지 않은 슬롯만 0).
- `center`: 각 프레임의 실제 관측 관절 centroid를 원점으로 이동.
- `shoulder`: 실제 관측 어깨 5/6의 중점으로 이동하고 어깨 거리로 나눔.
- center/shoulder는 먼저 y를 aspect로 나눠 화면 너비 단위로 변환한다. 기본 aspect=16/9는 landmarks.fit_16x9 계약이다. 다른 추출기의 좌표는 실제 aspect를 전달한다.
- 어깨 미검출·너비 epsilon 이하 프레임은 normalization_valid=False이고 해당 프레임의 모든 available=False다. 이전/이후 기준값을 빌려 쓰지 않으며 observed 자체는 보존한다. 프레임을 삭제하지 않으므로 T는 그대로다.
- aspect/epsilon은 실제 유한 양수 scalar만 허용한다. 정규화 중간 계산과 최종 출력의 NaN/Inf/표현 범위도 검사하여 극단 입력이 available=True인 NaN 특징이 되는 것을 막는다.

특징은 float32 `(T,F)`이며 관절 0부터 52까지 joint-major로 평탄화한다:

| motion | 관절별 채널 순서 | F |
| --- | --- | --- |
| none | x,y,observed,interpolated,available | 265 |
| delta | x,y,observed,interpolated,available,dx,dy,motion_valid | 424 |
| velocity (기본) | x,y,observed,interpolated,available,vx,vy,motion_valid | 424 |

속도는 `(현재 좌표-직전 좌표)/실제 dt(초)`다. fps 또는 timestamps를 반드시 지정한다. 둘 다 주면 timestamps가 계산에 우선하며 fps 자체도 유효해야 한다. timestamp는 T개 실제 numeric 유한값이며 엄격히 증가해야 한다. 생성된 시간축, 인접 dt 및 전체 지속 시간까지 유한성을 검사한다. 문자열·복소수·boolean timestamp/fps는 거부한다. delta는 프레임당 차분이며 시간 없이도 쓸 수 있다. 제공된 시간 메타데이터는 max_gap=0 전처리에서도 검증한다. 첫 프레임 또는 두 끝점 중 하나라도 available=False면 움직임과 motion_valid는 0이다. 원본 좌표를 유지하는 none 모드의 속도 단위는 이미지 정규화 단위/초, shoulder 모드는 어깨너비/초다. 프레임별 정규화 기준 이동 역시 차분에 반영된다.

## 연결 예시

프로젝트 루트에서 `src`를 PYTHONPATH에 넣거나 기존 테스트와 동일한 sys.path 설정을 사용한다.

```python
import numpy as np
from fingerspelling.ctc_batching import prepare_ctc_sample, collate_ctc_batch

# target_ids와 blank_id는 외부 타깃 생성기에서 받아야 한다.
with np.load(npz_path, allow_pickle=False) as data:
    timing = ({"timestamps": data["t"]} if "t" in data
              else {"fps": float(data["fps"])})
    sample = prepare_ctc_sample(
        video_id, data["xy"], data["mask"], target_ids,
        blank_id=blank_id, num_classes=num_classes,
        signer_id=confirmed_signer_id,  # 미확인이라면 None
        normalization="shoulder", max_gap=2, **timing)

batch = collate_ctc_batch(
    [sample], blank_id=blank_id, num_classes=num_classes,
    length_transform=model_output_length,  # 모델이 정의한 callable
    max_padded_bytes=256 * 1024 * 1024)
```

word_id/gloss는 CTC 타깃이 아니다. NPZ signer나 파일명도 자동으로 수행자 ID로 간주하지 않는다. caller가 검증된 ID만 전달하며 없으면 signer_verified=False다. PR #12/#13을 import하지 않는다. 공식 토큰표는 만들지 않는다. 외부 타깃은 1차원 비음수 int64 범위 정수이고 blank는 금지된다. boolean과 정수를 혼합한 Python list/tuple도 거부한다(NumPy 변환 시 boolean이 정수로 바뀌는 문제 방지). num_classes가 주어지면 ID 상한 및 blank 범위까지 확인한다. 빈 타깃은 CTC 음성 예제로 허용하며 타깃에 blank를 삽입하지 않는다.

CTC 최소 시간 길이는 `타깃 개수 + 인접 동일 라벨 쌍 수`다. 원본 T에서 먼저 검사하고, 모델 길이 변환을 제공하면 변환된 길이에서도 검사한다. callable에는 독립 복사한 int64 길이를 전달한다. 반환값은 동일 shape의 양의 정수이고 입력 길이 이하여야 한다. 모델 규칙을 모르면 output_lengths=None이며 모델 출력 길이 검증이 완료된 상태로 취급하면 안 된다. downsampling 없는 모델도 명시적 identity callable을 전달한다.

batch.features는 `(B,max(T),F)`이며 padding은 0이다. **padding_mask=True는 실제 프레임**을 뜻한다(일부 attention API는 반대이므로 변환 필요). 실제 프레임에 관절이 없어도 frame mask는 True이고 특징의 available 채널로 구분한다. input_lengths와 packed targets/target_lengths는 int64다. output_lengths를 알 때만 CTC loss 길이로 사용한다.

`iter_ctc_samples(records, blank_id=..., on_error=...)`는 video_id,xy,mask,target_ids와 선택적인 signer_id,fps,timestamps를 가진 mapping을 한 건씩 처리한다. 기본 오류 정책은 즉시 예외다. callback을 제공하면 (video_id,SequenceError)를 전달하고 해당 기록만 건너뛴다. `iter_ctc_batches(samples,batch_size,...)`는 한 배치만 모으고 마지막 작은 배치도 반환한다. 향후 PyTorch Dataset.__getitem__에서 prepare_ctc_sample, DataLoader.collate_fn wrapper에서 collate_ctc_batch를 사용할 수 있다. PyTorch 설치나 모델 코드는 포함하지 않는다.

## 메모리, 검증 및 별도 검토

특징/전처리는 O(T*53), 패딩은 O(B*배치 최대 T*F)다. 원본 영상 디코딩·전체 데이터셋 적재·전체 길이 고정은 하지 않는다. max_batch_size 및 max_padded_bytes로 출력 특징+frame mask 할당을 사전 제한한다. 이 제한은 전처리 중간 배열, 타깃, caller 보유 배열과 전체 프로세스 RSS까지 제한하지 않는다. 길이가 다른 표본의 패딩 낭비는 caller의 길이 버킷 구성으로 줄일 수 있다. 압축 NPZ는 배열 단위 해제가 필요하므로 진정한 memory map은 별도 비압축 저장 형식이 필요하다.

초기 구현 검증은 standalone Python 3.14 / NumPy 2.4.5를 사용했다. 최종 독립 리뷰에서는 `.conda/environments.txt`를 읽어 기존 Conda sign-language 환경을 발견했고 실제 import로 다음 환경을 확인했다. conda가 PATH에 없다는 사실과 환경 부재는 다르다. 새 패키지는 설치하지 않았다.

- 인터프리터: `D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe`
- Python 3.12.16, NumPy 1.26.4, pytest 9.2.0, MediaPipe 0.10.21, opencv-python-headless 4.11.0.86 (cv2.__version__=4.11.0), scikit-learn 1.9.1.
- torch는 패키지 메타데이터 및 실제 import 모두에서 부재다. 이 환경에서는 OpenCV/MediaPipe를 원인으로 미실행이라고 보고하지 않는다.

```powershell
$pipelinePython = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $pipelinePython -m pytest tests/test_sequence_validation.py tests/test_sequence_preprocessing.py tests/test_sequence_features.py tests/test_ctc_batching.py -q
& $pipelinePython -m pytest tests --ignore=tests/test_train_pipeline.py --ignore=tests/test_realtime.py -q
& $pipelinePython tests/test_select_classes.py
& $pipelinePython -m pytest tests/test_train_pipeline.py tests/test_realtime.py -q
```

- 최종 신규: **38개 통과, 59 subtests passed**. 초기 29개에 독립 리뷰 기반 테스트 9개를 추가했다. 원시 좌표/mask/결측/정규화/시간/특징/패딩/CTC/NPZ/재현성을 포함한다.
- 실행 가능한 전체 pytest 출력: **56 passed, 59 subtests passed, 2 warnings**. 이 중 실제 라벨 테스트는 내부 return으로 제외되어 pytest가 pass로 잘못 보일 수 있다. 유효한 검증 집계는 신규 38 + 기존 합성 17 = **55개 통과**, 실제 라벨 1개 내부 제외다.
- 기존 합성 17개: test_preprocess.py 10개, test_select_classes.py 3개, test_extract_full.py 4개. 클래스 선택 스크립트를 따로 실행해 실제 라벨 부재의 내부 skip 출력을 확인했다.
- test_train_pipeline.py와 test_realtime.py는 별도 실행 시 torch import에서 collection error 2개다. 설치하지 않았으므로 전체 회귀 실행은 불가능하다. OpenCV/MediaPipe 기반 추출 회귀는 실행·통과했다. 2 warnings는 설치된 protobuf 타입 관련 DeprecationWarning이다.
- 최종 Conda 측정: 합성 **30,000프레임** + 단일 배치의 특징 50,880,000 bytes, 배치 특징+mask 50,910,000 bytes; tracemalloc peak **118.54 MiB**, 약 0.230초. 입력 생성 후 추적 시작.
- 길이 64~127, 배치 8 lazy 처리: 512개 peak **5.42 MiB** (0.977초), 4,096개 peak **5.51 MiB** (7.720초). 데이터셋 수를 8배로 늘려도 전체 보관에 비례한 메모리 증가는 보이지 않았다. 수치는 단회 측정으로 RSS나 실제 학습 성능 보장이 아니다.
- 30,000/1프레임 두 표본의 패딩 할당은 101,820,000 bytes다. 제한이 이 값이면 허용, 1 byte 작으면 거부되는 경계를 확인했다.
- 학습/검증 전체에서 계산한 통계나 signer 추론은 사용하지 않는다. 실제 split의 signer 독립성 검증은 별도 필요하다. 보간은 각 영상 내부에서만 수행한다.

## 최종 독립 리뷰 및 수정

별도 리뷰 에이전트 `independent_review`가 코드와 테스트를 읽고 결함을 직접 재현했다. 구현자는 수정과 회귀 테스트를 담당했고, 같은 독립 에이전트가 수정 후 Conda 환경에서 재실행해 38개 통과 및 기존 재현 차단을 확인했다. 작성자 자체 재검토만으로 독립 리뷰를 대체하지 않았다. 검토 범위에서 잔여 차단 결함은 발견되지 않았다.

| 발견한 문제 | 수정 | 회귀 검증 |
| --- | --- | --- |
| tiny fps로 만든 시간축이 [0,inf,inf]여도 반환 | 실제 생성 times, dt, 전체 duration 유한성 검사 | fps=1e-320 다중 프레임 거부; 단일 프레임 t=0 허용 |
| [-1e308,0,1e308] 보간에서 전체 span=inf가 되어 midpoint를 0으로 계산 | float64에서 전체 지속 시간 표현 불가 시 명시적 오류 | 인접 dt가 유한해도 전체 span overflow 거부 |
| tiny aspect 정규화가 NaN을 available=True로 반환 | aspect 변환·기준값·최종 좌표 유한성 및 범위 검사 | center/shoulder 및 none/delta/velocity feature 경로에서 오류 |
| max_gap=0 전처리 직접 호출 시 잘못된 시간 메타데이터 무시 | 전달된 메타데이터는 항상 검증 | 직접 prepare_sequence(fps=0) 거부 |
| list [True,2]가 integer 타깃으로 coercion | 변환 전 boolean 거부 | 혼합 bool 타깃 거부, uint64/int64 상한 검증 |

추가 경계 검증: 20개 동일 타깃의 최소 길이를 독립 계산한 39(20 emission+19 separator)로 T=39 허용/T=38 거부; 29,998프레임 결측 구간 미보간; 수만 프레임; read-only strided 공유 입력; 정수/NaN/object mask 거부; 극소/극대 dt 및 0/음수 dt; 일정 좌표의 속도 0; 원본↔sample↔batch 특징/타깃 비공유 및 오류 callback video_id 보존. 극단 수치 재현은 warnings-as-errors에서도 의도한 ValueError/SequenceError로 종료됨을 독립 에이전트가 확인했다.

## 한계 및 후속 작업

AI-Hub 지문자 실제 데이터와 실제 CTC 모델 학습/CER는 검증하지 않았다. 실데이터의 검출 실패율·화면 밖 좌표 분포·어깨 미검출률을 측정해 bounds/max_gap/정규화 정책을 결정해야 한다. 본 변경은 정책을 임의 확정하거나 미검출 영상을 자동으로 정상화하지 않는다. 유효 슬롯이 극소수인 표본을 제외할 품질 임계값은 caller 정책으로 남겨 둔다.

PR #12/#13의 검증된 라벨/타깃, 공식 blank/vocabulary, 수행자 메타데이터와 연결하고 signer 분리와 실제 모델 downsampling 길이 계산 및 학습을 확인해야 한다. PyTorch가 있는 환경에서 남은 전체 회귀와 DataLoader/CTC loss 통합 검증을 수행한다. 독립 리뷰와 권장 Python/NumPy 환경의 실행 가능한 회귀는 완료했으므로 NumPy 파이프라인 범위의 커밋·리뷰용 PR 제출은 가능하다. 실제 데이터·실제 학습 또는 이슈 전체 완료를 주장하는 제출은 아직 불가능하다. 최신 PR #11~#15 경로는 최종 리뷰에서 다시 조회했으며 여전히 중복이 없다. 커밋/push/PR 생성/merge는 실행하지 않았다.

전체 신규 파일 경로 및 PR용 줄 범위는 [PR 초안](fingerspelling_data_pipeline_pr.md)에 기록한다.
