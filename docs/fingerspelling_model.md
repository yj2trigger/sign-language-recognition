# 지문자 CTC 모델 및 학습 시스템

## 사전 조사

2026-11-12 기준 작업 브랜치는 `feat/2-fingerspelling-model`이고 시작 작업 트리는 깨끗했다. 원격 `git ls-remote origin refs/heads/main`으로 확인한 최신 main은 `c627cf77eb55564fb1a85ece63628d5bba67bb9a`이며 로컬 HEAD와 일치했다. Issue #2와 PR #11–#16의 GitHub 변경 파일 목록을 확인하고 로컬 Git 객체의 관련 구현을 읽었다. 저장소에 적용되는 AGENTS.md는 발견되지 않았다.

Issue #2는 15fps landmark 입력, 미학습 수행자 CER 및 `read_fingerspelling(xy, mask)` 연결을 요구한다. 이번 작업은 요청된 모델·학습 하위 범위를 구현한다. 실제 CER과 실시간 통합은 후속 통합 단계에서 검증해야 한다. [Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2)

기존 `src/models.py`의 GRUNet은 2층 BiGRU와 시간 평균·최댓값 pooling을 사용하며 TCNNet도 pooling 후 영상 단위 클래스를 반환한다. `src/train.py`는 AdamW, cosine scheduler, label-smoothed CE와 수행자 분리 macro-F1을 사용한다. 기존 `.pt`는 state/model/n_feat/classes와 전처리 설정을 저장하며 지문자 체크포인트와 별개의 계약이다.

기존 landmark는 상체 11점과 양손 42점 순서의 `xy[T,53,2]`, 검출 `mask[T,53]`를 반환한다. PR #16의 전처리는 관절별 5채널 또는 8채널을 joint-major로 펼친 `[T,265]` 또는 `[T,424]` 특징을 제공한다. 모델의 input_size를 해당 특징 폭으로 설정한다.

## 구조 선택

| 고려 항목 | TCN | 선택한 BiGRU |
|---|---|---|
| 시간 계산 | convolution 병렬화에 유리 | recurrent 계산으로 긴 영상에 느릴 수 있음 |
| 문맥 | kernel/dilation에 따라 수용 영역 결정 | 발화 전체 양방향 문맥 |
| 가변 길이 | 블록별 마스킹과 normalization 검토 필요 | PackedSequence로 padding을 recurrent 계산에서 제외 |
| 출력 길이 | stride 도입 시 길이 변환 필요 | 시간 축 보존 |

`F → Linear(H) → LayerNorm → GELU → Dropout → packed 2-layer BiGRU(H) → Dropout → Linear(C)`를 새로 구현했다. 기본 H는 96이며 외부 설정으로 변경할 수 있다. blank를 포함한 C와 blank ID는 필수 인자다. 공식 37종 및 토큰 ID 순서는 정의하지 않는다. GRU 내부 dropout은 0이며 명시적 Torch dropout을 사용한다. cuDNN 내부 dropout RNG는 state_dict로 복원되지 않아 CUDA 학습 재개가 달라지는 결함을 실제 실행에서 발견하고 수정했다.

양방향 모델은 완성된 발화/구간을 사용한다. 실시간 streaming state 전달은 후속 설계가 필요하다. downsampling을 사용하지 않으므로 `output_lengths == input_lengths`다. CTC 반복 토큰의 정렬 가능성을 보존하는 선택이며 성능 우위는 실제 데이터 실험이 필요하다.

## 공개 API

| 모듈 | 클래스·함수 |
|---|---|
| ctc_model.py | CTCModelConfig, FingerspellingCTC, integer, lengths_tensor |
| ctc_training.py | CTCTrainingConfig, CTCTrainer, seed_everything, ctc_loss, batch_tensors |
| ctc_checkpoint.py | capture_rng, restore_rng, checkpoint_payload, write_checkpoint, save_checkpoint, load_checkpoint, restore_trainer, validate_optimizer_state |
| ctc_inference.py | CTCRecognizer |

`FingerspellingCTC.forward(features, input_lengths)`는 float32 `[B,T,F]`와 양의 정수 `[B]`를 받아 logits `[B,T,C]`, CPU int64 `[B]`를 반환한다. B=1을 지원한다. 시간 padding은 오른쪽의 연속 구간이다. padding을 포함한 모든 값은 유한해야 한다. padding logits는 0이며 손실과 디코더에 유효 길이를 반드시 전달한다. 원본 입력은 변경하지 않는다.

`ctc_loss(logits, output_lengths, targets, target_lengths, blank_id=...)`는 packed int64 `[sum(U)]` 정답을 받는다. 인접 반복 수 R에 대해 `T_out >= U + R`를 검사한다. blank/범위 밖 정답, 길이 불일치, NaN/무한대, 불가능한 정렬을 거부한다. `CTCLoss(reduction="none", zero_infinity=False)` 뒤 표본별 `max(U,1)`로 정규화하고 배치 평균을 계산한다. 빈 정답은 모든 프레임이 blank인 경로로 학습한다. 빈 영상은 거부한다. [PyTorch CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

`CTCTrainer.run_epoch(batches, training=...)`는 sample-weighted epoch loss를 반환한다. 학습은 AdamW, 유한 gradient 검사 및 norm clipping을 사용한다. 검증은 gradient를 비활성화한다. `fit(train_factory, validation_factory, epochs=N, checkpoint_dir=...)`의 N은 추가 epoch 수다. factory는 매 epoch 새로운 iterable을 반환해야 한다. history에 train_loss와 validation_loss를 기록한다. 재사용 시 소진되는 generator를 factory 바깥에 만들지 않는다.

`seed_everything(seed)`를 모델 생성 전에 호출해야 초기 가중치까지 재현된다. trainer도 seed를 설정한다. 전역 Python/NumPy/Torch RNG를 저장한다. 외부 DataLoader generator, worker RNG, sampler cursor는 호출자가 관리해야 한다. 재개는 완료된 epoch 경계에서 지원한다. 중간 epoch에서 중단되면 마지막 완료 epoch 체크포인트로 되돌아간다. 동일 장치 종류와 CUDA device topology를 재개 조건으로 검증한다.

CUDA 모델도 CTCLoss는 CPU에서 계산하고 autograd device copy를 통해 모델에 gradient를 전달한다. CUDA CTC backward의 결정성 제약을 피하기 위한 선택이며 전송 비용이 있다. `CUBLAS_WORKSPACE_CONFIG` 기본값은 `:4096:8`이며 CUDA 초기화 전에 seed 함수를 호출해야 한다. 결정적 알고리즘과 TF32 비활성 설정은 프로세스 전역에 적용되며 checkpoint에 보존된다. RTX 5060 Laptop에서 CUDA 학습과 재개를 검증했다. 실제 데이터 처리량은 후속 측정이 필요하다. [PyTorch 결정성 설정](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)

## 체크포인트와 추론

`.pt`에는 format/version, model_config/state, training_config, optimizer_state, epoch/step, best_loss/history, vocabulary, feature_schema, RNG, device_type 및 best checkpoint snapshot이 들어간다. NumPy RNG는 primitive 목록으로 저장한다. 로드는 `weights_only=True` 및 CPU map_location을 사용한다.

각 파일은 같은 디렉터리 임시 파일에 저장하고 fsync 후 os.replace한다. `last.pt`가 완료 epoch의 권위 있는 상태이며 내부에 이전 best의 완전한 재개 상태도 보존한다. 이후 `best.pt`를 작성한다. 두 번째 쓰기가 실패해도 last에서 다음 fit을 수행하면 best를 복구한다. 다른 폴더로 재개해도 이전 best를 보존한다. 동시 writer는 지원하지 않으므로 학습 실행마다 전용 checkpoint_dir을 사용한다.

중단 시 마지막 성공한 `last.pt` 경로로 새 trainer의 `resume(path)`를 호출한다. 모델 설정, 학습 설정, vocabulary 및 feature_schema가 일치해야 한다. 로딩은 임시 모델/optimizer에서 검증하고 실패 시 기존 trainer와 RNG를 유지한다. optimizer moment의 형상·유한성·분산 부호와 step도 검증한다. 추가 snapshot은 모델/optimizer 크기에 비례한 메모리와 저장 공간을 사용한다.

`CTCRecognizer(path, device="cpu", expected_vocabulary=..., expected_feature_schema=...)`는 가중치를 읽고 eval 모드로 설정한다. `predict(features, input_lengths)`는 inference_mode에서 CPU logits와 CPU 길이를 반환한다. 저장된 TF32 설정을 호출 동안 적용하고 성공·실패 후 호출자의 설정을 복구한다. 해당 backend 설정은 프로세스 전역이므로 다른 정밀도를 요구하는 스레드의 동시 호출은 지원하지 않는다. vocabulary/schema는 선택적 metadata이며 실제 통합에서는 외부에서 확정한 mapping과 전처리 계약을 함께 저장하고 expected 인자로 검증해야 한다. raw xy/mask의 전처리는 PR #16을 통해 전달한다.

## 기존 PR 연결

| PR | 제공 계약 | 이번 모듈과의 연결 |
|---|---|---|
| #12 | 영상명과 정답 문자열 매핑 | 영상 ID로 정확히 조인한 뒤 #13 encoder에 text 전달 |
| #13 | vocabulary.token_ids/blank_id/num_classes, CTCTarget.ids | 외부 ID 그대로 model config 및 packed targets에 전달 |
| #15 | greedy_decode_batch의 batch-major scores와 output lengths | `scores.tolist()`, `lengths.tolist()`를 외부에서 전달 |
| #16 | CTCBatch.features/input_lengths/targets/target_lengths | batch_tensors가 동일 필드 mapping 또는 객체를 받음 |

PR #16 `padding_mask=True`는 실제 프레임을 의미한다. adapter는 길이로 만든 연속 prefix와 mask를 비교한다. 선택적 output_lengths도 모델의 identity transform과 일치해야 한다. collator에는 `length_transform=lambda values: model.output_lengths(values).numpy()`를 전달할 수 있다. 특징은 np.float32, targets와 lengths는 np.int64다.

PR #11–#16의 변경 파일과 신규 경로 중 겹치는 파일은 없다. #14의 models.py/train.py 및 숫자 보조 학습도 조사했다. 어느 PR의 모듈도 직접 import하거나 복사하지 않았다. namespace package를 사용한다. 선행 PR 병합 후 실제 객체 연결 회귀가 필요하다.

## 합성 데이터 실행 예제

아래 ID는 테스트용 외부 inventory다. 실제 지문자 토큰표는 별도로 공급해야 한다.

```python
import torch
from fingerspelling.ctc_model import CTCModelConfig, FingerspellingCTC
from fingerspelling.ctc_training import CTCTrainer, CTCTrainingConfig, seed_everything
from fingerspelling.ctc_inference import CTCRecognizer

seed_everything(7)
model = FingerspellingCTC(CTCModelConfig(2, 3, blank_id=0, hidden_size=8))
trainer = CTCTrainer(model, CTCTrainingConfig(seed=7),
                     vocabulary={"synthetic_a": 1, "synthetic_b": 2},
                     feature_schema={"kind": "synthetic", "width": 2})
batch = {"features": torch.ones(2, 6, 2), "input_lengths": torch.tensor([6, 4]),
         "targets": torch.tensor([1, 1]), "target_lengths": torch.tensor([1, 1])}
trainer.fit(lambda: [batch], lambda: [batch], epochs=3, checkpoint_dir="runs/ctc-synthetic")
trainer.resume("runs/ctc-synthetic/last.pt")
trainer.fit(lambda: [batch], lambda: [batch], epochs=2, checkpoint_dir="runs/ctc-synthetic")
scores, lengths = CTCRecognizer("runs/ctc-synthetic/best.pt").predict(
    batch["features"], batch["input_lengths"])
```

`PYTHONPATH=src`를 설정한 Python 프로세스에서 실행한다. 실험 체크포인트와 실제 영상·라벨은 저장소에 추가하지 않는다.

## 검증 기록과 한계

2026-11-13 사용자 승인에 따라 기존 Conda `sign-language`에 공식 `torch==2.10.0+cu128`을 설치했다. 새 환경을 생성하지 않았다. Python 실행 경로는 `D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe`, 버전은 3.12.16이다. Conda 목록에 sign-language-recognition이라는 이름은 없으며 기존 프로젝트 검증에 사용한 환경을 선택했다. 셸에는 활성 Conda 환경이 없었고 PATH의 python은 WindowsApps 별칭이어서 절대 경로를 사용했다.

- 공식 인덱스 `https://download.pytorch.org/whl/cu128`에서 설치했다. 2.11 후보는 setuptools 하향을 요구하여 기존 setuptools 84.2.1을 유지하는 2.10을 선택했다. 설치 전후 기존 패키지 버전 변경은 0개이며 torch와 필수 의존성 8개만 추가했다. NumPy 1.26.4, MediaPipe 0.10.21, OpenCV 4.11.0.86을 유지했다. pip check는 통과했다.
- GPU는 NVIDIA GeForce RTX 5060 Laptop GPU, VRAM 8151MiB, 드라이버 595.71, compute capability 12.0이다. torch CUDA runtime은 12.8이며 sm_120을 포함한다. CPU/CUDA tensor 연산과 자동 미분을 확인했다.
- 설치 전 디스크 여유는 C 32,234,983,424bytes, D 235,431,854,080bytes였다.
- 최초 torch import와 최초 scikit-learn import에서 Windows 응용프로그램 제어 DLL 정책 오류가 각각 한 차례 발생했다. 정책 변경 없이 재실행에서 import와 테스트가 정상 완료됐다.
- 신규 테스트는 독립 CTC 경로 열거 oracle, padding 및 입력 gradient 불변성, 10,000프레임, 반복/빈 정답, 손실 역전파와 clipping, optimizer step, 원자 저장 실패, best 보존, CPU/CUDA RNG 및 optimizer까지 동일한 재개, 장치 간 추론을 실제 실행한다. 최종 개수와 명령은 PR 문서에 기록한다.
- 30step 합성 학습에서 CPU loss는 2.3109567165 → 0.6406862736, CUDA loss는 2.3109569550 → 0.6406863928로 감소했다.
- 최종 전체 회귀는 89 passed, 0 failed, 0 skipped, 기존 protobuf 경고 2개이며 96.35초가 걸렸다. 신규 CTC 62개와 기존 27개가 포함된다. 독립 최종 재리뷰도 신규 62 passed다.

별도 리뷰 에이전트가 실제 PyTorch로 신규 테스트 및 결함 재현을 독립 실행했다. CUDA의 cuDNN 내부 dropout으로 인한 재개 불일치, 손상 optimizer moment를 허용하는 문제, 새 추론 프로세스의 TF32 정밀도 불일치를 발견했다. 명시적 projection dropout, optimizer 상태 검증, 저장 정밀도의 scoped 추론 적용으로 수정하고 회귀 테스트를 추가했다. 최초 정적 리뷰에서 발견한 CTC 결정성, mixed boolean 및 best 보존 수정도 유지했다.

실제 영상·라벨, 확정된 inventory와 전처리 정책, 검증된 수행자 ID 및 signer-disjoint 분할을 확보한 뒤 학습하고 #15의 CER을 기록해야 한다. 데이터가 확보되면 클래스 불균형과 길이별 메모리/지연도 측정한다. 기존 test_v1_unchanged_on_real_labels는 지정된 ZIP이 없으면 return하여 pytest에서는 pass로 집계된다. 해당 테스트의 통과가 실제 라벨 검증을 의미하지 않는다.

모델·학습 시스템의 리뷰용 PR 제출이 가능하다. 실제 데이터 통합과 Issue #2의 수행자 독립 CER 및 실시간 호출 완료 기준은 후속 작업이다.
