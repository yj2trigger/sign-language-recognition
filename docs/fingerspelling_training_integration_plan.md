# Issue #2 지문자 학습·통합 설계 검증

조사일은 2026-11-13, 시간대는 Asia/Seoul이다. 이 문서는 코드와 확보 자료에 대한 분석 및 후속 설계다. 실데이터 변환·학습·CER 실험은 이번 작업에서 수행하지 않았다.

## 1. 조사 상태와 근거 수준

시작 브랜치는 `research/2-fingerspelling-training-integration`이며 작업 트리는 깨끗했다. HEAD와 로컬 main은 `c627cf77eb55564fb1a85ece63628d5bba67bb9a`다. 읽기 전용 `git ls-remote origin refs/heads/main`으로 원격 main도 같은 SHA임을 확인했다. 브랜치를 이동하지 않았다. 상위 경로를 포함한 확인 대상에서 AGENTS.md는 발견되지 않았다.

GitHub 커넥터로 Issue #2/#4/#6 본문, PR #11/#12/#13/#15/#16/#17/#18의 실제 diff와 head SHA를 조회했다. 해당 로컬 Git 객체의 파일을 `git show <ref>:<path>`로 추가 확인했다. 이들 PR은 조회 시 모두 open, 미병합이다. 현재 main에는 지문자 모듈이 없다. 아래의 구현 존재는 각 PR head 기준이며 main에서 실행 가능하다는 의미로 사용하지 않는다.

소스 검증 후속 원격 브랜치 `feat/2-crowd-source-validation`의 SHA는 `f7631c32b734abc1027329b8d29a9d520fc82987`이며 로컬 ref와 일치했다. GitHub head 검색에서 해당 PR은 발견되지 않았고 #19 조회는 404였다. 따라서 PR 번호를 부여하지 않고 후속 브랜치의 실제 코드·문서를 분석했다. 이 브랜치는 #18을 포함하며 파서·어댑터를 추가 수정한다.

근거 표시는 다음 원칙을 따른다.

- **직접 확인**은 이번에 읽은 코드, 원격 ref, 파일 목록과 기존 진단 JSON의 내용이다.
- **기존 실행 기록**은 #18 및 소스 검증 브랜치 보고서와 외부 진단 JSON이 기록한 전수 검사·테스트 결과다. 이번에 ZIP 671,745개 전체를 다시 파싱하지 않았다.
- **설계 제안**은 아직 구현·실험하지 않은 연결, 데이터 정책, 분할과 후속 작업이다. 실제 성능이나 성공률을 보증하지 않는다.

Issue #2는 미학습 수행자 CER과 `read_fingerspelling(xy, mask)` 호출 가능성을 완료 조건으로 둔다. #4는 BIO 종류·경계, #6은 실시간 버퍼와 라우팅·데모를 담당한다. 현재 main의 구간 탐지기는 `HandRaiseSegmenter`다. [Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2), [Issue #4](https://github.com/yj2trigger/sign-language-recognition/issues/4), [Issue #6](https://github.com/yj2trigger/sign-language-recognition/issues/6)

## 2. 기존 일반 수어 시스템 구조

아래 경로는 저장소 루트 기준이며 main SHA에 고정한 코드 조사다.

| 기능 | 실제 경로와 함수 | 확인된 동작·제약 |
| --- | --- | --- |
| 영상 입력 | `src/landmarks.py` `extract_clip`, `extract_video` | OpenCV VideoCapture. clip은 라벨 구간 ±0.5초의 원본 프레임을 처리. video는 전체 영상에서 선택한 프레임만 Holistic에 전달 |
| 전처리와 추출 | `src/landmarks.py` `fit_16x9`, `process_frame`, `_frame`, `_face` | 검은 padding으로 16:9 canvas를 구성하고 BGR→RGB. MediaPipe 0.10.21 legacy Holistic 사용. `_frame`은 xy/mask/z, 얼굴33은 별도 배열 |
| 좌표계 | `src/landmarks.py` `_frame`, `fit_16x9` | padding 후 canvas 너비·높이로 정규화된 xy. 원점은 이미지 좌상단, x 우측·y 아래 방향. 반전하지 않은 입력 전제. anatomical left/right 결과를 각 슬롯에 저장 |
| 53점 | `src/landmarks.py` `POSE_IDX`, `_frame` | pose 원본 `[0,2,5,7,8,11,12,13,14,15,16]`→슬롯0–10. 왼손21→11–31, 오른손21→32–52. pose 손목과 손 그룹 손목은 각각 유지 |
| 검출 마스크 | `src/landmarks.py` `_frame` | pose visibility≥0.5. 손 그룹 검출 시 21점 모두 True. 미검출 좌표는 0, mask False. 손별 confidence threshold 기능은 없음 |
| 15fps | `src/landmarks.py` `select_frames`, `extract_video` | `round(phase+k*src_fps/target_fps)` 인덱스, 선택되지 않은 프레임은 grab. `t=i/src_fps`, phase/src_fps/wh 보존. 원본 fps가 목표보다 작으면 거부. FPS 미보고 시 30 fallback 존재 |
| ZIP→NPZ | `src/build_dataset.py` `NAME`, `_init`, `_work`, `main` | WORD/REAL 정면 파일명과 morpheme 경로 전제. signer는 REAL 번호. `--full-fps 15` 지원. CROWD 파일명을 그대로 처리할 수 없음 |
| 저장·추출 재실행 | `src/build_dataset.py` `_work` | xy float32, mask bool, fps, gloss, word_id, signer, complexity, z, wh. full이면 face/face_mask/t/src_fps/phase/seg_start/seg_end/full 추가. tmp.npz→os.replace. 기존 dst는 skip하며 설정 일치 재검증은 없음 |
| 구간·수행자 로딩 | `src/dataset.py` `segment`, `load_clips` | full NPZ는 t로 라벨 ±0.5초 절단. word_id별 필터. signer 유지. 구형 NPZ의 wh fallback은 1920×1080 |
| 결측·정규화 | `src/preprocess.py` `normalize`, `_hold_nearest`, `fill_missing` | 어깨 기준 원점·폭. 어깨가 한번도 유효하지 않으면 None. 짧은 양끝 유효 결측≤5는 보간, 나머지는 앞값 유지와 시작부 뒷값 대체. 미관측 관절은 0 |
| 고정길이 특징 | `src/preprocess.py` `build_features`, `resample`, `hand_relative` | 몸 좌표·선택적 손 상대/각도/궤적/깊이/얼굴 등과 mask를 결합한 뒤 고정 length로 선형 리샘플. mask 채널도 실수 보간됨 |
| 데이터 증강 | `src/dataset.py` `augment`, `to_arrays`, `augmented_bank`, `subsample` | 고정길이 분류용 crop·노이즈·회전·프레임 결측과 사전 증강 bank. 부분 crop은 전체 문자열 CTC 정답과 어긋날 수 있음 |
| 수행자 분할 | `src/dataset.py` `make_folds` 및 `src/train.py` `main` | 수행자 shuffle 후 test=i, val=i+1, 나머지 train. main은 클래스90% 이상 보유 수행자 필터와 최소6명 조건. 기존 일반 수어의 기준이며 지문자에 그대로 강제할 근거는 없음 |
| 학습 | `src/train.py` `train_one`, `predict` | CPU 모델·from_numpy 텐서. CUDA 장치 선택·device transfer 없음. AdamW, cosine scheduler, CE label smoothing0.1, 검증 macro-F1 조기종료. gradient clipping 없음 |
| 체크포인트·재개 | `src/train.py` `main`, `train_one` | state/model/n_feat/classes/config/split/git 저장. 완료 JSON이 있으면 실행 skip. optimizer/scheduler/RNG 상태와 epoch 재개 기능 없음 |
| 성능·로그 | `src/train.py` `metrics`, `main`, `git_state` | accuracy/macro-F1/top5, 검증 곡선·학습시간·split/class hash·git SHA·test signer를 JSON과 stdout에 기록. CER 없음 |
| 실시간 | `src/realtime.py` `VideoClock.frame_at`, `CameraClock.frame_at`, `run`, `classify`, `load_model` | 목표15fps의 wall-clock tick, 지연 시 frame skip, process_frame 결과를10초 버퍼에 보존. 규칙 구간 후 고정64 기본 분류 경로. 완료된 CTC 인식 호출은 없음 |

`extract_clip`의 fps와 `extract_video(...,15)`의 fps를 구별해야 한다. `build_dataset.py` 기본 실행만으로 15fps가 되지 않는다. `extract_video`의 CFR 시각 계산과 30fps fallback도 실제 파일 FPS·PTS 검증을 대체하지 못한다. 기존 정규화는 y/aspect 보정 없이 어깨 폭을 계산하며 #16 shoulder 정규화는 이를 보정하므로 특징·가중치 계약이 다르다. [main 코드](https://github.com/yj2trigger/sign-language-recognition/tree/c627cf77eb55564fb1a85ece63628d5bba67bb9a/src)

### 눈 인덱스 정정

MediaPipe v0.10.21의 pose2/5는 `LEFT_EYE`/`RIGHT_EYE`다. 안쪽 눈은1/4다. #18 문서·docstring의 inner-eye 설명은 잘못됐고 소스 검증 후속 브랜치가 이를 정정한다. 기존 API 이름 `allow_missing_inner_eyes`는 호환용으로 유지된다. OpenPose와의 눈 대응 검증이 없어 common51에서 슬롯1/2를 제외하는 정책은 여전히 조건부다. [공식 enum](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/pose.py)

## 3. 지문자 PR 실제 코드와 재사용 분류

A는 수정 없이 재사용, B는 독립 어댑터 연결, C는 지문자 전용 구현, D는 미구현을 뜻한다. C의 기존 PR 구현은 그대로 활용한다.

| PR / head SHA | 실제 경로·주요 함수 | 구현 범위·남은 계약 | 분류 |
| --- | --- | --- | --- |
| #11 `8f23314` | `src/fingerspelling/jamo.py` `decompose_hangul`, `compose_hangul` | Unicode 현대 한글 분리·조합, 선택적 문자 경계. 복합 모음·겹받침을 단일 토큰으로 유지. 실제 수어37종 정의 없음. 경계 없는 분리의 역변환은 독립 자모 혼합에서 원문 보장 안 됨 | C 구현 있음 |
| #12 `2f10cda` | `src/label_preparation.py` `parse_morphemes`, `target_from_morphemes`, `validate_signer_splits`; `src/fingerspelling/crowd_labels.py` `expected_video_name`, `parse_crowd_label`, `read_crowd_zip`, `compare_video_names` | exact-video 라벨 조인과 오류·충돌 보존. `CROWDxx`는 그룹 키. 실제 사람 검증 없음. metaData/URL 일치는 영상 내용 대조와 별개 | C 구현 있음, B 연결 |
| #13 `b37954a` | `src/fingerspelling/tokens.py` `TokenVocabulary`; `ctc_targets.py` `TargetEncoder.encode`, `audit_crowd` | 외부 token ID·blank 정의 사용. PR11을 런타임 import 또는 decompose 주입. 숫자·공백 literal 정책 명시 필요. 인접 반복 R 포함 최소길이 U+R 검사 | C 구현 있음 |
| #15 `5a29221` | `src/fingerspelling/ctc_decode.py` `greedy_decode_batch`, `TokenRenderer`; `cer_metrics.py` `TextPolicy`, `edit_counts`, `character_error_rate`; `evaluation.py` `Reference`, `evaluate_scores`, `evaluate_texts`, `validate_evaluation_data` | 유효길이만 CTC 반복/blank 제거, 외부 composer. NFC 등 명시 정책, corpus·영상·수행자별 CER와 leakage 검사. 실제 checkpoint 연결과 CLI 없음 | C 구현 있음, B 연결 |
| #16 `9e22ab2` | `src/fingerspelling/sequence_validation.py` `validate_sequence`, `sequence_times`; `sequence_preprocessing.py` `prepare_sequence`; `sequence_features.py` `build_sequence_features`; `ctc_batching.py` `prepare_ctc_sample`, `iter_ctc_samples`, `collate_ctc_batch`, `iter_ctc_batches` | raw53 검증, 결측 provenance, 시간 기반 velocity, batch-local padding, packed targets. 5채널×53=265 또는8×53=424. temporal resampling 없음. 수행자 ID provenance는 호출자 책임 | C 구현 있음 |
| #17 `05acbd2` | `src/fingerspelling/ctc_model.py` `CTCModelConfig`, `FingerspellingCTC`; `ctc_training.py` `CTCTrainingConfig`, `seed_everything`, `ctc_loss`, `batch_tensors`, `CTCTrainer`; `ctc_checkpoint.py` `save_checkpoint`, `load_checkpoint`, `restore_trainer`; `ctc_inference.py` `CTCRecognizer.predict` | packed BiGRU, 시간축 보존, CPU/CUDA, CTC·clipping·epoch 재개. default CPU이므로 runner가 CUDA 지정. epoch history는 train/val loss. CER·분할·실데이터 runner 없음 | C 구현 있음 |
| #18 `d71a74b` | `src/fingerspelling/crowd_keypoints.py` `parse_keypoint_json`, `CrowdKeypointArchive`; `crowd_keypoint_adapter.py` `SourceSpec`, `conversion_blockers`, `adapt_crowd_sequence`, `adapt_mediapipe53`, `to_ctc53`, `adapt_labeled_sequence`; `crowd_dataset_validation.py` `inventory_dataset`; `tools/audit_crowd_keypoints.py` `audit_archive` | 실제 스키마의 엄격 파서, common51, 조건부53 변환. geometry·좌우·시간 근거 필요. 실제 그룹 변환 미실행,15fps 변환 없음 | B 구현 있음 |

표의 #18 구현 범위는 실제 스키마 검증과 조건부 변환이다.

소스 검증 후속 브랜치의 `src/fingerspelling/crowd_source_validation.py` `ConfidenceDiagnostics`, `audit_source`, `tools/validate_crowd_source.py`, `tools/fetch_crowd_docs.py`는 범위 밖 confidence 원값 진단, 공식 문서 다운로드 결과·provenance 검증을 추가한다. 공개 엄격 파서와 진단이 내부 `_parse_keypoint_json`을 공유하며 진단 경로를 학습 입력으로 반환하지 않는다. 소스 문서 확보·좌표계·시간 검증은 계속 미완료다.

| 기존 일반 수어 기능 | 분류와 적용 |
| --- | --- |
| `landmarks.process_frame`, `fit_16x9`, `_frame`, CFR `select_frames` | A. 영상 확보 후 같은 추출 함수 호출. FPS metadata fallback은 호출 전 검증 |
| `landmarks.extract_video` | A. 검증된 영상에15fps로 호출. 결과 저장·CROWD 조인은 B |
| `dataset.make_folds`, `sha` | A. 검증된 수행자 목록·설정 hash에 사용. 지문자 분포와 그룹 수에 맞춘 호출 검증은 B |
| ZIP 다중처리·atomic 저장 패턴 | B. 기존 `build_dataset._work`는 WORD/REAL 전용이므로 독립 CROWD runner에서 기존 추출을 호출 |
| `dataset.load_clips`, `to_arrays`, `augment`, `augmented_bank` | C. word_id·고정길이·부분 crop 의존. #16 및 지문자 manifest로 대체 |
| `preprocess.normalize`, `fill_missing`, `build_features` | C. #16의 missingness·시간축 계약 우선. 기존 함수의 보간/hold·고정길이 출력으로 우회하지 않음 |
| `models.GRUNet/TCNNet`, `train.train_one`, `metrics` | C. temporal pooling과 CE/F1 구조. #17/#15 사용 |
| source 확정, 실제15fps CTC manifest, 검증 수행자 inventory, 실데이터 학습 CLI, epoch CER 로그, 최종 wrapper | D. 후속 작업 필요 |

근거 PR은 실제 diff와 파일을 확인했다. [#11](https://github.com/yj2trigger/sign-language-recognition/pull/11), [#12](https://github.com/yj2trigger/sign-language-recognition/pull/12), [#13](https://github.com/yj2trigger/sign-language-recognition/pull/13), [#15](https://github.com/yj2trigger/sign-language-recognition/pull/15), [#16](https://github.com/yj2trigger/sign-language-recognition/pull/16), [#17](https://github.com/yj2trigger/sign-language-recognition/pull/17), [#18](https://github.com/yj2trigger/sign-language-recognition/pull/18)

### 의존 관계

- GitHub base 기준 #13만 `feat/2-label-preparation` 위에 쌓여 있다. #11/#12/#15/#16/#17/#18은 main base다. #13 diff의 #12 상속분을 별도 신규 구현으로 중복 집계하면 안 된다.
- 런타임 기준 #13→#11, #12→`label_preparation`이 있다. #15는 composer·vocabulary를 주입받고 #16은 외부 target을 받는다. #17은 #16과 duck typing으로 연결하며 현재 직접 import 의존은 자기 모듈에 한정된다.
- 실데이터 전체 연결은 #12→#13(+ #11), 영상 A 또는 #18 B→#16→#17→#15(+ #11 조합)다. GitHub base와 통합에 필요한 논리 의존은 구분해 검토했다.
- source-validation은 #18 파일을 수정하므로 #18 선행 반영 또는 그 head 기반 stacked PR이 적합하다. #11–#18은 서로 신규 경로 위주지만 API·token/schema 계약의 통합 검증이 필요하다.

## 4. 실제 확보 데이터와 기존 검증 결과

이번 파일 목록 확인에서 다음 자료가 존재했다. 지정 CROWD 루트 내 영상 파일은 0개다. 다른 디스크 전체의 영상 보유 여부는 조사하지 않았다.

```text
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\downloads\수어 영상\2.Validation\[라벨]01_crowd_keypoint.zip
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\수어 영상\2.Validation\[라벨]01_crowd_morpheme.zip
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\source-validation.json
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\source-metadata-validation.json
```

| 항목 | 기존 전수 검사 기록 및 이번 진단 JSON 확인 |
| --- | --- |
| ZIP 용량 | 키포인트1,411,121,080 bytes, 형태소952,783 bytes |
| 프레임 JSON | 671,745. 문법 정상671,744, 문법 오류1 |
| 엄격 파서 | 통과529,700, confidence [0,1] 위반142,044 |
| 영상 그룹·라벨 조인 | 각각2,000, exact-video 일치2,000, 양방향 누락0 |
| 전체 프레임 strict 통과 그룹 | 446. confidence 위반 보유 그룹1,554 |
| 관절 구조 | 정상 JSON의 Pose25/Face70/왼손21/오른손21, 3D 배열 비어 있음 |
| 시간 정보 | 모든 그룹 frame index0 시작·연속. FPS/timestamp/크기 필드 부재. camparam 비어 있지 않은 leaf0 |
| 수행자 근거 | CROWD 그룹 수2. person_id는 동일 음수 sentinel. 실제 사람 매핑 미검증 |
| 라벨 정책 | 동일 FS의 원문 불일치3그룹, 숫자 포함382 라벨. Issue의 숫자192개와 표본·단위가 달라 그대로 대입할 수 없음 |
| 실제 학습 입력 | 변환 시도0, 성공0, 시도 후 실패0, source/time 부족으로 차단2,000. verified CTC0 |

446개를 학습 가능한 그룹으로 간주할 수 없다. strict 통과는 구조·수치 판정이며 좌표 의미·시간·영상 내용·정답·수행자 증거가 추가로 필요하다. 진단 JSON의 `conversion_failure_reasons`는 시도 전 차단 사유를 담으므로 실제 변환 실행 실패 수와 구분해 해석한다.

confidence 위반은 face120,410 / pose534 / 왼손15,312 / 오른손11,545 프레임으로 그룹 간 중복된다. 원값 범위는 약 -0.000175198~1.20531이다. 얼굴을 특징에서 제외해도 현재 strict parser는 얼굴 confidence까지 검증한다. 목표 관절만 검사하도록 자동 우회할 수 없으며 정책 변경 시 별도 근거·버전·회귀 검증이 필요하다. 전체142,044개를 손 좌표 오류로 해석하거나 clamp해 성공률을 만들지 않는다.

문서 폴더의 다운로드 provenance와 후속 보고서에는 설명서7969/가이드18941 응답이 HTTP200, 79-byte HTML 접근 오류로 기록돼 있다. 정상 PDF/ZIP 문서 확보는0개다. 공개 AI-Hub 페이지의30fps 분할 설명과 Issue의60fps 계획을 개별 CROWD ZIP의 시간 명세로 확정할 수 없다. [AI-Hub 데이터셋103](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)

## 5. 영상 추출 A와 OpenPose 변환 B 비교 및 우선순위

| 기준 | 방법 A 원본 영상→MediaPipe→53→15fps | 방법 B OpenPose→검증 어댑터→53→15fps |
| --- | --- | --- |
| 좌표계 | process_frame의16:9 padded normalized 계약 재사용. 회전·반전·원본 metadata 검증 필요 | 원본 scale/canvas/padding/crop/방향 증거 필요. 수치 크기로 pixel 추정 불가 |
| FPS | 영상 CFR/PTS 확인 후 extract_video15 호출 가능. VFR은 별도 시각 기반 선택 필요 | frame 번호와 실제 sampling period/PTS 연결 필요. 연속 번호만으로 시간 확정 불가 |
| 인덱스 | 기존 MediaPipe53과 동일 | BODY25→공통 body9+손42=51. 53 슬롯1/2 결측화 정책 필요 |
| 좌우 | 기존 Holistic의 좌우 의미와 unmirrored 입력 사용 | hand_left/right의 anatomical 의미와 반전 검증 필요 |
| Confidence | pose visibility, 손 그룹 검출 mask라는 기존 계약 | 관절별 confidence threshold. MediaPipe visibility와 다른 척도. strict 범위 위반 다수 |
| 품질 | 손 가림·해상도·추적 실패를 직접 점검 가능. 실제 지문자 성능 미측정 | 이미 제공된 검출 품질에 의존. 실제 overlay·결측률·검출기 간 분포 검증 필요 |
| 확보 조건 | 해당 영상과 exact-video 라벨·metadata 필요. 현재 CROWD 루트에 영상0 | ZIP·라벨 확보. source 명세와 수행자 매핑 부족 |
| 디스크 | 원본 영상 ZIP/임시 영상/특징 필요. Issue는 묶음20–31GB를 계획했으나 실제 파일 크기 미확인 | ZIP 약1.41GB, 전체 JSON 해제6,461,753,903 bytes라는 기존 기록. 순차 ZIP 읽기로 전체 해제 생략 가능 |
| 처리 비용 | 영상 decoding+MediaPipe CPU 추출. 병렬수·임시공간 실측 필요 | MediaPipe 재추출 비용 절약 가능. 엄격 검사·시간선택·NPZ 생성 비용 실측 필요 |
| 코드 재사용 | extract_video/process_frame 직접 재사용, CROWD runner 추가 | #18 parser/adapter 직접 활용,15fps selector와 manifest 연결 추가 |
| 실시간 호환 | 실시간과 같은 검출기·canvas·53 인덱스 | 검출기 domain shift. 추론에서도 눈 결측 정책을 맞춰야 함. 양 검출기 동등성 미검증 |
| 현재 실행 수준 | 영상 부재로 CROWD 추출 미실행 | 파싱·진단 기록만 존재. 실제 CTC 변환0 |

**권장 우선순위는 B의 source 검증을 먼저 완료하고, 첫 학습 입력 생성 후보로 B를 조건부 채택하는 것이다.** 이미 ZIP이 있고 #18의 실제 파서·어댑터가 있으며 새로운 검출기 구현이 필요하지 않다. 단 source/time/quality 근거가 확보되지 않으면 B 학습 생성을 진행할 수 없다. 검증 자료 확보 작업과 함께 대응 원본 영상 소량을 확보해 overlay 및 A/B 비교에 사용한다.

실시간 배포용 기준 입력은 A의 MediaPipe 경로로 검증한다. B의 source 증거를 확보할 수 없거나 목표 관절 품질이 부족하면 A를 실학습 경로로 전환한다. 따라서 지금 대규모 영상 전량 다운로드·재추출부터 시작하는 근거는 부족하며, B를 검증 완료된 학습 방식으로 선언할 근거도 없다. 이 우선순위는 확보 자료와 코드 재사용성에 따른 설계 판단이다.

저장 비용 산정 예로 raw53 xy+bool mask는 프레임당477 bytes이고 t float64를 더하면485 bytes다. 원시671,745프레임을 전부 보존한다고 가정하면 약326MB이며 NPZ 압축률·헤더·정답·메타데이터와15fps 선택을 반영하지 않은 계산이다. 영상 A 용량과 처리시간은 확보 후 측정한다.

## 6. 좌표계·FPS와 실제 CTC 입력 생성 설계

### 검증 자료와 실험

1. 정상 설명서·가이드 또는 생산자 확인에서 export scale, canvas W/H, crop/padding/회전, x/y 축, BODY25/Hand21, anatomical 좌우, frame index 시간 규칙과 confidence 정의를 확보한다. 각 증거의 문서 페이지·파일 hash·대상 범위를 SourceSpec/manifest에 기록한다. 잘못 받은 HTML을 문서 근거로 채택하지 않는다.
2. 동일 video ID의 원본 영상과 JSON을 확보하고 영상 W/H·rotation·CFR/VFR·PTS를 검사한다. 원본60fps와 keypoint30fps가 함께 존재할 가능성은 개별 프레임 대응으로 검증한다. label duration 또는 파일 수로 FPS를 역산하지 않는다.
3. 시작·중간·끝, 손 교차·한손·가림·서로 다른 CROWD 그룹을 포함한 대응 프레임을 overlay해 상체9점과 양손21점, 손목·손가락 순서·좌우를 확인한다. 선택 표본 수와 오류·오차·한계를 기록하고 검증 범위를 파일/동일 export 그룹으로 제한한다.
4. B는 SourceSpec을 통과한 좌표에 #18의 정수 padding 규칙을 적용한다. body25 `[0,18,17,5,2,6,3,7,4]`를 기존53의 `[0,3,4,5,6,7,8,9,10]`에 연결하고 손0–20을11–31/32–52에 저장한다. 눈1/2는 정책 채택 시0/False. A/B 공통51 상대오차와 mask 분포를 측정한다.
5. Confidence 이상 원인을 확인하고 strict whole-sequence 제외를 baseline 후보로 평가한다. 관절별 제외 등은 별도 정책 버전과 문자·길이·수행자 분포 영향 비교 후 채택한다. 프레임을 삭제해 시간축을 압축하지 않는다. 문법 오류1개도 감사 결과에 남긴다.

### 15fps의 구현 공백

#16은 `fps`를 시간 생성·속도 계산에 사용하며 입력 프레임을15fps로 줄이지 않는다. #18도 frame_index/fps 또는 explicit timestamps만 보존한다. 새 독립 연결 계층에서15fps 선택을 수행해야 한다.

- A의 검증된 CFR 영상은 `landmarks.extract_video(target_fps=15)`와 `select_frames`를 재사용한다. 원본 `src_fps`, `phase`, `t`, wh를 저장하고 fallback 사용 여부를 확인한다.
- B의 검증된 균일 source sampling은 source keypoint FPS 기준 `select_frames` 인덱스를 재사용하고 원본 index/timestamp를 보존한다. fps 인수만15로 덮어쓰는 방식은 금지한다.
- VFR/불규칙 PTS는15Hz grid와 실제 시각 기반의 선택·결측 정책이 필요하다. 첫 구현은 검증된 CFR 범위를 명시하고 그 밖의 파일을 진단 대상으로 둔다. 무조건 중복 프레임을 생성하는 업샘플링은 도입하지 않는다.
- 원본 선택시각과 목표 grid 차이, 중복·누락·역전, 시작·끝·최대 오차, T와 실제 duration을 검사한다. CFR 최근접 선택의 이론 오차 상한은 약 source frame 간격의 절반이며 실제 오차를 함께 보고한다. 반올림 phase·endpoint 차이도 기록한다.

### 실제 sample 연결

설계 흐름은 `#12 exact-video text → #13 encoder`, `A 추출 또는 B #18 변환 → 15fps 선택 → #16 prepare_ctc_sample/collate_ctc_batch → #17`이다. label timestamp로 구간을 자를 경우 원본 전체 시작 기준과 fps를 맞추고 문자열 전체를 포함하는지 영상으로 확인한다. 기존 WORD의 첫 segment·±0.5초를 무조건 적용하지 않는다.

저장 manifest에는 private video_id, source hash, 좌표/시간 증거, 원본 frame_indices/timestamps, fps15 선택 정책, xy/mask shape·dtype, 품질 결과, 검증된 performer_id 또는 unknown, target/vocabulary/schema version과 split을 담는다. 원본 text와 ID는 저장소 밖에 보관한다. NPZ는 임시파일→atomic replace하며 재실행 시 source/schema/config hash를 비교해 안전하게 재개한다. 단순 파일 존재 skip만으로 설정 변경을 놓치지 않도록 한다.

#16 sample 생성 후 시간 길이 `T_out >= U + R`를 검사한다. #17은 downsampling이 없어 T_out=T_input이다. raw53 자체를 #17에 전달하지 않고265/424폭 float32 특징으로 변환한다. `padding_mask=True`는 실제 연속 prefix, target은 blank 없는 packed int64, 길이는 CPU int64다. all-missing·정규화 후 usable0·라벨 미지원·짧은 길이는 사유별 집계한다. 초기 품질 기준과 합격 수는 실데이터에서 확정한다.

## 7. GPU 학습과 CER 평가 설계

| 항목 | 일반 수어 코드 | 지문자 재사용 및 후속 연결 |
| --- | --- | --- |
| 장치 | CPU, 선택 옵션 없음 | #17 `CTCTrainer(...,device='cuda:0')`, recognizer에도 장치 지정. CUDA 가용성 확인 후 명시적 오류 처리 |
| 배치 | 고정 Xbank, numpy permutation | #16 lazy samples/가변길이 padding, #17 packed GRU. train만 shuffle. 길이 bucket은 필요시 실측 후 연결 |
| 분할 | REAL signer fold | 검증된 전역 performer mapping으로 train/val/test manifest를 먼저 고정 |
| 손실 | CE | #17 `ctc_loss` 사용. CUDA logits를 CPU로 전송해 CTCLoss 계산하고 autograd로 CUDA 모델에 역전파 |
| clipping | 없음 | #17 finite gradient 검사, 기본 norm5 |
| optimizer·스케줄 | AdamW+cosine | #17 AdamW. scheduler·조기종료는 현재 제공되지 않으며 baseline에서 고정 epoch 사용 가능 |
| 체크포인트 | weights·설정 | #17 atomic last.pt/best.pt, optimizer·epoch/step·history·vocabulary/schema·Python/NumPy/Torch CPU/CUDA RNG |
| 재개 | 완료 실행 skip | #17 완료 epoch 경계에서 resume, 동일 장치 종류·CUDA topology/설정 계약. 중간 epoch는 마지막 완료 epoch부터 재실행 |
| epoch 기록 | val F1·stdout·JSON | #17 history train/val loss와 best validation-loss 저장. 새 runner가 epoch별 val CER·시간·학습률·장치·입출력수·제외수·split/source/schema hash를 JSONL/요약에 기록 |
| 모델 선택 | 검증 F1 | 초기 baseline은 #17의 validation-loss best 사용. val CER도 병기. CER best 채택은 별도 명시 설정과 checkpoint 정책 검증 후 적용 |
| 최종 평가 | acc/F1 | #17 recognizer→#15 greedy decode→renderer→NFC text CER. test는 모델·정책 선택 완료 후 평가 |

`seed_everything`은 모델 생성 전에 호출한다. #17은 결정적 알고리즘과 TF32 설정을 관리한다. 외부 DataLoader generator/worker RNG/sampler cursor는 체크포인트가 자동 보존하지 않으므로 초기 runner는 epoch 경계 seed·재현 가능한 순서를 사용한다. fit의 epochs는 추가 epoch 수다. 각 epoch factory가 새로운 iterable을 반환하도록 하고 실험별 전용 checkpoint_dir을 사용한다.

PR #17의 CUDA·재개 합성 테스트 성공은 기존 보고 기록이다. 이번에 기존 sign-language Conda Python으로 확인한 환경은 torch2.10.0+cu128, CUDA available=True, NVIDIA GeForce RTX5060 Laptop GPU다. 이번 작업은 실제 지문자 batch 역전파·처리량·메모리·CER를 검증하지 않았다. 실제 학습 전 GPU 메모리와 작은 실데이터 batch의 finite loss/gradient·저장/복원·추론 동등성·소규모 과적합을 확인한다. loss CPU 왕복 비용과 긴 시퀀스 메모리를 측정해 batch를 결정한다.

### 수행자 독립 평가

현재 CROWDxx 두 그룹과 음수 person_id로 실제 사람 분리를 입증할 수 없다. #12 `validate_signer_splits`와 #15 leakage 검사도 입력 ID의 provenance를 만들어 주지는 않는다. 공식 performer mapping 또는 검증된 metadata를 확보하고 모든 train/val/test에 동일 사람 namespace를 사용해야 한다. #16 `CTCSample.signer_verified`는 `signer_id is not None`을 뜻하므로 검증 전에는 signer_id=None을 유지하고 원본 CROWD 그룹은 별도 metadata에 보존한다.

수행자·영상 집합의 교집합0, 같은 영상의 재인코딩·중복 및 구간 파생본이 같은 split에 속하는지 확인한다. 기존 `make_folds`를 재사용할 때 n_folds≥3과 각 train/val/test 비어 있지 않음을 검사한다. 확정된 실제 수행자가2명뿐이면 세 집합의 수행자 독립 분할은 불가능하므로 추가 수행자 데이터가 필요하다. 현재 Validation ZIP 전체를 임의 분할해 공식 수행자 독립 평가라고 부르지 않는다. Training 자료와 held-out 자료의 실제 인물 중복도 확인한다.

#15 평가에 `training_performer_ids`, `training_video_ids`, `identities_verified=True`, `require_verified=True`를 제공한다. 결과는 corpus CER=`Σ(S+D+I)/ΣN`, 수행자별 CER, 영상별 CER·mean-video CER를 구분해 보고한다. 분모0의 undefined/Infinity는 JSON에서 명시 상태로 직렬화한다. 목표는 조합된 한글 문자열의 NFC 글자 CER이며 자모 CER은 별도 보조 지표로만 기록한다. 숫자·공백·문장부호 정책과 전체/정책별 subset coverage, 제외 사유를 함께 고정한다. 누락 prediction과 정상 빈 문자열을 구분한다.

37종 자모 표와 Unicode 분리 토큰이 일치한다는 보장은 없다. 복합 모음·쌍자음·겹받침·숫자의 실제 수어 표기와 출력 조합 정책을 검증해야 한다. #13은 한글 literal override를 거부하므로 실제 수어 토큰으로 재분해가 필요하면 기존 decompose 주입 지점에 검증된 어댑터를 연결한다. #11 composer는 불완전 예측 자모를 거부할 수 있고 #15 renderer는 composition_rejected 예외를 발생시키므로, 실제 예측에 대한 문자열 복원·오류 처리 정책도 필요하다. 조합 실패 영상을 CER 분모에서 조용히 제외하면 안 된다. 토큰 복원 정책과 실패 집계를 확정한 후 모든 평가 영상에 예측 문자열을 생성한다.

## 8. 최종 인터페이스와 팀 협의

Issue #2의 제공 계약은 `read_fingerspelling(xy, mask) -> str`, `xy(T,53,2)`, `mask(T,53)`, fps15다. 새 Issue #2 wrapper는 검증→학습과 동일 전처리/schema→#17 recognizer→#15 decoder/renderer를 묶는다. checkpoint와 vocabulary는 초기화 시 검증·캐시하고 모든 호출에서 파일을 다시 로드하지 않는다. 빈 입력·완전 결측·조합 실패 처리도 계약 테스트로 확정한다.

B 학습의 common51 정책을 채택하면 추론에서도 `adapt_mediapipe53→to_ctc53(allow_missing_inner_eyes=True)`로 동일 눈 슬롯을 제외한다. A 원본53 학습 모델과 B common51 모델을 구분할 schema를 checkpoint에 보존한다. 기존 BIO 입력까지 눈 마스크를 바꾸지 않도록 wrapper 내부에서 복사해 적용한다.

현재 실시간 루프는 tick을 건너뛰므로 수집 배열이 항상 균일15Hz라는 보장은 없다. xy/mask만 받는 최종 API는 유실 시각을 복구할 수 없다. #6 담당과 실제 tick 기준15Hz grid 재구성·누락 mask·버퍼 보존 정책을 협의하고 #2는 균일15Hz 입력 계약을 검증한다. 긴 지문자가10초 버퍼를 넘는 경우의 구간 보존과 BiGRU의 완료 구간 추론 지연도 협의한다.

협의 대상은 다음과 같다.

- #4 담당과 구간 종류·시작/끝 포함 규칙·원본 시각·지문자 전체 구간 보존을 합의한다.
- #6 담당과15Hz 유실 처리, 호출 위치·초기화·오류·빈 문자열 표시, 장치·latency 측정·긴 구간 버퍼를 합의한다. 현재 분류64-frame 변환을 CTC 입력에 적용하지 않는 호출 경로가 필요하다.
- 데이터/모델 담당과 SourceSpec 증거, 눈 결측, confidence 품질 기준, performer provenance, token37종/숫자/조합 정책, split·CER 목표값을 합의한다.

Issue #2가 wrapper와 호출 계약 예제를 제공하고 실제 realtime 변경은 #6 담당이 수행한다. 종류 라우팅은 #4/#6 계약을 이용한다. 이번 문서 작성 과정에서 팀원 코드 변경과 외부 메시지 전송은 없었다.

## 9. 개발 우선순위와 후속 브랜치·PR 단위

아래 파일은 후속 구현 후보이며 이번 변경에는 문서만 추가한다. 선행 PR 코드는 병합된 공용 모듈을 import해 사용한다.

### P0 소스 근거·품질 정책 확정

- 브랜치 후보 `feat/2-crowd-source-validation`의 후속 PR. #18 기반 stacked 구성 또는 #18 병합 후 main을 base로 선택한다.
- 목적은 실제 canvas·시간·좌우·confidence 의미와 수행자 metadata의 근거 확보다. 정상 설명서/가이드, 대응 영상 표본·metadata, performer mapping이 필요하다.
- #18 및 후속 `crowd_source_validation.py`, `fetch_crowd_docs.py`, `validate_crowd_source.py`를 재사용한다. evidence manifest와 overlay·품질정책 판정만 보강한다.
- 수정 후보는 위 Issue #2 전용 파일과 `docs/crowd_source_validation.md`, 해당 테스트다. 기존 일반 수어 파일 변경은 필요하지 않다.
- 테스트는 근거 부재 차단, 실제 대응 프레임/좌우/크기/시각, strict와 진단 분리, 원본 hash 불변성, 품질 제외 분포다.
- 충돌은 #18 parser/adapter 수정부에 집중된다. #18 작성자·데이터 담당과 base 및 정책을 협의한다. 이 단계만으로 학습 시작 조건은 충족되지 않는다.

### P1 실제15fps CTC 데이터 생성·통합

- 브랜치 후보 `feat/2-fingerspelling-dataset-integration`. #11/#12/#13/#16/#18과 source 검증 반영 후 구성하며 #17 output-length 계약을 연결 검증한다.
- 목적은 실제 npz/manifest와 사용·제외 수를 생성하는 것이다. 검증 source의 키포인트·라벨 또는 A 영상, 시간 metadata, 채택 token표·품질 정책이 필요하다.
- #12 exact-video join, #13 target, #18 geometry, `landmarks.select_frames/extract_video`, #16 sample/batch를 재사용한다.
- 추가 후보는 `src/fingerspelling/dataset_integration.py`, `tools/build_fingerspelling_dataset.py`, `tests/test_fingerspelling_dataset_integration.py`다. 새로운 parser·MediaPipe·보간 코드를 중복 작성하지 않는다.
- 새로운 부분은 source별 orchestration,15fps 선택·검증, 저장 schema와 재개·오류 집계, performer manifest다.
- 테스트는30/60/29.97→15의 선택·시각 오차, 미검증/VFR 차단 또는 명시 지원, 누락 index 보존, 눈 정책 일치, label exact join·숫자·CTC U+R, dtype·padding·원본 불변·설정 변경 재개다.
- 파일 충돌 가능성은 신규 경로로 낮추고 기존 API 변경은 소유자와 협의한다. 실제 입력>0과 split/token 검증을 통과해야 학습 단계로 진행한다.

### P2 실데이터 GPU 학습·수행자 독립 CER

- 브랜치 후보 `feat/2-fingerspelling-training-evaluation`. P1 및 #15/#17 반영에 의존한다.
- 목적은 #17 runner, 고정 수행자 split, epoch 로그·평가 산출물 연결이다. 학습/검증/시험의 실데이터와 검증 performer mapping, 동일 vocabulary/schema가 필요하다.
- #17 trainer/checkpoint/recognizer, #16 batch, #15 CER, `dataset.make_folds/sha`를 활용한다.
- 추가 후보는 `tools/train_fingerspelling.py`, `tools/evaluate_fingerspelling.py`, `src/fingerspelling/training_run.py`, 관련 runner 통합 테스트다. #17 내부 변경은 실제 통합에서 필요한 부분에 한정한다.
- 새 기능은 device/데이터/split CLI, epoch별 val CER·JSONL, metadata 저장, split 누출 검사, 실제 평가 orchestration이다.
- 테스트는 CUDA finite backward, 작은 실제 subset 과적합, epoch 재개와 연속 결과 비교, 소진 generator 방지, checkpoint schema 불일치, leakage·corpus CER oracle·조합 실패 포함 coverage다.
- #17 loss-history/checkpoint 정책과 충돌 가능성이 있어 모델 작성자와 협의한다. 최소 학습 smoke는 데이터·장치·loss 조건 충족 후, 완료 CER는 검증된 held-out 사람 확보 후 가능하다.

### P3 추론 wrapper와 인계

- 브랜치 후보 `feat/2-fingerspelling-reader`. P2의 checkpoint/schema와 #11/#15/#16/#17에 의존한다. B 정책이면 #18 어댑터도 필요하다.
- 목적은 최종 `read_fingerspelling` 제공이다. 확정 checkpoint/vocabulary/feature schema와 완료 구간 표본이 필요하다.
- 추가 후보는 `src/fingerspelling/reader.py`, `tests/test_fingerspelling_reader.py`, 호출 계약 문서다. 기존 전처리·추론·디코더를 연결한다.
- 테스트는 raw53→동일 특징→str, batch/single 일치, 눈 마스크·입력 불변, 빈/결측·불완전 자모·숫자 처리, CPU/CUDA·schema 검증·지연 측정이다.
- #4/#6의 파일을 수정하지 않는 독립 경로로 충돌을 줄인다. #6 담당의 별도 통합 PR에서 호출·유실 시각·버퍼·데모를 검증해야 Issue #2 최종 접점을 충족한다.

## 10. Issue #2 완료 기준과 학습 시작 판정

완료 기준은 다음과 같이 증거를 남긴다.

1. 좌표·53슬롯·좌우·15fps와 추출기/변환기 버전이 실제 자료로 검증되고, 학습·추론에 같은 schema가 적용된다.
2. 실제 CTC sample과 blank 없는 target이 생성되고 사용·제외 수, 품질·token coverage, U+R 검사 결과를 기록한다.
3. 공식 근거가 있는 performer namespace와 학습/검증/시험 비중복 manifest를 고정한다. Validation 외 학습자료와 추가 인물이 필요한지 확인한다.
4. 실제 CUDA batch 학습, checkpoint 저장·epoch 재개·추론을 검증하고 실데이터 학습 로그를 보존한다.
5. 학습에 없는 사람의 이름에 대한 NFC 문자열 CER, S/D/I/N, 수행자별 결과·coverage·정책·선택 checkpoint를 기록한다. Issue에 수치 목표가 없어 성공 threshold는 팀 합의 항목으로 둔다.
6. `read_fingerspelling(xy,mask)->str`와 호출 계약 테스트를 제공하고 #6 담당의 실제 realtime 호출 가능성을 확인한다. BIO 구간 탐지는 #4 담당이다.

**현재 실데이터 GPU 학습 시작은 보류다.** 장치 지원 코드가 있어도 verified CTC 입력이0개이며 source/time·품질·token 및 performer 조건이 미완료다. 합성 코드 시험과 실데이터 학습 시작을 구분한다. 먼저 P0/P1을 완료하고, 실제 학습 smoke와 수행자 독립 평가의 준비 조건을 각각 확인한다.

## 11. 독립 검토 기록

별도 리뷰 에이전트가 로컬 Git refs의 실제 코드와 문서를 독립 정적 검토했다. 코드 존재·실행 근거·검출기 차이·수행자 provenance·담당 범위·PR 의존·학습 시작 조건을 대조했다.

- 내부 공유 파서 함수명을 실제 `_parse_keypoint_json`으로 정정했다.
- #13 한글 재분해의 decompose 주입 지점, #16 signer_verified의 단순 non-None 판정과 unknown 유지, #15 조합 실패 예외와 CER 전수 포함 정책을 보강했다.
- 눈2/5 의미 정정, #16 resampling 부재, #17 CUDA 모델과 CPU CTC·loss best·epoch 재개, 검증 입력0, 조건부 B 우선순위와 #4/#6 담당 범위를 재확인했다.
- 재검토에서 잔여 필수 수정 사항은 발견되지 않았다. 사람의 리뷰와 실데이터 변환·학습·CER·기능 테스트 재실행은 수행하지 않았다.

최종 변경물은 이 문서1개다. 새 문서를 대상으로 whitespace diff 검사를 수행했고 오류가 없었다. 브랜치는 시작 상태를 유지하며 기존 추적 파일 변경은0개다. Commit, Push, PR 생성, Merge는 수행하지 않았다.
