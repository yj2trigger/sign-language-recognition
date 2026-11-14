# PR 제목

[feat] #2 MediaPipe 지문자 CTC 데이터 확장 및 손 상대좌표 추론 지원

# PR 본문

## 문제와 동작

기존 59개 MediaPipe 샘플은 Train에 자모 24종만 등장하고 Validation 자모 CER가
0.7583이었다. 검증된 기존 특징을 재사용하면서 Train의 미등장·희소 자모를 우선하는
영상 84개를 추가했다. Train123/Validation12/Test8에서 같은 BiGRU와 두 특징 표현을
비교하고, checkpoint 스키마에 명시된 손 상대좌표 표현을 공개 추론 API에 연결했다.

Refs #2. 기능 구현과 탐색적 성능 개선을 보고하는 PR이며 Issue를 닫지 않는다.
수행자 독립 평가와 영상·라벨 의미 검증은 미완료다. PR 생성·Merge는 수행하지 않았다.
기존 파이프라인 통합 범위와 선행 PR 출처는 `issue2_mediapipe_pr_draft.md` 및
`src/fingerspelling/provenance.md`에 보존한다. 이번 추가 변경은 아래 16개 파일이다.

## 파일별 변경과 실제 줄 범위

기준은 이 추가 변경 직전 `651983003edb29389dfaaa6836e6d5ca715f7690`이다.
신규 파일의 이전 범위는 없음이다. 기존 팀원 모듈과 OpenPose 실험 파일은 수정하지 않았다.

| 파일 | 이전 → 이후 줄 | 변경 이유 |
|---|---|---|
| `.gitattributes` | 4 뒤 → 5 | 손 표현 소스 SHA의 LF 고정 |
| `src/fingerspelling_runtime.py` | 14 뒤/63 뒤/66/79 뒤 → 15/65–70/73/87 | 알려진 손 표현 스키마만 로드하고 동일 추론 변환 적용 |
| `src/fingerspelling_hand_features.py` | 없음 → 1–38 | 손가락 상대좌표·손목 전역 이동·안전 마스크 |
| `tools/run_mediapipe_ctc.py` | 48/116–122/155–176/235/429–432 → 48–56/124–172/206–236/296/491–502 | CRC/해시 검증 특징 재사용, 명시 선택·품질·임시 경로, 제한된 journal 교체 재시도 |
| `tools/experiment_mediapipe_performance.py` | 148 뒤/163/218 → 149–150/165/220 | 학습·추론 export에 입력 스키마 전달 |
| `tools/expand_mediapipe_dataset.py` | 없음 → 1–173 | 전수 inventory, Train 희소 선택, 고정 holdout, 동결 |
| `tools/train_expanded_mediapipe.py` | 없음 → 1–61 | 두 표현 공정 비교·Val 선택·선택 후 Test 진단 |
| `tools/build_video_label_review.py` | 없음 → 1–139 | 원본 PTS 접촉 시트·uncertain CSV·중단 복구 |
| `tests/test_expand_mediapipe_dataset.py` | 없음 → 1–123 | 선택·누수·재사용·품질·IO 복구 |
| `tests/test_fingerspelling_hand_features.py` | 없음 → 1–84 | 이동/크기·마스크·극단 좌표·스키마 |
| `tests/test_video_label_review.py` | 없음 → 1–63 | PTS·전체 디코딩·중단/타인 파일 보존 |
| `docs/mediapipe_data_expansion.md` | 없음 → 1–181 | 실제 데이터·학습·추론·보존·리뷰·재현 보고 |
| `docs/fingerspelling_handoff.md` | 2 뒤 → 3–10 | 최신 연구용 checkpoint·성능 안내 |
| `docs/mediapipe_performance.md` | 2 뒤 → 3–5 | 이전 실험과 최신 실험 구분 |
| `docs/issue2_mediapipe_pr_draft.md` | 2 뒤 → 3–5 | 최신 PR 초안 연결 |
| `docs/issue2_expansion_pr_draft.md` | 없음 → 이 파일 전체 | 최신 변경 이유·줄 범위·검증·한계 |

## 실제 측정

2,000개 ZIP 영상과 2,000개 JSON을 매핑했다. 문자열 정책 적합1,616,
숫자 정책 필요382, 기타 지원 불가2다. 선택한 새96개 중84개 성공,
12개는 실제 FPS10.6263–13.8961로 기존15fps 최소 정책에 미달했다.
기존59개 특징/고유 영상은 보존됐다. 기존 보고의 “56개 특징”은 재현되지 않았다.
NPZ143개·고유 영상143개·Train123개를 구분한다. Train 자모 종류24→37이다.

| 측정 | 확장 기본 표현 | 확장 손 상대좌표 |
|---|---:|---:|
| 선택 epoch | 38 | 39 |
| 선택 Train 자모 CER | .032195 | .080000 |
| 선택 Validation 자모 CER | .325000 | .308333 |
| 최종 Train Loss(epoch40) | .123537 | .189019 |
| 최종 Validation Loss(epoch40) | 1.432314 | 1.540405 |
| 학습+epoch 평가 초 | 132.164 | 136.824 |
| 파라미터 | 158452 | 158452 |

설정은 동일 split·seed20261114·batch8·AdamW(lr.003,weight_decay.01)·40epoch다.
손 상대좌표 후보의 Test 진단 자모 CER .269841, resync 한글 CER는
Train.235154/Val.695652/Test진단.64다. Val 한글 완전 일치는0%, Test진단12.5%다.
기본 strict 한글 CER는 오히려 높아지므로 자모 개선을 실사용 한글 인식 성공으로 표현하지 않는다.
strict 기본값은 유지하고 resync만 명시적으로 선택한다.
두 표현의 Val 차이는 120자 중 오류2개이며 단일 시드다.
기존 최초 모델의 Train/Val/Test 자모 CER .8095/.8500/.8730 및 정확0%는 역사적 기준으로 보존했다.

원본 RGB 12개 접촉 시트·원문/자모 CSV를 비공개 D에 생성했다. 모든 라벨 uncertain,
자동 verified/교정0개다. 6개 정지 시각만으로 전체 자모 순서를 확인했다고 주장하지 않는다.
현재 자료에 신뢰 가능한 수행자 매핑이 없어 수행자 독립 평가를 수행하지 않았다.

## 검증

기존 Conda Python만 사용했다. 실행 디렉터리는 지정 D 저장소다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$env:TEMP="$repo\.tmp-issue2"; $env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
& $py -m pytest -q -p no:cacheprovider --basetemp "$repo\.tmp-issue2\regression-expanded-final"
```

- 전체402 passed/경고2,108.89초. 분리한 PR 코드만으로 관련109 passed/경고2,12.85초.
- 실제143개 특징의 재계산·해시, 두40epoch GPU 학습, export 동등성, CPU/GPU/공개 API12개 출력 일치.
- 49프레임 CPU 특징+추론5회 평균20.255ms. 전처리를 포함한 실시간 지표가 아니다.
- 기존 baseline79파일·이전 학습1,632파일·기존 미추적34파일·키포인트/라벨 ZIP SHA 불변.
- 독립 리뷰가 전체 CER·렌더링의 개별 예측 재합산 및 토큰 보존을 확인했다.
  LF 고정·중단 복구·heldout checkpoint SHA 지적을 수정했다.

상세 실행 명령과 D 출력 경로는 `docs/mediapipe_data_expansion.md`를 따른다.
원본 ZIP·MP4·라벨·NPZ·모델·학습 산출물·개인별 예측은 Commit하지 않는다.
기존 미추적 파일·환경·팀원 코드도 보존한다.

## 남은 작업

수행자 매핑을 공식 자료로 확보하고, 전문가가 원본 영상 전체와 라벨 의미·시각을 확인해야 한다.
현재 Test는 이전부터 관찰한 8개 진단 표본이며 독립 일반화 성능으로 해석하지 않는다.
전체2,000개 영상 전처리·복수 시드·외부 실시간 평가가 남아 있다.
연구용 후보만 제시하며 검증된 실사용 기본 모델로 승격하지 않는다.
