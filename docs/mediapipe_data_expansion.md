# Issue #2: 실제 영상 데이터 확장 및 손 표현 비교

2026-11-14, `feat/2-mediapipe-fingerspelling`. 이전 59개 특징과 체크포인트를 재사용·보존하고,
새 출력 폴더에서 실제 전처리와 RTX 5060 Laptop GPU 학습을 수행했다.
**기능 구현과 탐색적 성능 개선은 확인했지만 Issue 전체 완료나 실사용 검증을 뜻하지 않는다.**
수행자 독립 평가와 전문가의 영상·라벨 의미 검증은 미완료다.

## 데이터와 선택

원본 MP4 ZIP 2,000개와 형태소 JSON 2,000개를 영상 ID로 조사했다.
현재 자모 토큰 정책상 문자열 처리 가능 1,616개, 숫자 정책 필요 382개, 지원하지 않는 문자 2개다.
문자열 처리 가능은 영상·라벨 의미 검증 완료를 뜻하지 않는다.
이 수는 이전 OpenPose 학습 가능 1,606개와 다른 단계의 집계다.

| 항목 | 이전 데이터 | 확장 데이터 |
|---|---:|---:|
| NPZ 파일 / 고유 영상 | 59 / 59 | 143 / 143 |
| Train | 39 | 123 |
| Validation | 12 | 12 |
| Test | 8 | 8 |
| Train 자모 종류 | 24 | 37 |

기존 59개는 배열 재계산·해시·원본 ZIP 메타데이터·라벨·추출 설정 대조 후 복사했다.
새 후보 96개 중 84개가 성공했다. 나머지 12개는 실제 OpenCV FPS가
10.6263~13.8961로 기존 최소 15fps 정책에 미달했다. `invalid_video_metadata`로 기록되며
손상 영상이라고 단정하지 않는다. 실패 임시 MP4는 보존했다. 미처리 선택 후보는 0개다.
전체 2,000개의 영상 전처리를 완료한 것은 아니다.
이전 관찰의 “56개 파일”은 현재 기준 데이터에서 재현되지 않았고, 실제 기준은 59개다.

선택 시드 `20261114`, 추가 상한 96, 처리 예산 1,200초다.
기존 Train의 미등장·희소 토큰을 우선하고 파일 크기와 시드 순서로 동률을 해소했다.
Validation/Test의 토큰 빈도는 점수에 사용하지 않았다. 해당 영상·FS 그룹·NFC 문자열은
후보에서 제외했고 두 기존 split의 ID와 순서를 유지했다.
이는 **영상/단어 그룹 분리**이며 수행자 분리가 아니다.

공통 기준: 실제 PTS 전수 디코딩 검사, 목표 15Hz에서 중복 없는 실제 프레임 선택,
라벨 start/end 구간, CRC, 유한 좌표, CTC 반복 포함 최소 길이, 특징 재계산 일치다.
추가 품질 기준은 전체 손 42점의 관측 마스크 평균 ≥0.25와 양어깨 동시 관측 프레임 비율 ≥0.5다.
손 기준은 “양손 검출 프레임 25%”라는 뜻이 아니다. VFR을 균일한 15fps로 주장하지 않는다.

## 같은 데이터에서 비교한 표현

두 모델 모두 기존 BiGRU hidden64/2층, 424차원, 52클래스(blank0), batch8,
AdamW 학습률0.003/weight_decay0.01, dropout0, 40epoch, 시드20261114다. 표준화 통계는 Train만 사용하고
추론 첫 Linear에 접어 넣었다. Epoch 선택은 Validation 자모 CER→loss→이른 epoch 순이다.
모델 선택은 Validation 자모 CER→한글 CER 순이다. Test는 선택 후 기존 관찰된 진단용으로만 읽었다.

- `expanded-base`: 기존 어깨 중심/너비 정규화, xy·관측·보간·사용 가능·이동 채널.
- `expanded-hand-local`: 각 손가락을 손목 기준 및 손목–중지 MCP 거리로 정규화한다.
  손목 자체는 전역 어깨 정규화 위치를 유지하고 이동 채널을 다시 계산한다.
  손바닥 기준이 없거나 크기가 1e-6 이하면 해당 손가락 사용 가능 마스크를 내린다.
  기존 53점 입력 계약과 몸통 11점은 유지한다. 얼굴/좌우 반전/좌우 교환을 새로 추가하지 않는다.
  이는 이전 OpenPose 양손42/상반신48 모델 A/B와 다른 실험이다.

| 측정값 | 확장 기본 표현 | 확장 손 상대좌표 |
|---|---:|---:|
| 선택 epoch | 38 | 39 |
| 선택 시 Train 자모 CER | 0.032195 | 0.080000 |
| 선택 시 Validation 자모 CER | 0.325000 | **0.308333** |
| 선택 시 Train blank argmax 비율 | 0.781011 | 0.683197 |
| 선택 시 Validation blank argmax 비율 | 0.777637 | 0.726811 |
| 선택 시 Train 평균 예측 자모 길이 | 8.195122 | 7.780488 |
| 학습+epoch 평가 시간(초) | 132.164 | 136.824 |
| 선택 시 Validation strict 한글 CER | 1.847826 | 1.782609 |
| 최종 epoch40 Train Loss | 0.123537 | 0.189019 |
| 최종 epoch40 Validation Loss | 1.432314 | 1.540405 |
| 최종 epoch40 Train 자모 CER | 0.027317 | 0.094634 |
| 최종 epoch40 Validation 자모 CER | 0.358333 | 0.350000 |
| 파라미터 수 | 158,452 | 158,452 |

손 상대좌표 모델을 Validation 기준 후보로 선택했다. 차이는 자모 2개 오류
(39/120→37/120)에 불과하며 단일 시드·소규모 표본이므로 안정적 우월성으로 단정하지 않는다.
데이터 수까지 다른 기존 소규모 실험의 Validation 0.758333과 직접적인 표현 효과를 분리할 수 없다.
최종 epoch loss/CER와 선택 checkpoint의 CER를 혼동하지 않는다. 상세 epoch 기록은 `history.json`에 있다.

선택 모델의 기존 Test 진단 자모 CER는 **0.269841**(17/63), 삭제7/삽입0/치환10이다.
이는 독립 일반화 성능이 아니다. Train 0.08과 Validation 0.3083 차이에서 과적합 가능성이 남는다.

## 한글 조합과 실제 추론

자모 오류가 남으면 strict 조합이 전체 실패하여 원시 자모 문자열로 남을 수 있다.
선택 모델의 strict 한글 CER는 Train0.598575/Val1.782609/Test진단1.56이다.
1보다 큰 CER도 삽입 오류 때문에 가능하다.
기존 토큰 보존 `resync`를 Validation만으로 선택하면
Train0.235154/Val0.695652/Test진단0.64이며 한글 완전 일치율은
Train57.72%/Val0%/Test진단12.5%다. 자모를 추가·삭제하거나 정답 사전을 적용하지 않는다.
기본 렌더러는 여전히 strict이며 resync는 명시적 옵션이다.

추론 파일: `expanded-models-01/expanded-hand-local/inference.pt`.
SHA-256: `0ec9ded244354bf00ede986e66d7619b6b19d6b413996f6129f6b1815329e3f5`.
배포 파일은 추론 전용이며 학습 재개는 각 `training/last.pt`와 config/통계를 사용한다.
손 표현 버전·소스 SHA를 checkpoint에 넣고 Reader가 명시적으로 동일 변환을 적용한다.
기존 424차원 checkpoint는 기존 경로로 계속 로드한다. 해시 대상 파일은 LF를 고정했다.
실제 Validation 12개에서 CPU Reader, GPU Reader, 공개 `read_fingerspelling()`의
resync 문자열이 모두 일치했다. 한 49프레임 입력의 CPU 특징+추론 5회 평균은
0.020255초였다. MP4 디코딩/MediaPipe 추출을 포함한 실시간 처리율 측정은 아니다.
기존 baseline 79개 파일, 이전 OpenPose 학습 산출물 1,632개, 백업 manifest의
기존 미추적 파일 34개 및 원본 keypoint/morpheme ZIP SHA가 일치했다.
원본 영상 ZIP은 크기/mtime과 사용한 개별 MP4 CRC를 확인했다.
기존 전체 영상 ZIP SHA가 없어 전체 ZIP 바이트 전후 일치를 주장하지 않는다.

## 원본 영상·라벨 검토

AI-Hub 공식 출처: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103 .
공식 페이지는 CROWD 수집과 형태소 표제어를 설명하지만, 현재 확보한 자료만으로
개별 영상의 모든 자모 순서 전사를 확정하지 못했다. 설명서 다운로드 링크의 실제 PDF는
확보하지 못해 읽었다고 주장하지 않는다. 실제 JSON의 metaData(url/name/duration/exportedOn),
data(start/end/attributes.name)에는 검증된 수행자 ID나 자모별 시각이 없다.
CROWD18/19를 사람 ID로 간주하지 않는다. 의미 판정은 **PARTIAL**, 검증 완료 샘플 0개다.

Train 희소 토큰 우선 12개를 원본 RGB에서 각 6개 실제 PTS로 접촉 시트화했다.
캐시의 16:9 패딩을 역변환해 손 영역을 자르고 좌우를 뒤집지 않았다.
12개 모두 CRC/전체 디코딩/PTS 대조/JPEG 재읽기 성공, 실패0개다.
생성한 접촉 시트 한 개를 직접 확인해 손 영역과 시각 표시를 확인했다.
6개 정지 프레임은 전체 철자 순서를 증명하지 않는다. 자동 의미 판정을 하지 않았다.
전문가 CSV의 상태는 모두 uncertain, corrected_label은 빈 값이다.
이전 40개 검토 CSV와 기존 학습 라벨도 수정하지 않았다.

비공개 D 출력 `video-review-expanded-01/index.html`, `expert-review.csv`,
`review-progress.json`에 실제 12개 ID·원문·자모·시각·이미지 해시가 있다.
영상·라벨·개별 예측·좌표는 공개 저장소에 넣지 않는다.
전문가는 원본 영상 전체와 표제어를 함께 보고 실제 철자 여부, 시간 구간, 반복/쉼,
수행자 매핑 근거를 확인해야 한다. 확인 전 수행자 독립 CER는 계산하지 않는다.

## 재현과 결과 위치

기존 출력은 보존한다. 아래 `*-new`는 아직 없는 출력 경로로 바꾼다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$crowd='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$area="$crowd\outputs\mediapipe-ctc-performance"
$env:TEMP="$area\controlled-01\temp"; $env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
& $py tools/expand_mediapipe_dataset.py --source "$crowd\outputs\mediapipe-ctc-v1\pts-run-01" `
  --videos "$crowd\original-video\downloads\수어 영상\2.Validation\[원천]01_crowd_video.zip" `
  --labels "$crowd\수어 영상\2.Validation\[라벨]01_crowd_morpheme.zip" `
  --output "$area\data-expansion-new" --additional 96 --seconds 1200 --seed 20261114
& $py tools/train_expanded_mediapipe.py --source "$area\data-expansion-01" --output "$area\expanded-models-new" --epochs 40
& $py tools/build_video_label_review.py --dataset "$area\data-expansion-01" `
  --videos "$crowd\original-video\downloads\수어 영상\2.Validation\[원천]01_crowd_video.zip" `
  --output "$area\video-review-expanded-new" --count 12
& $py tools/compare_mediapipe_rendering.py --source "$area\data-expansion-01" `
  --experiment "$area\expanded-models-01" --output "$area\expanded-models-01\rendering-new.json"
& $py -m pytest -q -p no:cacheprovider --basetemp 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\tests-new'
```

실제 결과는 `data-expansion-01/expansion-summary.json`, `inventory-final.jsonl`,
`expanded-models-01/{summary,selection,heldout-diagnostic,rendering-comparison,verification}.json`,
각 모델의 `history.json`, `result.json`에 있다. 모두 D드라이브에 저장한다.

## 장애 대응과 미완료

기존 특징 재사용 중 Windows 임시 journal 교체의 PermissionError가 있었다.
이전 progress/selection/실행 코드를 별도 파일로 보존하고 56개 완료 특징의 해시와
전체 선택 ID가 같음을 확인한 뒤, PermissionError에만 제한된 8회 재시도를 적용했다.
원본·설정 변화는 여전히 실패 처리한다. ACL·보안 설정은 바꾸지 않았다.
검토 이미지 도구도 소유 sidecar+CRC를 확인해 완료 임시 MP4를 재사용하고,
부분/타인 파일과 불일치 orphan JPG를 보존한 채 새 파일명으로 복구한다.

라벨 의미·수행자 매핑·전체 2,000개 영상 전처리·다중 시드 안정성·실시간 외부 입력 평가는
남은 작업이다. 현 후보는 연구용이며 실사용 기본 모델로 승격하지 않는다.

## 테스트와 독립 리뷰

최종 전체 회귀 **402 passed, 2 warnings, 108.89초**.
위 pytest 명령의 basetemp를 `.tmp-issue2/regression-expanded-final`로 실행했으며,
로그는 `.tmp-issue2/regression-expanded-final.log`에 있다. 경고는 기존 protobuf deprecation 2개다.
기존 미추적 실험 모듈을 제외하고 Git 추적 코드와 이번 추가 코드만 복사한 폴더에서도
지문자 관련 5개 테스트 파일 **109 passed, 2 warnings, 12.85초**를 확인했다.
로그는 `.tmp-issue2/clean-expanded-01.log`다.

새 합성 테스트 31개는 Train만 사용하는 선택·누수 차단, 손실 없는 특징 재사용,
품질 마스크·이동·크기 변화·0 palm·NaN/Inf·극단 overflow·스키마 불일치,
실제 PTS 매칭 계약·전체 디코딩·중단 후 소유/CRC 복구·타인 파일 보존을 검사한다.
합성 테스트와 실제 84개 추가 전처리·두 GPU 학습·143개 특징 감사는 구분한다.

독립 리뷰에서 희소 선택/분할, 손목 전역 이동 보존, 학습·추론 변환 일치를 점검했고,
손 표현 LF 고정·중단 복구·기존 heldout checkpoint SHA 대조 지적을 반영했다.
모든 모델 및 렌더러의 Train/Val/Test CER를 개별 예측에서 독립 재합산했고 토큰 왕복도 일치했다.
실제 optimizer 표기는 AdamW로 바로잡았다. 확인된 필수 코드 지적은 모두 반영했다.
