# AI-Hub CROWD 실제 키포인트 검증 실험

조사일은 2026-11-13이며 시간대는 Asia/Seoul이다. Refs #2.

## 작업 기준과 사용 데이터

현재 브랜치는 `experiment/2-crowd-keypoint-validation`이다. 시작 HEAD와 main은
`c627cf77eb55564fb1a85ece63628d5bba67bb9a`이며 시작 작업 트리는 깨끗했다.
Issue #2 원문에서 `xy(T,53,2)`, `mask(T,53)`, 15fps 입력 계약을 확인했다.
main의 `src/landmarks.py`, `src/build_dataset.py`, `src/preprocess.py`를 직접 읽었다.
PR #18·#19·#20은 모두 열려 있고 아래 로컬 ref와 GitHub head SHA가 일치했다.
[Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2)

| 검토 대상 | 로컬 ref | 확인한 head |
| --- | --- | --- |
| PR #18 어댑터 | feat/2-crowd-keypoint-adapter | d71a74beaa3026c8018074f14476df2e9d722513 |
| PR #19 소스 검증 | feat/2-crowd-source-validation | f7631c32b734abc1027329b8d29a9d520fc82987 |
| PR #20 학습·통합 분석 | research/2-fingerspelling-training-integration | b5069789113446679513a2da0214319f41f7e93a |

기존 `docs/fingerspelling_crowd_keypoints.md`, `docs/crowd_source_validation.md`,
`docs/fingerspelling_training_integration_plan.md`를 해당 ref에서 읽었다.
#18의 파서·geometry·시간 보존, #19의 공유 진단 파서와 confidence 집계,
#20의 A/B·15fps 연결 공백 분석을 실제 코드와 대조했다.
[PR #18](https://github.com/yj2trigger/sign-language-recognition/pull/18),
[PR #19](https://github.com/yj2trigger/sign-language-recognition/pull/19),
[PR #20](https://github.com/yj2trigger/sign-language-recognition/pull/20)

데이터 루트는 `D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd`이다.

| 실제 입력 | 루트 아래 상대 경로 | 크기 bytes |
| --- | --- | ---: |
| 키포인트 | downloads\수어 영상\2.Validation\[라벨]01_crowd_keypoint.zip | 1,411,121,080 |
| 형태소 | 수어 영상\2.Validation\[라벨]01_crowd_morpheme.zip | 952,783 |

ZIP을 `rb`와 ZipFile 기본 읽기 모드로 열고 압축 해제 없이 member를 읽었다.
파서 구현은 #19의 고정 SHA에서 `git show`로 읽어 격리된 메모리 모듈로 재사용했다.
geometry 어댑터도 같은 SHA에서 로드하지만 실데이터 변환은 호출하지 않았다.
현재 브랜치에 선행 PR을 가져오는 작업은 수행하지 않았다.

이번 추가 범위는 x/y별 histogram, 관절별 통계, 손목 필드 간 거리, 라벨 metadata 검사다.
#19의 confidence 처리 정책과 파서를 재구현하지 않았다.
metadata 검사는 영상명 exact join만 확인하며 원문 정답·자모 target 의미는 평가하지 않는다.

## 실제 전수 검사 결과

| 항목 | 이번 측정 |
| --- | ---: |
| 프레임 JSON | 671,745 |
| 구조·유한값 검사 통과 | 671,744 |
| JSON 문법 오류 | 1 |
| confidence 범위 위반 프레임 | 142,044 |
| 엄격 파서 기준 통과 프레임 | 529,700 |
| 영상 그룹 | 2,000 |
| 모든 프레임 엄격 검증 통과 그룹 | 446 |
| frame index가 0에서 시작하는 그룹 | 2,000 |
| 누락 index / 중복 index | 0 / 0 |
| 영상별 프레임 수 최소 / 25% / 중앙 / 75% / 최대 | 12 / 188.75 / 261 / 480 / 974 |
| 라벨 metadata 영상명 / 중복명 | 2,000 / 0 |
| 키포인트·라벨 exact-video 일치 / 양방향 누락 | 2,000 / 0 |
| 라벨 duration 최소 / 최대, 초 | 7.55 / 16.234 |
| 실제 CTC 입력 생성 | 0 |

중복 frame/member는 재사용한 ZIP 인덱서가 예외로 차단한다. 이번 인덱싱은 성공했다.
문법 오류 1개를 제외한 모든 프레임을 좌표·confidence 통계에 포함했다.
모든 member를 끝까지 읽어 zipfile CRC 검사를 거쳤고 읽기 예외는 없었다.
숫자형·구조 통과와 실제 좌표·시간 의미의 검증은 각각 따로 기록한다.

검사 전후 SHA-256은 두 ZIP 모두 동일했다.

```text
keypoint  91803b0969a00b45f418ec407c6594d14aef3daec816afd7998ecaff66fe36b1
morpheme  1d0874d0f63acd7c42a9c1fe762c4f951b92ff526ca02e6596e50324ef84fbe3
```

## 좌표 범위와 분포

아래 범위와 histogram은 confidence가 0 또는 범위 밖인 관절도 포함한 원시 유한 좌표다.
원시 좌표 단위를 유지했다. clipping·정규화·손 교환은 적용하지 않았다.

| 필드 | x 최소 | x 최대 | y 최소 | y 최대 |
| --- | ---: | ---: | ---: | ---: |
| Pose25 | 0 | 1074.46 | 0 | 1920 |
| Face70 | 224.18 | 1068.77 | 162.513 | 934.478 |
| left hand21 | 0 | 1221.6 | 0 | 2159.27 |
| right hand21 | -151.61 | 885.905 | 0 | 2225.99 |

고정 구간 histogram의 각 행은 전체 관절 좌표 개수다. 구간은 왼쪽 경계를 포함하고
오른쪽 경계를 제외한다. 2560 이상은 모두 0개였다.

| 필드·축 | <0 | [0,1) | [1,256) | [256,512) | [512,768) | [768,1024) | [1024,1280) | [1280,1920) | [1920,2560) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pose x | 0 | 5185907 | 529987 | 3964672 | 5384576 | 1727044 | 1414 | 0 | 0 |
| pose y | 0 | 5185907 | 0 | 780688 | 2735967 | 2150002 | 1203353 | 4520771 | 216912 |
| face x | 0 | 0 | 71 | 11275327 | 35740412 | 6238 | 32 | 0 | 0 |
| face y | 0 | 0 | 5957 | 9304637 | 33954754 | 3756732 | 0 | 0 | 0 |
| left x | 0 | 1953 | 0 | 14111 | 2878868 | 11188254 | 23438 | 0 | 0 |
| left y | 0 | 1953 | 0 | 5276 | 98343 | 131015 | 130826 | 13251988 | 487223 |
| right x | 2390 | 29226 | 2765133 | 10411881 | 896968 | 1026 | 0 | 0 | 0 |
| right y | 0 | 29211 | 0 | 54792 | 1535927 | 3398481 | 1763003 | 7101854 | 223356 |

관찰상 x/y의 범위가 다르고 정규화된0–1 좌표로 취급할 수 없다.
1080×1920 canvas의 pixel 좌표와 유사한 분포라는 해석은 추정C에 해당한다.
손 좌표에는 그 가설의 범위를 벗어나는 값도 있어 분포만으로 canvas를 확정하지 않는다.
OpenPose 공식은 여러 `keypoint_scale`을 허용하므로 형식 일치만으로 scale을 특정할 수 없다.
[OpenPose 출력 명세](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md)

### 관절별 패턴과 좌우

관절별 통계는 confidence>0인 값에 대해 계산했다. confidence>1도 측정에 포함된다.
각 관절의 count·x/y 평균·최소·최대는 로컬 결과 JSON에 모두 기록된다.
아래 관절명은 OpenPose BODY_25 적용을 가정한 해석이며 CROWD export 의미는 미확정이다.

| 필드·index | 의미 후보 | 평균 x | 평균 y |
| --- | --- | ---: | ---: |
| pose 0 | 코 | 552.626 | 588.258 |
| pose 2 / 5 | 오른 / 왼 어깨 | 346.733 / 771.856 | 860.714 / 864.674 |
| pose 4 / 7 | 오른 / 왼 손목 | 291.112 / 846.656 | 1254.522 / 1510.997 |
| pose 15 / 16 | 오른 / 왼 눈 | 503.368 / 600.298 | 547.777 / 544.001 |
| pose 17 / 18 | 오른 / 왼 귀 | 442.227 / 663.909 | 596.083 / 586.904 |
| left hand 0 | 왼 손목 | 841.270 | 1516.141 |
| right hand 0 | 오른 손목 | 298.147 | 1251.221 |

양쪽 pose 손목과 해당 hand0의 confidence가 모두 양수인 프레임은 각 670,127개다.
left hand0에서 pose7/pose4까지의 거리 중앙값은 각각 10.675/638.962이고,
right hand0에서는 640.721/13.790이다. 단위는 원시 좌표 단위다.
이는 필드 내부 연결의 관찰C 근거다. 실제 사람의 해부학적 좌우·영상 반전·손가락 순서는
영상 대조가 필요하다. 얼굴·어깨·손목의 평균 위치 관계도 축 방향 확정 근거로 사용하지 않는다.

## 근거 수준과 metadata

A는 공식 명세 확정, B는 실제 대응 영상 대조, C는 관찰 기반 추정, D는 확인 불가다.

| 검증 항목 | 수준 | 근거와 한계 |
| --- | --- | --- |
| Pose25·Face70·양손21 및 x/y/confidence 필드 | A | AI-Hub 공개 설명과 실제 배열 길이 일치 |
| OpenPose 일반 BODY_25·Hand21 인덱스 | A | OpenPose 일반 명세. CROWD 적용 여부는 C |
| CROWD의 실제 관절 의미 | C | 배열 형태·신체 위치·손목 필드 거리 일치 관찰 |
| CROWD 좌표 단위·원점·축·해상도·crop/padding/rotation | D | 실제 영상과 export 설정 없음 |
| 1080×1920 pixel 가설 | C | 좌표 분포 기반 가설 |
| 해부학적 좌우·반전 여부 | D | 손 필드와 pose 연결 관계만 C |
| 개별 영상 FPS·keypoint sampling rate·index→영상 frame 대응 | D | 원본·PTS·추출 설정 없음 |
| 실제 영상 대조 | B 확보 0건 | 대응 영상 미확보 |

AI-Hub 공개 페이지는 데이터셋 전반에 대해 30fps 분할 이미지의 키포인트를 설명한다.
이는 A 수준의 일반 설명이며 이번 ZIP 각각의 index→시각 규칙이나 원본 FPS를 확정하지 못한다.
Issue #2의 60fps·세로 해상도 설명도 개별 파일 검증 근거로 채택하지 않았다.
[AI-Hub 데이터 구조 및 통계](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)

실제 정상 JSON의 top-level 필드는 version·people이고 585,816개에는 camparam도 있다.
people에는 person_id, 2D/3D 관절 배열만 존재했다. FPS·timestamp·W/H 필드는 관찰되지 않았다.
camparam은 nonempty dict로 585,816개에 존재하지만 내용이 영상 metadata임을 입증하지 못한다.
이번 첫 member의 Intrinsics.data·CameraMatrix.data·Distortion.rows/data는 모두 빈 문자열이었다.
기존 `source-metadata-validation.json`의 전수 결과도 nonempty leaf=0이다.
camparam leaf 전수 검사는 기존 결과를 인용했으며 이번 도구는 container 타입·존재 횟수를 재측정했다.

형태소 metadata 2,000개는 url·name·duration·exportedOn 필드만 갖는다.
duration으로 FPS를 역산하지 않았다. 프레임 번호는 파일명의 12자리 정수이고
모든 영상 그룹에서 0부터 연속한다. 프레임 누락 없음은 시각 간격의 균일성을 입증하지 못한다.
실제 시간축과 15fps 변환 성공 수는 각각 미확인과 0이다.

기존 공식 문서 다운로드 provenance는 두 파일 모두 79-byte HTML 접근 오류를 기록한다.
해당 로컬 파일과 provenance를 다시 읽었으며 정상 PDF/가이드는 확보되지 않았다.
이번에는 공개 페이지와 공식 파일 목록을 조회했다.

## 원본 영상 확보

`D:\UNIVERSITY\UAI_project2\sign-language-data` 전체를 재귀 조사했다.
파일은 확장자 없음 2개, HTML 2개, JSON 4개, ZIP 2개였고 영상 파일은 0개였다.
두 ZIP은 위 라벨 ZIP이다. 원본 영상 확보와 overlay 대조가 필요하다.

공식 페이지 JavaScript의 `ajaxS3FileList`를 읽고 실제 목록 조회 주소를 확인했다.
응답 `success=true`의 파일명·분할·fileSn·byte size를 2026-11-13에 조회했다.
초기 시도한 api.aihub.or.kr의 `/down/0.5/datasets/103/files`는 페이지 없음 응답이었다.
사용한 목록 근거는 아래 공식 페이지의 실제 조회 endpoint다.
[AI-Hub 공식 파일 목록](https://aihub.or.kr/aihubdata/data/aJaxS3FileList.do?dataSetSn=103&dataPckageSn=)

| 구분 | fileSn | 표시 파일명 | 목록의 크기 bytes |
| --- | ---: | --- | ---: |
| Validation keypoint | 39474 | [라벨]01_crowd_keypoint.zip | 1,411,121,080 |
| Validation morpheme | 39475 | [라벨]01_crowd_morpheme.zip | 952,783 |
| **필요한 Validation 원천 영상 후보** | **39482** | **[원천]01_crowd_video.zip** | **22,443,361,135** |
| Training 원천 01 | 39619 | [원천]01_crowd_video.zip | 31,120,167,329 |
| Training 원천 02 | 39620 | [원천]02_crowd_video.zip | 32,570,559,507 |
| Training 원천 03 | 39621 | [원천]03_crowd_video.zip | 33,754,208,721 |
| Training 원천 04 | 39622 | [원천]04_crowd_video.zip | 33,051,907,070 |
| Training 원천 05 | 39623 | [원천]05_crowd_video.zip | 32,758,486,798 |
| Training 원천 06 | 39624 | [원천]06_crowd_video.zip | 21,596,278,934 |

39482의 공식 경로는 `004.수어영상/2.Validation/원천데이터/CROWD/01_crowd_video.zip`이다.
예상 다운로드 용량은 목록 기준 약22.44GB, 20.90GiB다. 압축 해제 크기는 미확인이다.
대응 대상은 이번 Validation의 `NIA_SL_FSdddd_CROWDcc_F` 2,000개 영상 그룹이다.
라벨에서 관찰된 CROWD 그룹은 18·19다. 실제 원천 ZIP member 목록을 아직 읽지 못했으므로
2,000개 포함 여부와 exact basename 일치는 확보 후 검증해야 한다.
목적은 W/H·rotation·FPS/PTS·프레임 대응 및 상체·양손 overlay 검증이다.
원천 영상 파일을 다운로드하지 않았다.

## MediaPipe 53점 및 방법 A/B 비교

main은 pose 인덱스 `[0,2,5,7,8,11,12,13,14,15,16]`과 left21/right21을 사용한다.
`process_frame`은 원본을 fit_16x9로 padding한 뒤 MediaPipe에 넣는다.
좌표는 그 canvas 기준 정규화 좌표이고 mask는 pose visibility≥0.5와 hand 검출 여부다.
`build_dataset.py`는 WORD/REAL 정면 영상과 첫 형태소 구간을 대상으로 한다.
CROWD/FS 라벨에 바로 적용할 수 있는 영상명·정답 연결 구현은 현재 main에 없다.
`--full-fps 15` 경로는 실제 프레임을 선택하고 t·src_fps·wh를 저장한다.
원본 FPS 조회 실패 시 main에 존재하는 30fps fallback은 이번 검증의 근거로 사용하지 않았다.

| 항목 | 방법 A, 영상→MediaPipe | 방법 B, 제공 키포인트→어댑터 |
| --- | --- | --- |
| 실제 입력 확보 | 영상 0개 | 671,745 JSON 확보 |
| 실제 이번 추출·변환 | 미실행 | 근거 부족으로 차단 |
| 기존 53점 접점 | process_frame의 원래53점 | 공통51점과 눈2슬롯 결측 정책 필요 |
| 좌표 | fit_16x9 뒤 정규화 | 원시 좌표 단위·canvas 검증 후 동일 padding 필요 |
| 시간 | 영상 FPS/PTS를 확보 후 확인 | keypoint sampling·index↔PTS 검증 필요 |
| mask | MediaPipe visibility/검출 | 제공 confidence 기반. 의미·보정 동등성 미확인 |
| z·얼굴33점 | main의 추출 경로 존재 | 3D 배열은 비어 있음. 얼굴70→MediaPipe33 대응 미검증 |
| 실제 A/B 오차·속도·CER | 미측정 | 미측정 |

#18의 BODY25 `[0,18,17,5,2,6,3,7,4]`는53슬롯 `[0,3,4,5,6,7,8,9,10]`에
연결되며 손0–20은11–31/32–52에 연결된다. 이 대응은 SourceSpec 검증을 전제로 한다.
MediaPipe pose2/5는 LEFT_EYE/RIGHT_EYE이고 OpenPose 눈15/16의 export-specific 일치는
검증되지 않았다. `to_ctc53(allow_missing_inner_eyes=True)`는 눈 슬롯1/2를0/False로 둔다.
따라서 shape53 생성 API가 있어도 모든53관절의 의미 일치는 입증되지 않았다.
이번에는 이 opt-in을 실데이터에 적용하지 않았다.

main의 `preprocess.build_features`는 어깨 중점/너비 정규화, 결측 보간, 고정 길이 resample을 한다.
이 고정 길이 resample은 CTC 입력의15Hz 시간 검증을 대신할 수 없다.
#18은 index/fps 또는 명시 timestamps를 보존하고15fps로 재표본화하지 않는다.
PR #16의 CTC 특징 연결도 현재 브랜치에는 없다. 전처리·target·batch 통합은 후속 작업이다.
두 검출기의 성능·좌표 동등성·GPU 처리량에 대한 비교 결론은 실제 영상 실험 전까지 추정이다.

## Confidence 영향과 실제 변환 차단

| 필드 | confidence 최소 | 최대 | [0,1] 밖 값 개수 |
| --- | ---: | ---: | ---: |
| pose | 0 | 1.01479 | 534 |
| face | -0.000175198 | 1.11719 | 181,300 |
| left hand | 0 | 1.17766 | 16,380 |
| right hand | -0.0000185549 | 1.20531 | 12,432 |

총210,646개 범위 밖 값이142,044프레임에 있다. 생성 원인과 허용 범위는 미확인이다.
strict 파서는 최종53점에서 사용하지 않는 face 오류도 거부한다.
현재 정책으로1,554개 그룹은 완전한 엄격 시퀀스로 통과하지 못한다.
나머지446개도 좌표계·좌우·시간 근거가 없어 변환을 진행할 수 없다.
좌표 범위 밖 값도 추후 검증된 canvas와 confidence mask에 따라 별도 검사해야 한다.
Confidence clipping, 오류 프레임 삭제, 손 교환, 60/30fps 가정은 적용하지 않았다.

실제 변환 호출0회, 성공0개이며 입력 형상·15fps·target을 갖춘 실제 CTC sample은 없다.
최초 차단 원인은 `source_spec_missing`이며 좌표·canvas·BODY25/Hand21 의미·좌우 근거와
keypoint 시간축이 미확정이다. 그 뒤에도 confidence 정책·눈 결측 일관성·target 연결이 필요하다.

현재 환경의 torch는2.10.0+cu128, CUDA available=True,
장치는 NVIDIA GeForce RTX5060 Laptop GPU다. 실제 학습 입력0개이므로
이번 데이터로 GPU 학습을 시작할 수 없다. 실제 역전파·학습·CER는 미실행이다.

## 재현 명령과 산출물

기존 Conda Python3.12 환경을 사용한다. PATH의 기본 python은 실행 불가했다.
도구는 해당 Git object와 NumPy가 필요하며 checkout 브랜치를 바꾸지 않는다.
결과 JSON은 재생성 가능한 로컬 진단 자료다. 원본 라벨·개별 좌표·영상 ID 목록은 출력하지 않는다.

```powershell
$taskPython = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$crowdRoot = 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$keypointZip = Join-Path $crowdRoot 'downloads\수어 영상\2.Validation\[라벨]01_crowd_keypoint.zip'
$morphemeZip = Join-Path $crowdRoot '수어 영상\2.Validation\[라벨]01_crowd_morpheme.zip'
& $taskPython -m pytest -q
& $taskPython tools/experiment_crowd_keypoints.py $keypointZip --morpheme-zip $morphemeZip |
    Out-File -Encoding utf8 .pytest_cache/crowd-experiment.json
# metadata만 재검사할 경우
& $taskPython tools/experiment_crowd_keypoints.py $keypointZip --morpheme-zip $morphemeZip --metadata-only
# 영상 보유 현황
Get-ChildItem -LiteralPath 'D:\UNIVERSITY\UAI_project2\sign-language-data' -Recurse -File |
    Group-Object Extension | Select-Object Name,Count
# 공식 목록의 CROWD 항목 확인, 네트워크 접근 필요
$listing = (Invoke-WebRequest -UseBasicParsing -Uri 'https://aihub.or.kr/aihubdata/data/aJaxS3FileList.do?dataSetSn=103&dataPckageSn=').Content | ConvertFrom-Json
$listing.list | Where-Object { $_.streFileNm -match 'crowd' } |
    Select-Object fileSn,rootNm,streFileNm,fileSize,fileStreCours
```

이번 전수 측정은 geometry 도구를 먼저 실행하고 metadata를 `--metadata-only`로 별도 실행했다.
각 결과는 `.pytest_cache/crowd-experiment.json`, `.pytest_cache/crowd-label-metadata.json`에 있다.
실행 중 source commit 고정은 동일하며 후속 metadata 옵션 추가가 좌표 측정 로직을 변경하지 않았다.
GitHub에 전달할 변경물은 도구·합성 테스트·이 문서의3개 파일이다.

## 검증과 다음 구현 방향

합성 테스트는 정상 좌표, histogram 경계, confidence0/1/1초과, 빈 배열, frame gap,
NaN/Inf/극단 정수, metadata duration 오류, exact basename 연결, source 근거 부재 차단,
원본 ZIP bytes·SHA 불변을 검사한다. 실제 데이터 검사는 위671,745프레임 전수 측정이다.
전체 테스트와 독립 리뷰 결과는 다음과 같다.

전체 테스트는 **41 passed, 2 warnings**, 91.47초다. 신규 합성 테스트는14개다.
경고2건은 기존 protobuf의 Python3.14 관련 DeprecationWarning이다.
실제 전수 검사와 형태소 metadata 검사는 각각 정상 종료했고 두 ZIP의 전후 hash가 일치했다.
`git diff --check`도 오류 없이 통과했다. 신규 파일은 별도 내용 검토 대상으로 포함했다.

별도 리뷰 에이전트가 코드·테스트·문서를 독립 검토하고 실제 측정 JSON의 수치와 대조했다.
신규14개 테스트를 독립 재실행해 **14 passed**, 5.41초를 확인했다.
좌표 범위·histogram·손목 거리·confidence·metadata join·hash 및 근거 수준 해석을 확인했고
최종 필수 수정 사항은0건이었다. 사람의 리뷰는 수행하지 않았다.

이번 로컬 진단 JSON의 SHA-256은 다음과 같다. PowerShell Out-File UTF-8 출력 bytes 기준이다.

```text
crowd-experiment.json      f5e95678fc88f537aeb6452ad935eba7aaf8268651f08651a2e3856c1163c19f
crowd-label-metadata.json  9c4315e1a7eca44f7c46f9d6a83f1f7539e01a3496e91edfd901b7e5b73aa1ad
```

다음 구현은39482 원천 ZIP 또는 동일 영상 표본을 확보해 실제 basename과 W/H·rotation·PTS를
검증하는 단계다. 시작·중간·끝과 손 교차 구간 overlay로 관절·좌우·좌표 scale 및
JSON index 대응을 확인하고 근거의 적용 범위를 기록한다.
그 뒤 SourceSpec·시간 근거를 제공하고 confidence·눈 정책을 확정해 #18을 호출한다.
검증된 source rate에서15fps 선택과 원본 시각 보존을 구현한 후 CTC target·batch를 연결한다.
GPU 학습은 실제 sample 생성·유한 loss/gradient 검증 후 진행할 수 있다.
수행자 독립 CER에는 별도의 검증된 인물 mapping과 held-out 분할이 필요하다.

브랜치 변경, 기존 추적 코드 삭제·덮어쓰기, Commit, Push, PR 생성, Merge는 수행하지 않았다.
