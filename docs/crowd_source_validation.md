# CROWD 공식 명세 확보 및 실제 데이터 재검증

조사일은 2026-11-13, 시간대는 Asia/Seoul이다. 관련 이슈는 Refs #2이며 선행 PR은 #18이다.

## 저장소와 적용 범위

시작 브랜치는 `feat/2-crowd-source-validation`, HEAD는 `d71a74beaa3026c8018074f14476df2e9d722513`이다.
시작 시 작업 트리는 깨끗했다. HEAD가 PR #18의 head와 같음을 GitHub와 로컬에서 확인했다.
PR #18은 열려 있다. PR 초안의 제안 Base는 `feat/2-crowd-keypoint-adapter`다.
이번 변경은 Issue #2 전용 파서·진단·다운로드 도구·테스트·보고서에 한정한다.

Issue #2는 `xy (T,53,2)`, `mask (T,53)`, 15fps와
`read_fingerspelling(xy, mask) -> 한글 문자열`을 정의한다.
Issue #4는 `update(t, xy, mask)`의 구간 반환을 정의하며 Issue #6은 실시간 통합을 담당한다.
해당 GitHub 본문과 PR #18의 입력 계약을 직접 확인했다.
[Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2),
[Issue #4](https://github.com/yj2trigger/sign-language-recognition/issues/4),
[Issue #6](https://github.com/yj2trigger/sign-language-recognition/issues/6),
[PR #18](https://github.com/yj2trigger/sign-language-recognition/pull/18)

## 공식 자료 확보 결과

AI-Hub 데이터셋 103 공개 페이지에서 데이터 설명서와 구축활용가이드 링크를 확인했다.
각 링크의 `dataSetSn=103`, `fileSn` 및 문서 종류별 식별자를 대조했다.
페이지의 링크 이름은 각각 데이터 설명서와 구축활용가이드다.
실제 PDF 내부 제목과 파일명은 확보 실패로 확인할 수 없었다.
[AI-Hub 수어 영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)

| 구분 | 설명서 | 구축활용가이드 |
| --- | --- | --- |
| 파일 식별자 | 7969 | 18941 |
| 요청 URL | `https://aihub.or.kr/file/down.do?fileSn=7969&dataMnlFileSn=7969&dataSetSn=103` | `https://aihub.or.kr/file/down.do?fileSn=18941&cnstcPrcuseFileSn=18941&dataSetSn=103` |
| HTTP 상태 | 200 | 200 |
| Content-Type | text/html;charset=UTF-8 | text/html;charset=UTF-8 |
| Content-Length 헤더 | 없음 | 없음 |
| 실제 응답 크기 | 79 bytes | 79 bytes |
| 실제 내용 | 접근 오류 script | 접근 오류 script |
| SHA-256 | f02c3009109228e39726b59d6c1f21806dd3ffbd2faa97ea02df8b51cbecd469 | 동일 |
| 설명서 다운로드 성공 | 실패 | 실패 |

응답은 `<script>`로 시작하고 접근 오류를 알린 뒤 창을 닫는다.
서버가 반환한 원본 파일명은 없다. HTML 조각으로 판정하며 다음 외부 폴더에 보존한다.

```text
D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\docs
```

초기 형식 미확정 수신 파일 `aihub-103-7969`, `aihub-103-18941`도 보존한다.
형식 확인 후 `.html` 사본과 `download-provenance.json`에 출처·형식·크기·해시를 기록한다.
PDF 또는 ZIP 설명서 확보 수는 0개다. 공식 설명서 내부의 키포인트 명세는 미검증이다.
서버 응답만으로 로그인 필요 여부나 접근 오류의 구체적 원인을 확정할 수 없다.

수동 확보 시 AI-Hub 데이터셋 103 페이지에서 정상적인 사용자 세션으로
데이터 설명서와 구축활용가이드 버튼을 사용한다. 필요할 경우 사이트가 요구하는 인증을 완료한다.
받은 파일을 위 `docs` 폴더에 저장하고 실제 PDF/ZIP 형식과 열림을 확인해야 한다.
자동 도구는 인증정보를 사용하지 않는다.

공개 페이지는 x/y/confidence 필드, Pose25·Face70·양손21과 30fps 분할 이미지를 설명한다.
개별 CROWD 파일의 원본 영상 FPS·해상도·반전·추출 시각 규칙과 confidence 허용 범위는
이 페이지에서 확인하지 못했다. OpenPose 출력 문서는 confidence [0,1]과 가변 좌표 scale을
설명한다. 이 규칙이 AI-Hub의 수정된 내보내기에 그대로 적용된다는 근거는 미확보다.
[AI-Hub 데이터 구조](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103),
[OpenPose 출력 명세](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md)

## 실제 ZIP 검사

키포인트 ZIP과 형태소 ZIP은 사용자 지정 경로에서 발견했다. 원본을 읽기 전용으로 열고
압축 해제 없이 ZIP member를 순차적으로 읽었다. 로컬 데이터 루트의 파일 목록에는 영상과
별도 영상 메타데이터 파일이 없었다. 새로 받은 접근 오류 응답은 위 문서 폴더에 있다.

| 항목 | 이번 재검증 |
| --- | ---: |
| 키포인트 ZIP 크기 | 1,411,121,080 bytes |
| 프레임 JSON | 671,745 |
| JSON 문법 정상 / 오류 | 671,744 / 1 |
| 엄격 파서 통과 | 529,700 |
| confidence 범위 위반 프레임 | 142,044 |
| 영상 그룹 | 2,000 |
| 모든 프레임이 엄격 파서를 통과한 그룹 | 446 |
| confidence 오류가 있는 그룹 | 1,554 |
| 형태소 JSON / 매핑 성공 / 오류 | 2,000 / 2,000 / 0 |
| 키포인트와 exact-video 라벨 대응 | 2,000 |
| 양방향 라벨 대응 누락 | 각각 0 |
| 동일 FS 내 원문 라벨 충돌 | 3그룹 |
| 프레임 번호 시작 | 모든 그룹 0 |
| 프레임 번호 누락 / 중복 | 0 / 0 |
| member 읽기·CRC 오류 | 0 |
| 원본 ZIP 검사 전후 SHA-256 | 두 ZIP 모두 동일 |

기존 조사에 제시된 위 수치와 차이는 없었다. 446개는 PR #18의 구조·수치 기준을
충족한 그룹 수다. 실제 좌표·관절·시간 의미가 검증된 시퀀스 수는 0개다.
NaN·무한대 오류는 문법 정상 프레임에서 발견되지 않았다. 문법 오류 1개는 내용 검증 대상에서 제외된다.

키포인트 SHA-256은 `91803b0969a00b45f418ec407c6594d14aef3daec816afd7998ecaff66fe36b1`이다.
형태소 SHA-256은 `1d0874d0f63acd7c42a9c1fe762c4f951b92ff526ca02e6596e50324ef84fbe3`이다.

라벨 검증은 PR #12의 `crowd_labels.py`와 `label_preparation.py`를 commit
`2f10cda1cd5de56d3f74609cdf39ce9e8ec47e7a`에서 읽어 별도 로컬 검사에 재사용했다.
`read_crowd_zip`의 exact-video 연결·오류·라벨 충돌을 그대로 유지했다.
원본 라벨·식별자·문자열은 보고서와 Git 추적 파일에 포함하지 않는다.
[PR #12](https://github.com/yj2trigger/sign-language-recognition/pull/12)

## Confidence 분석과 정책

| 배열 | 최소 | 최대 | 음수 값 | 1 초과 값 | 위반 프레임 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pose | 0 | 1.01479 | 0 | 534 | 534 |
| face | -0.000175198 | 1.11719 | 37,793 | 143,507 | 120,410 |
| left hand | 0 | 1.17766 | 0 | 16,380 | 15,312 |
| right hand | -0.0000185549 | 1.20531 | 2 | 12,430 | 11,545 |
| 값 개수 합계 | -0.000175198 | 1.20531 | 37,795 | 172,851 | 중복 있음 |

위반 값 210,646개가 142,044프레임에 존재한다. 관절 그룹별 위반 프레임 수는 중복되므로
합산하지 않는다. Face는 전체 위반 값의 대부분을 차지한다.
빈 관절 배열은 정상 문법 프레임에서 발견되지 않았다. confidence=0과 0.5 미만 개수도
진단 JSON에 그룹별로 저장하지만 최종53점 mask 결측률은 변환 차단으로 미측정이다.

관절 인덱스별 위반 빈도는 진단 JSON의 `invalid_by_joint`에 기록한다.
얼굴에서는 43번 31,015개, 42번 28,918개, 40번 17,852개가 발생한다.
Pose에서는 18번 303개, 15번 227개, 17번 4개다.
왼손에서는 10번 6,583개, 6번 4,314개, 8번 2,362개가 발생한다.
오른손에서는 14번 3,168개, 6번 1,806개, 10번 1,687개가 발생한다.

영상당 confidence 위반 프레임 수는 최대 479개이며 위반 수 상위 10개 그룹의
기여율은 약 3.1751%다. 오류는 여러 영상에 퍼져 있다. 각 영상의 위반 수 히스토그램과
영상 내 상대 프레임 위치 10구간별 분모·위반 수를 진단 JSON에 기록한다.
이는 수행자별 편향을 입증하지 않는다. 수행자 식별자 정의가 미확정이다.

상대 프레임 위치 10구간의 위반율은 순서대로
21.2178%, 21.2688%, 21.2731%, 21.7255%, 21.6877%, 21.5307%,
20.4468%, 21.0168%, 21.1411%, 20.1357%다. 구간 전체에 오류가 분포한다.

문법 정상 JSON은 모두 version 1.3과 단일 `people` 객체를 사용한다.
배열 길이는 75/210/63/63으로 x,y,c 세 항목 구조와 일치한다.
같은 구조 검증을 재사용해 float64로 읽었고 정수·실수 외 자료형과 비유한 값은 거부한다.
confidence 이상은 원본의 세 번째 값에서 측정했다. float32 변환이나 정규화 전에 발생한다.
현재 파서의 열 선택·형 변환이 범위 위반을 생성했다는 증거는 발견되지 않았다.

생산 과정에서 값이 범위를 벗어난 원인은 미확정이다. 보간·후처리·다른 score 정의 등은
생산자 문서로 확인하지 못했다. OpenPose 문서와의 차이만으로 원본을 자동 보정하지 않는다.
기존 엄격 파서의 [0,1] 기준을 유지하고 진단 전용 경로에서만 원값을 집계한다.
진단 결과는 학습 입력으로 반환되지 않는다.

정책 후보로 관절 단위 mask와 영상 단위 제외를 검토했으나 실제 변환에는 적용하지 않았다.
영상 단위 제외는 77.7%의 그룹을 잃어 편향·길이·문자 분포 변화 가능성이 있다.
관절 mask 정책은 관측 정보와 CTC 정렬 가능성을 줄일 수 있다. 특히 얼굴의 오류가
목표51점에 영향을 주는지와 목표 관절의 이상 값을 구분해 추가 평가해야 한다.
어떤 정책도 미검증 좌표계·시간축 문제를 해결하지 못한다.

## 좌표·관절·시간축 판정

| 항목 | 확인 수준과 결과 |
| --- | --- |
| x/y/c | AI-Hub 공개 설명과 실제 3열 배열 구조 확인 |
| 2D/3D | 모든 문법 정상 프레임에서 2D 형상 일치, 3D 배열 비어 있음 |
| 원시 좌표 범위 | 전체 그룹 최소 -151.61, 최대 2225.99 |
| 픽셀 / 정규화 | 미확정 |
| 원점, 축 방향, padding, crop, 회전 | 해당 내보내기 근거 미확보 |
| 원본 너비·높이, 가로·세로 | 실제 영상·메타데이터 미확보 |
| 정면 여부 | 파일명 F만으로 확정하지 않음 |
| anatomical left/right, 좌우 반전 | 해당 내보내기 근거 미확보 |
| 원본 영상 FPS | 미확정 |
| 키포인트 FPS | 공개 페이지는 30fps 분할 설명, 개별 파일 시간 대응 미확정 |
| 프레임 번호 | 0 시작 연속 인덱스 관찰, 실제 시간과의 연결 미확정 |
| metadata 필드 | version/people 및 일부 camparam, FPS·시각·영상 크기 필드 없음 |
| person_id | 한 종류의 음수 sentinel, 공식 수행자 ID로 사용할 수 없음 |

`camparam`은 585,816프레임에 존재하지만 중첩된 값까지 확인했을 때 비어 있지 않은
leaf를 가진 프레임은 0개다. 컨테이너의 존재와 실제 calibration 값의 존재를 구분했다.

Issue #2 본문에 있는 1080×1920·60fps는 이번 ZIP별 영상 규격 근거로 채택하지 않았다.
형태소 duration과 프레임 수의 비율을 FPS로 사용하지 않았다.
원본 좌표를 정규화하거나 15fps로 샘플링할 조건이 확보되지 않았다.
따라서 재샘플링 오차·초 단위 길이·15fps 성공 여부는 검증할 수 없다.

기존 MediaPipe 상체 인덱스는 `[0,2,5,7,8,11,12,13,14,15,16]`이다.
PR #18의 common51 순서와 상체 대응은 다음과 같다.

| 의미 | BODY_25 | 기존53 슬롯 |
| --- | --- | --- |
| 코 | 0 | 0 |
| 왼귀·오른귀 | 18·17 | 3·4 |
| 왼어깨·오른어깨 | 5·2 | 5·6 |
| 왼팔꿈치·오른팔꿈치 | 6·3 | 7·8 |
| 왼손목·오른손목 | 7·4 | 9·10 |
| 왼손21 | hand_left 0–20 | 11–31 |
| 오른손21 | hand_right 0–20 | 32–52 |

공식 v0.10.21 enum과 설치된 enum에서 pose 2/5는 LEFT_EYE/RIGHT_EYE다.
눈 안쪽 점은 1/4이므로 PR #18의 기존 설명을 정정했다.
OpenPose와 MediaPipe의 눈 이름이 같아도 해당 CROWD 내보내기의 대응 검증이 필요하다.
요청한 common51 정책에서는 두 눈을 제외한다. 기존 파라미터와 오류 코드의 inner 표기는
API 호환을 위해 유지한다.
[MediaPipe PoseLandmark 소스](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/pose.py)

기존 opt-in `to_ctc53(..., allow_missing_inner_eyes=True)`는 슬롯1/2를 0/False로 만든다.
이 정책을 채택할 때 추론도 `adapt_mediapipe53` 다음 `to_ctc53`을 사용해 같은 눈 슬롯을
제외해야 한다. 양쪽의 일치 여부는 합성 회귀 테스트로 확인했다.
실제 CROWD에 대한 관절·좌우 의미 검증과 모델 소유자의 정책 채택은 여전히 필요하다.
[OpenPose 관절 대응](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md),
[MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

## 실제 변환 결과와 학습 판단

| 결과 구분 | 영상 수 |
| --- | ---: |
| 실제 변환 시도 | 0 |
| 실제 변환 성공 | 0 |
| 시도 후 변환 실패 | 0 |
| 명세 부족으로 차단 | 2,000 |
| 별도 품질 정책으로 최종 제외 | 미결정 |
| 검증된 CTC 학습 입력 | 0 |

차단 사유는 source 명세 미검증 2,000건과 프레임 시간 미검증 2,000건이다.
동일 영상에 중복되는 사유이며 총 4,000개 영상으로 합산하지 않는다.
데이터 품질 문제는 위 엄격 파서 통계로 별도 집계한다.

원시 프레임 길이의 최소/25%/중앙/75%/90%/95%/최대는
12 / 188.75 / 261 / 480 / 484 / 658.05 / 974다.
길이 전체 히스토그램은 진단 JSON에 있다. 이는 15fps 변환 후 길이가 아니다.
CTC 최소 시간 길이는 target token 수와 인접 동일 token의 추가 blank 필요량 및
모델의 시간축 감소 규칙을 반영해야 한다. 이번에는 검증된 target·15fps 입력 연결이
차단되어 CTC 최소 길이 검사를 실행하지 않았다.

외부 `crowd/outputs`에 비식별 진단 JSON과 메타데이터 검사 JSON을 저장한다.
학습용 배열 생성 수는 0개다. 실제 GPU 학습을 시작할 수 없다.

저장한 산출물은 다음과 같다.

| 외부 outputs 파일 | 크기 | SHA-256 |
| --- | ---: | --- |
| source-validation.json | 12,519 bytes | 03089c7ecb9710b7f18d63e9e3bc40123967034b7cb1f92ebfc4b3b5554aba80 |
| source-metadata-validation.json | 1,327 bytes | 93a72e6cbfd9652b41a118828b85a381620b3b9b416796d44d48c0d9ce80c149 |

공식 명세 확보 실패와 품질 진단·변환 차단을 보고하는 범위의 PR은 제출 가능하다.
실제 학습 데이터 생성 완료나 Issue #2 완료를 주장할 수 없다.

## 구현과 테스트

`parse_keypoint_json`의 공개 계약을 유지하고 내부 구조 검증을 진단 경로와 공유한다.
기존 어댑터와 설명서에서 pose 2/5의 잘못된 inner-eye 설명을 정정했다.
`ConfidenceDiagnostics`는 범위 밖 값도 원값으로 집계하며 strict frame을 반환하지 않는다.
`audit_source`는 기존 ZIP 인덱스·중복 검사·CRC 읽기를 재사용한다.
`fetch_crowd_docs`는 공식 링크 검증, bounded read, 형식 판정, 해시, 파일별 provenance와
부분 중단 후 누락 문서 재개를 제공한다. PDF header만 확인되면 열림 검증 보류로 기록한다.

Conda `sign-language`의 기존 Python 3.12.16을 사용했다.

```powershell
$taskPython = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $taskPython -m pytest -q
& $taskPython tools/validate_crowd_source.py 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\downloads\수어 영상\2.Validation\[라벨]01_crowd_keypoint.zip'
& $taskPython tools/fetch_crowd_docs.py 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\docs'
```

검사 전 전체 테스트는 102 passed, 62 subtests passed였다.
최종 전체 테스트는 **120 passed, 69 subtests passed**, 경고 2건이며 91.04초가 소요됐다.
경고는 기존 protobuf의 Python 3.14 관련 DeprecationWarning이다.
신규 테스트는 총 18개이며 기존 102개가 함께 통과했다.
신규 테스트는 confidence 이상·자료형·범위 경계·극단 정수·NaN/Inf·관절 누락·중복 프레임·
ZIP 불변성·학습/추론 눈 정책 일치·HTML 접근 실패·파일명 이탈·손상 PDF·다운로드 중단과
재실행을 포함한다. 기존 PR #18 테스트로 좌표 정규화·이미지 크기·좌우·FPS 누락·잘못된
시간·프레임 번호·빈/긴 시퀀스·라벨 누락을 회귀 확인한다.
실제 ZIP 검사는 앞의 전수 통계이며 합성 테스트 성공으로 변환 성공을 대신하지 않는다.

별도 리뷰 에이전트가 손상 PDF 성공 판정, 후속 요청 실패 시 provenance 유실,
부분 checkpoint 재개 누락을 재현했다. 수정하고 각각 회귀 테스트를 추가했다.
최종 재리뷰에서 잔여 필수 수정 결함은 발견되지 않았다.
독립 재실행 결과는 관련 72 passed, 57 subtests passed다. 사람의 리뷰는 미실행이다.
눈 의미 정정 후 별도 재리뷰와 관련 32 passed, 36 subtests passed를 추가 확인했다.

PR #11–#17의 변경 파일 목록과 이번 파일 경로를 비교해 겹침이 없음을 확인했다.
`crowd_keypoints.py`는 선행 PR #18의 파일이므로 해당 브랜치를 Base로 제안한다.
선행 PR 통합 후 #12 라벨, #13 target, #16 전처리, #17 모델의 통합 검증은 후속 작업이다.

## 남은 확인 사항

- 공식 설명서 원본과 문서 내부의 confidence 정의·허용 범위
- 내보내기 좌표 scale·canvas·방향·반전과 BODY_25/Hand21 적용 근거
- 영상별 너비·높이·FPS·timestamp와 JSON frame index 대응
- 원본 영상과 관절 좌표의 시각적 일치
- confidence 이상 생성 원인과 품질 정책의 문자·수행자·길이 분포 영향
- 공식 수행자 분리, Validation 외 학습 자료, label 충돌 3그룹 검토
- 눈 결측 정책 채택, 자모·숫자 target 정책, CTC 입력 검증, GPU 학습과 CER

커밋·push·PR 생성·merge는 수행하지 않았다.

## 실제 변경 행

아래 기존 행은 HEAD `d71a74b` 기준이다. 신규 파일의 기존 행은 없다.

| 파일 | 기존 행 | 변경 후 행 |
| --- | --- | --- |
| src/fingerspelling/crowd_keypoints.py | 52 뒤, 89 | 53–57, 94 |
| src/fingerspelling/crowd_keypoint_adapter.py | 3–4, 235–237 | 3–5, 236–239 |
| docs/fingerspelling_crowd_keypoints.md | 121 | 121–125 |
| src/fingerspelling/crowd_source_validation.py | 신규 | 1–131 |
| tools/fetch_crowd_docs.py | 신규 | 1–157 |
| tools/validate_crowd_source.py | 신규 | 1–26 |
| tests/test_crowd_source_validation.py | 신규 | 1–137 |
| tests/test_fetch_crowd_docs.py | 신규 | 1–119 |
| docs/crowd_source_validation.md | 신규 | 1–303 |
| docs/crowd_source_validation_pr.md | 신규 | 1–175 |
