# Issue #2 최종 제출 검증

2026-11-14 KST. Refs #2. 연구용 구현의 검증 보고이며 Issue 완료 선언은 아니다.
새 영상 전처리·재학습·체크포인트 교체·기존 분할 변경·원본 라벨 교정을 하지 않았다.
Issue의 원본 재배포 제한에 따라 실제 라벨, 예측 문자열, 좌표, GIF, 원본 PDF는
이 문서 및 Git에 포함하지 않는다. 실제 입출력은 아래 D드라이브 비공개 결과에 있다.

## A. 완료 조건 판정

| 완료 조건 | 결과 | 실제 증거 | 남은 문제 |
|---|---|---|---|
| 지문자 인식 모델 | 구현·호출 확인 | 기존 checkpoint CPU/CUDA 로드 | 인식 품질·일반화 미검증 |
| 53점 MediaPipe 입력 | 검증 | 실제 3개 `(T,53,2)/(T,53)` 입력 | 다른 시각·정규화 계약은 호환 보장 안 함 |
| 자모 처리 | 검증 | 기존 구현 회귀, 자모/한글 CER 구분 검사 | Unicode 토큰과 실제 수형의 일대일 대응 미확정 |
| 공개 API | 검증 | 실제 3개, 계약 사례 12개, CPU/GPU 문자열 3/3 일치 | 실시간 루프 연결 자체는 수정하지 않음 |
| 수행자 독립 CER | 미완료 | 원본 JSON 2,000개·split·URL·ZIP 조사 | 인증된 수행자 매핑 없음 |
| 영상·라벨 의미 검증 | 부분 수행 | 공식 수형 도판·3영상·18시점·384관측 분석 | 완전한 독립 전사 0개 |
| 전체 회귀 | 통과 | 새 브랜치 191 passed / 2 warnings | 통합 검사 결과는 아래 기록 |
| PR 제출 준비 | 연구용 검증 PR 초안 작성 | 실제 Base/Head·파일·테스트·한계 명시 | 선행 브랜치 미병합; Issue 종료 불가 |

## B. 수행자 독립 평가

우선순위 1로 실제 원본 형태소 ZIP 2,000개와 영상 ZIP의 2,000개 MP4 멤버를 읽었다.
원본 필드는 `metaData.{name,url,duration,exportedOn}`, `data[]`,
`data[].{start,end,attributes}`, `data[].attributes[].name`이며 각각 2,000개 문서에 있다.
signer/performer/person/speaker/subject/participant/session/actor에 해당하는 필드는 없다.
ZIP 디렉터리는 `CROWD/18`·`CROWD/19`에 각각 1,000개다.

공식 소개는 CROWD 수집군의 언어제공자 21명을 설명하지만 개별 영상의 수행자 매핑은
제공하지 않는다. [AI-Hub 103 공식 소개](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103).
로컬 공식 PDF는 찾지 못했다. 공개 링크의 존재 및 응답은 확인했으나 설명서·가이드
본문은 읽지 못했으므로 그 내용을 판정 근거로 사용하지 않았다.
[설명서](https://aihub.or.kr/file/down.do?fileSn=7969&dataMnlFileSn=7969&dataSetSn=103),
[가이드](https://aihub.or.kr/file/down.do?fileSn=18941&cnstcPrcuseFileSn=18941&dataSetSn=103).

필드명 부재에서 조사를 끝내지 않고 URL 저장 경로도 전수 검사했다.
부모 경로는 1,302종이며 날짜·문자 접두·번호 패턴은 1,996개에 매칭되고 4개는 미매칭이다.
문자 접두를 익명화한 후보는 10종이다. 이 값은 업로더·세션일 수도 있으므로 사람이 아니다.

| 기존 분할 | 영상 수 | URL 패턴 매칭 | 미확인 접두 후보 수 | Train과 후보 중복 |
|---|---:|---:|---:|---:|
| Train | 123 | 123 | 10 | 해당 없음 |
| Validation | 12 | 12 | 6 | 6 |
| Test | 8 | 8 | 3 | 3 |

영상 ID 중복은 분할 간 0개지만 수행자 독립을 뜻하지 않는다. `CROWD18/19`도 모든
분할에 등장한다. 가정적으로 URL 후보를 사람으로 취급해도 이 후보 기준 holdout은 없다.
얼굴·외형·파일명으로 신원을 추정하지 않았다.

**수행자 수 미확정, 독립 평가 영상 0개, 독립 자모 CER·한글 CER·완전 일치율 미측정.**
인증된 `video_id → 익명 performer_id` 대응표와 해당 ID의 정의·출처가 필요하다.
그 매핑을 기존 Train123에 적용해 실제로 미학습 사람이 존재할 때만 기존 모델을 평가할 수
있다. 존재하지 않으면 별도의 미학습 수행자 자료가 필요하다. 기존 Test 반복을 독립 평가로
바꾸거나, 금지된 재학습·분할 변경으로 조건을 맞추지 않았다.

기존 후보의 기록은 자모 CER Train .08 / Val .308333 / Test진단 .269841,
resync 한글 CER .235154 / .695652 / .64다. Test진단은 8개이며 수행자 독립 지표가 아니다.
이번에 새 측정한 독립 CER로 표현하지 않는다. CER는 NFC 후 코드포인트 Levenshtein
S+D+I / 정답 길이이며, 전체 집계는 정답 길이 가중 방식이다. 자모는 분해한 호환 자모,
한글은 렌더링 문자열을 비교하며 두 단위를 혼합하지 않는다.

## C. 독립 기준을 사용한 의미 판독 시험

국립국어원 2020-03-02, `(한국수어 교육을 위한) 언어 교육 이론`의 실제 PDF를 받았다.
인쇄 186·187쪽(PDF 188·189쪽)의 표8·표9를 렌더링해 눈으로 확인했다.
자음 19종·모음 21종 수형 도판이며 기존 모델 예측과 독립적인 참고 기준이다.
[공식 교육 PDF](https://www.korean.go.kr/common/download.do?c_file_name=c7f3fcaa-016a-4b72-8803-b597ab5ca52b_0.pdf&file_path=etcData&o_file_name=ksl-education.pdf).
도판의 일부 복합모음에는 순차 수형이 나온다. 정적 수형 하나를 모델 토큰 하나로
가정할 수 없으며, 이 사실만으로 기존 CTC 시퀀스 타깃이 잘못됐다고 단정할 수도 없다.

이전의 3영상에서 접촉표만 보는 한계를 줄이기 위해 기존 28블록의 384개 관측을
시간순으로 연결했다. 새 RGB 디코딩이나 MediaPipe 검출은 하지 않았다.
원래 선택 프레임 인덱스는 연속이고 tracking context 변화는 0이었다.
실제 PTS 간격은 두 영상에서 .016667–.066667초, 나머지 저FPS 영상에서
.033333–.266667초다. 긴 시간 간격을 가상의 관측으로 채우지 않았다.

18개 시점에서 양손 각각 관절 각도, 굽힘·펴짐 관찰, 인접 손끝 거리/손바닥 크기,
투영된 손바닥 signed area를 계산하고 기존 RGB 접촉표 및 공식 도판과 비교했다.
18개는 라벨과 이전 검토를 참고해 고른 확인 시점이며 맹검 독립 전사가 아니다.
공식 기준의 확보와 개별 영상 정답의 독립 확정을 구분한다.

실제 측정 예를 익명 표본으로 기록한다. 오른손 PIP 각도 순서는 검지·중지·약지·새끼다.

| 표본/PTS | 관절에서 확인한 구별 | 한계 |
|---|---|---|
| S1 / 4.466667초 | 약 177/178/177/6도로 세 손가락과 접힌 새끼손가락 패턴 | 도판의 세 손가락 수형과 비교 가능하지만 전체 철자 전사는 아님 |
| S3 / 1.766667초 | 약 178/180/58/74도로 두 손가락 패턴 | 모음 간 수향 차이를 2D 각도만으로 고유 판별 못 함 |
| S3 / 3.700000초 | 약 174/9/15/176도로 검지·새끼 펴짐 패턴 | 손바닥 앞뒤와 깊이를 추가 확인해야 함 |

반례도 확인했다. S1의 1.466667초에는 RGB와 대조가 어려운 가림·투영이 있으며,
마스크 21개가 모두 true여도 여러 관절이 거의 직선으로 나타난다. valid mask는
보정된 관절별 신뢰 확률이 아니다. 손바닥 signed area도 3D 손바닥 앞뒤 판정이 아니다.
대략 .5초 간격 RGB 접촉표는 짧은 동작을 놓칠 수 있고, xy에는 손가락 깊이가 없다.

따라서 수형 차이의 측정은 성공했지만 18개 시점의 고유 자모 확정은 0개다.
후보와 정답을 혼동하지 않고 아래 결과를 유지했다.

| 의미 검증 항목 | 수 |
|---|---:|
| 실제 시험 영상 | 3 |
| 자모 대응 확인 시도 시점 | 18 |
| 독립 공식 수형 기준을 대조한 영상 | 3 |
| 완전한 독립 자모 전사를 확보한 영상 | 0 |
| VERIFIED / SUSPECTED_MISMATCH / UNCERTAIN / TECHNICAL_FAILURE | 0 / 0 / 3 / 0 |
| 라벨 수정 제안 | 0 |

블록 경계를 넘는 별도 관절 GIF도 생성했다. 첫 버전은 GIF 지연 양자화로 시간이
짧아지는 문제가 리뷰에서 확인돼 보존하고 v2를 새로 생성했다. v2는 누적 PTS를 10ms
단위로 양자화하며 저장된 모든 지연을 재읽었다. 384프레임은 그대로이고 각 영상의
최대 누적 표시 오차는 약 3.334ms 미만이다. 마지막 10ms는 표시용이며 새 관측이 아니다.
정확한 원본 PTS는 별도 JSON에 전부 보존했다.

다음 의미 검증에는 고해상도 기존 RGB의 연속 판독 또는 숙련 검토자의 맹검 전사가
필요하다. 지문자의 수향·깊이·복합 수형 순서와 실제 자모 구간을 명시해야 한다.
자료 부족을 반복 추론으로 해결하거나 CTC 예측을 정답으로 쓰지 않았다.

## D. 실제 공개 API 검증

기존 `expanded-hand-local/inference.pt`를 그대로 사용했다.
SHA-256: `0ec9ded244354bf00ede986e66d7619b6b19d6b413996f6129f6b1815329e3f5`.
실행은 기존 `sign-language`, PyTorch `2.10.0+cu128`, RTX 5060 Laptop GPU다.
`read_fingerspelling`은 원래 CPU 기본값이며 같은 모델의 CUDA reader를 별도로 비교했다.

| 실제 표본 | xy / mask 형상 | 출력형 | CPU/CUDA 문자열 일치 | 예외 |
|---|---|---|---|---|
| API-S1 | `(63,53,2)` / `(63,53)` | str | 예 | 없음 |
| API-S2 | `(97,53,2)` / `(97,53)` | str | 예 | 없음 |
| API-S3 | `(45,53,2)` / `(45,53)` | str | 예 | 없음 |

실제 문자열은 비공개 `api-final.json`에 기록했다. strict 렌더러에서 한 표본은 미조합
자모열로 반환됐다. 문자열 타입 검증과 올바른 이름 인식은 별개다. 위 결과는 일반적인
CPU/GPU 수치·정확도 동등성이나 실시간 성능을 보장하지 않는다.

계약 12사례도 실제 호출했다. 정상·450프레임·한쪽 손 결측·일부 프레임 결측은 str,
빈 입력·1프레임·양손 결측은 빈 str, NaN·Inf·잘못된 shape·문자 dtype·비boolean mask는
SequenceError였다. 검사기가 예상 반환·예외 및 CPU/GPU 차이를 명시적으로 실패 처리한다.
유효 좌표는 실수 또는 정수 numeric 배열이며 임의의 dtype 정책을 바꾸지 않았다.
GPU 비교의 입력도 NumPy 배열이다. GPU tensor 자체의 API 지원을 주장하지 않는다.

`src/realtime.py`는 변경하지 않았다. 향후 담당자는 `classify`의 구간 선택 뒤,
기존 단어용 특징을 만들기 전에 시간순 53점 `b[1]`과 boolean `b[2]`를 쌓아
공개 함수를 호출할 수 있다. 얼굴 확장 좌표를 붙이거나 프레임별로 호출하면 안 된다.
호출 경로는 한 번 설정한 checkpoint와 캐시 reader를 사용하며, 실제 모드 선택·UI 연결은
팀원 담당 통합 사항이다.

## E. 테스트 및 독립 리뷰

새 브랜치 전체: **191 passed, 2 warnings**. 신규 검증 테스트는 17개이며 실제 영상
검사·API 실측과 합성 테스트를 구분한다. 경고는 기존 protobuf Python 3.14 관련 경고다.
검사 환경은 원래 Conda만 사용했고 PDF 읽기용 PyMuPDF 1.26.7만 별도 의존성 없이 추가했다.
GIF 도구는 기존 Pillow를 사용한다. PyMuPDF는 도판 열람용이며 공개 API 의존성이 아니다.
PyTorch·CUDA·기존 핵심 패키지·모델을 바꾸지 않았다.

```powershell
$env:TEMP='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2'
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe' `
  -m pytest -q -p no:cacheprovider --basetemp `
  'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\final-validation\regression-final2'
```

기존 PR 테스트까지 포함한 별도 detached 통합 검사는 **554 passed, 2 warnings,
254 subtests passed in 131.51s**였다. 실제 명령은 위와 같은 환경에서 다음과 같다.

```powershell
& 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe' `
  -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" `
  -q -p no:cacheprovider --basetemp `
  'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\final-validation\integrated-tests'
```

이 명령의 작업 디렉터리는 `.tmp-issue2/final-validation/integrated-check`다.
검사 트리는 이전 비파괴 통합 객체 `95fc0eb...`에 새 검증 파일 5개를 추가한 복사본이다.
현재 브랜치의 production Python 48개와 바이트 일치하며 실제 Merge나 동일 ancestry를
주장하지 않는다. 기존 보호 브랜치·PR은 수정하지 않았다.

독립 리뷰는 URL 후보/수행자 구분, API 판정, GIF 시각, 보호 범위를 검사했다.
GIF의 누적 시간 보정과 저장 지연 전수 검증, 학습 재개용 체크포인트 보호 확대를
수정 후 재검토받았으며 추가 필수 결함은 발견되지 않았다.

## F. Git·자료 보호와 제출 의존성

원래 브랜치 `feat/2-mediapipe-fingerspelling`의 보호 SHA `95d53b3...`를 유지한다.
새 작업은 `feat/2-final-submission-validation`, Base는 이미 원격에 있는
`feat/2-crowd-block-validation`의 `80b006b...`다. 새 검증 파일만 별도 Commit·Push하며
실제 결과물은 Git에 넣지 않는다. Commit·Push의 최종 SHA는 제출 실행 결과로 확인한다.

해시 보호 범위는 원본 ZIP 3개, 기존 선정 산출물 149개, 미추적 파일 35개,
기존 블록 결과 파일 11,784개다. 추가로 발견된 checkpoint 346개(학습 재개 모델 포함),
확장 학습 JSON 15개와 원래 training-config SHA도 기록했다.
최종 before/after는 모든 필드에서 완전히 일치했고 초기 보호 목록도 일치했다.
근거는 비공개 `guard-comparison.json`이다.

기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20은 실제 조회에서 모두 open·미병합이다.
#12는 #13의 실제 Git ancestor이지만 #11은 runtime 의존이며 #13의 ancestor는 아니다.
#18은 #19의 실제 ancestor다. #17과 #18은 서로 ancestor가 아니고 불필요한 선행 관계를
새로 만들지 않았다. 기존 16개 PR용 브랜치와 기준용 1개를 원격과 대조해 모두 일치했다.
별도 문서 브랜치 2개도 보존했다.

앞서 구성한 기능 통합용 5개 로컬 브랜치는 여전히 원격에 없고 이 작업에서 수정·게시하지
않았다. 기존 활성 확장 PR 초안은 당시 143개 실험의 기록이다. 신규 검증 PR의 Base/Head는
아래 초안을 사용하며, 기존 초안을 현 통합 상태로 오해하지 않는다.

## G. 재현 및 비공개 결과

결과 루트는 원래 프로젝트의 `.tmp-issue2/final-validation/results`다.
`signers-final.json`, `api-final.json`, `semantics-v2/semantics.json` 및 3 GIF,
보호 manifests, 실제 PR 메타데이터를 보존한다. 원본 영상·라벨·좌표는 재배포하지 않는다.
출력 경로가 이미 존재하면 도구는 덮어쓰기 대신 거부한다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$blocks="$data\outputs\crowd-block15-audit-v1"
# 새 결과 경로를 지정한다. 기존 원본/결과는 수정하지 않는다.
& $py tools/validate_issue2_submission.py signers --data $data --blocks $blocks --output 'D:\new-results\signers.json'
& $py tools/validate_issue2_submission.py api --data $data --blocks $blocks --checkpoint "$data\outputs\mediapipe-ctc-performance\expanded-models-01\expanded-hand-local\inference.pt" --output 'D:\new-results\api.json'
& $py tools/inspect_issue2_semantics.py --blocks $blocks --output 'D:\new-results\semantics'
```

PR 본문은 [최종 검증 PR 초안](issue2_submission_validation_pr.md)을 사용한다.
**Refs #2만 사용하며 이 Issue를 자동 종료하는 키워드는 사용하지 않는다.**
