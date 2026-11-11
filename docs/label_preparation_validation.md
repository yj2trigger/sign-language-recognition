# 실행 및 리뷰 기록

## CROWD 확장 후 검증

Windows PowerShell, 기존 Conda `sign-language`, Python 3.12.16에서 실행했다.
이 절은 아래 최초 구현 시의 기록 이후 결과다. 데이터 기반 코드는 집계만 출력했고
실제 문자열·URL·파일별 ID·검토 ID는 로그나 문서에 남기지 않았다.

| 실제 ZIP 집계 | 사전 분석 | 어댑터 재검증 |
|---|---:|---:|
| JSON | 2,000 | 2,000 |
| 정상 영상명·문자열 매핑 | 2,000 | 2,000 |
| 오류 / 파일명 불일치 | 0 / 0 | 0 / 0 |
| FS / CROWD ID 종류 | 1,000 / 2 | 1,000 / 2 |
| 서로 다른 문자열을 가진 FS 그룹 | 3 | 3 |
| 한글 완성형 음절 포함 | 1,950 | 1,950 |
| ASCII 숫자 포함 | 382 | 382 |
| 도로명 접미사 후보 | 1,356 | 1,356 |

정상 레코드는 영상별 문자열을 보존했으며 FS 그룹 1,000개 모두 레코드 2개였다.
기존 `decode_morphemes`로도 2,000개 전체를 다시 읽어 통과했다. 사전 분석과 차이 없음.
실제 실패/예외 파일은 0개였다. ZIP SHA-256은 읽기 검증 전후 동일했다(해시는 비공개).
요청의 `D:/UNIVERS/` 경로에는 파일이 없고 `D:/UNIVERSITY/` 아래 실제 파일을 사용했다.

실행한 명령(기존 Conda Python 실행 파일은 아래 최초 기록 참조):

```powershell
python -B -m unittest discover -s tests -p test_crowd_labels.py -v
python -B -m unittest discover -s tests -p test_label_preparation.py -q
python -B -m pytest tests/test_preprocess.py tests/test_select_classes.py tests/test_extract_full.py -k 'not real_labels' -q
git diff --check
```

실제 명령은 PATH의 python 대신 아래 기록의 절대 경로 실행 파일을 사용했다.
실제 ZIP 검증은 Python stdin 스크립트에서 `read_crowd_zip`, `summary`,
`compare_video_names(report)`를 호출하고, 별도 zipfile 순회에서 기존 파서를 호출했다.
원문·매핑 manifest는 저장하지 않았다. 위 표는 그 실행의 비식별 집계 기록이다.

| 구분 | 결과 |
|---|---|
| 신규 CROWD 합성 테스트 | 22개 통과 (손상 deflate 회귀 포함) |
| 최종 선택 테스트 통합 실행 | 52개 통과, 76 subtests 통과, 1개 제외, 동일 경고 2개 |
| 기존 라벨 파서 합성 회귀 | 13개 통과 |
| 기존 전처리·클래스 선정·추출 합성 회귀 | 17개 통과, 실제 WORD 테스트 1개 제외, protobuf DeprecationWarning 2개 |
| 공백 오류 검사 | `git diff --check` 통과 (Git의 LF→CRLF 안내는 있음) |
| 실제 CROWD 라벨 매핑·기존 파서 호환성 | 2,000개 통과 |
| 실제 영상 존재·내용·파일 목록 대조 | 미실행: 영상/목록 미제공, `not_provided` |
| 전체 pytest·학습·실시간 | 미실행: torch 미설치 상태, 기존 실행 가능 범위만 검사 |
| 전용 린트·포맷·타입·보안 검사 / GitHub Actions | 미실행: 해당 도구/CI 실행하지 않음 |

구현 후 작성자 AI가 별도 검토 단계에서 정규식, 폴더 ID, 메타데이터 대응, Unicode와
공백 보존, ZIP 제한, 예외 메시지, 기존 계약을 다시 읽었다. 손상 deflate에서 `zlib.error`가
파일별 오류 집계 밖으로 빠질 수 있음을 발견해 `member_read`로 처리하도록 수정하고,
압축 스트림을 손상시킨 합성 ZIP 회귀 테스트를 추가했다. 오류 라벨이 제외된 보고서를
정상 영상 목록으로 오인하지 않는 테스트도 추가했다. 외부 독립 리뷰어/사람의 리뷰는
미실행이며 팀 리뷰가 필요하다. 사람의 실행·검증을 AI의 결과와 혼동하지 않는다.

## 최초 형태소 파서 구현 당시 기록

2026-11-11, Windows PowerShell, Conda `sign-language`, Python **3.12.16**.
PATH와 Windows py 런처에서는 Python을 찾지 못했으나 Conda 환경 등록 파일로
기존 환경을 찾아 절대 경로로 실행했다. 패키지 설치·환경 변경 없음.

Python 실행 파일:
`D:/UNIVERSITY/DOWNLOAD/MINI_CONDA/REAL_MINICONDA/envs/sign-language/python.exe`.
아래 명령의 `python`은 이 실행 파일을 뜻한다.

| 검증 | 결과 |
|---|---|
| `python -m unittest discover -s tests -p test_label_preparation.py -v` | 신규 13개 통과 (리뷰 수정 후 실행) |
| `python -m pytest tests/test_preprocess.py tests/test_select_classes.py tests/test_extract_full.py -k 'not real_labels' -q` | 기존 합성 테스트 17개 통과, 실제 라벨 테스트 1개 의도적으로 제외, 의존 라이브러리 DeprecationWarning 2개 |
| `git diff --check` | 통과 |
| 도로명 지문자 실제 라벨 파싱·영상 대응·수행자 검증 | 미실행: 해당 자료/명세 확인 불가 |
| `test_v1_unchanged_on_real_labels` | 미실행: 기존 WORD 실제 자료 대상이라 합성 회귀 실행에서 제외 |
| `tests/test_train_pipeline.py` | 미실행: 현재 환경에 torch 없음 |
| `tests/test_realtime.py` | 미실행: realtime→models가 torch에 의존, 현재 환경에 torch 없음 |
| 전체 pytest | 미실행: 위 의존성 제한. 통과했다고 주장하지 않음 |
| CTC 학습·전체 영상 추출·실시간 통합 | 미실행: 작업 범위 밖 |

별도 작성자 리뷰에서 발견한 구간 객체 검증 우회 및 JSON 초대형 정수 오류를 수정하고
각각 회귀 테스트를 추가했다. 외부 독립 리뷰어 리뷰는 미실행이며 제출 전 팀 리뷰가 필요하다.
테스트는 코드 계약을 검사하며 실제 도로명 데이터 적합성을 입증하지 않는다.

로컬 단어 ZIP 검사는 디렉터리 집계와 JSON 1개의 키 확인만 수행했다.
실제 값은 문서·fixture·로그에 남기지 않았다. 신규 자료 다운로드나 가공 출력 생성 없음.

지정 PR 템플릿은 파일 조회 당시 발견하지 못했으나 이후 사용자가 채팅으로 제공한
0~11번 템플릿을 `PR_body_label_preparation.md`에 적용했다. 해당 수정 시 신규 13개와
기존 선택 합성 테스트 17개를 재실행해 동일한 결과를 확인했다. 사람이 직접 검증한
결과와 AI의 실행 결과를 구분했으며 사람의 독립 검증은 미확인이다.

커밋·Push·PR 생성·Merge는 수행하지 않았다.
