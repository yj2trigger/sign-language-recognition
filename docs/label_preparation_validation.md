# 실행 및 리뷰 기록

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
