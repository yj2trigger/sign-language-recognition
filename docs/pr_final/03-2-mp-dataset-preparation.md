# [feat] #2 개별 영상 CRC·PTS 전처리와 중단 복구

Refs #2

Base `feat/2-mp-dataset-contracts` → Head `feat/2-mp-dataset-preparation`, 검토 Head `444b293c5284e4cca0621e2d4ebdfe8375b9602a`.
변경 규모는 3파일 / +228 / -26다. 비교는 실제 Base...Head다.
변경 목적은 `tools/run_mediapipe_ctc.py`에 해당하는 기존 구현을 검토 가능한 단계로 제공하는 것이다.
원래 검증된 함수·테스트를 재사용하고 소스·데이터·checkpoint의 동작을 보존한다.
선행 관계는 현재 Base의 코드다. 검토 순서는 기존 스택의 3번째이며 새 PR 번호는 생성 후 부여한다.

| 변경 파일 | 실제 이전→이후 줄 | 변경 내용 |
|---|---|---|
| `docs/pr_split/03-2-mp-dataset-preparation.md` | -0,0 +1,19 | 검증·재현·PR 설명 기록 |
| `tests/test_mediapipe_fingerspelling.py` | -33 +32,0<br>-46 +44,0<br>-56 +53,0<br>-62 +58,0<br>-72 +67,0<br>-80 +74,0<br>-90 +83,0<br>-100 +92,0<br>-107 +98,0<br>-113 +103,0<br>-120 +109,0<br>-133 +121,0<br>-143 +130,0<br>-155 +141,0<br>-160 +145,0<br>-168 +152,0<br>-176 +159,0<br>-188 +170,0<br>-193 +174,0<br>-202 +182,0<br>-220 +199,0<br>-233 +211,0<br>-244 +221,0<br>-253 +229,0<br>-258 +233,0<br>-267 +241,0<br>-270,0 +245,66 | 해당 기능의 정상·경계·오류 회귀 |
| `tools/run_mediapipe_ctc.py` | -142,0 +143,143 | 영상 매핑·특징·복구·학습의 해당 단계 |

기존 브랜치 전체 회귀 `441 passed, 2 warnings, 254 subtests passed in 106.32s (0:01:46)`.
이번 실제 병합 시뮬레이션의 단계 회귀 `51 passed, 2 warnings in 9.36s`.
테스트 명령·로그는 .tmp-issue2/dependency-final/mediapipe-03-tests.log 및 기존 git-split 로그에 있다.
기존 PR 12개 구현 모듈은 Base에서 상속한다. 초기화 설명문 한 줄을 제외한 기존 PR 구현 중복 제출은 0개다.
기존 측정은 손 표현 Val 자모 CER .308333 / Test 진단 .269841이다. 이번 정리는 모델 재학습·새 CER 측정을 포함하지 않는다.
수행자 독립·라벨 의미 검증은 미완료다. 현재 Base는 검토용 스택이며 main 제출 전 실제 병합된 main에서 패치 재적용과 diff 재검증이 필요하다.
