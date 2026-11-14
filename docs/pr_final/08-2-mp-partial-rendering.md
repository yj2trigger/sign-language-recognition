# [feat] #2 토큰을 보존하는 한글 부분 조합과 전달 API

Refs #2

Base `feat/2-mp-model-evaluation` → Head `feat/2-mp-partial-rendering`, 검토 Head `7c31f23cd23e765a846276dcda9b0c6cb6caa748`.
변경 규모는 6파일 / +294 / -18다. 비교는 실제 Base...Head다.
변경 목적은 `src/fingerspelling/partial_compose.py`에 해당하는 기존 구현을 검토 가능한 단계로 제공하는 것이다.
원래 검증된 함수·테스트를 재사용하고 소스·데이터·checkpoint의 동작을 보존한다.
선행 관계는 현재 Base의 코드다. 검토 순서는 기존 스택의 8번째이며 새 PR 번호는 생성 후 부여한다.

| 변경 파일 | 실제 이전→이후 줄 | 변경 내용 |
|---|---|---|
| `docs/fingerspelling_handoff.md` | -0,0 +1,76 | 검증·재현·PR 설명 기록 |
| `docs/pr_split/08-2-mp-partial-rendering.md` | -0,0 +1,22 | 검증·재현·PR 설명 기록 |
| `src/fingerspelling/partial_compose.py` | -0,0 +1,87 | 해당 기능 구현과 안전 조건 |
| `src/fingerspelling_runtime.py` | -46 +46<br>-47,0 +48,4<br>-56 +60,3<br>-80 +86<br>-84 +90<br>-86 +92<br>-94 +100 | 공개 추론·스키마·해당 단계 표현 연결 |
| `tests/test_mediapipe_performance.py` | -22 +21,0<br>-29 +27,0<br>-36 +33,0<br>-42 +38,0<br>-59 +54,0<br>-65 +59,0<br>-74 +67,0<br>-86 +78,0<br>-97 +88,0<br>-102 +92,0<br>-111 +100,0<br>-127 +115,0<br>-135,0 +124,40 | 해당 기능의 정상·경계·오류 회귀 |
| `tools/compare_mediapipe_rendering.py` | -0,0 +1,57 | 해당 기능 구현과 안전 조건 |

기존 브랜치 전체 회귀 `468 passed, 2 warnings, 254 subtests passed in 108.79s (0:01:48)`.
이번 실제 병합 시뮬레이션의 단계 회귀 `78 passed, 2 warnings in 9.76s`.
테스트 명령·로그는 .tmp-issue2/dependency-final/mediapipe-08-tests.log 및 기존 git-split 로그에 있다.
기존 PR 12개 구현 모듈은 Base에서 상속한다. 초기화 설명문 한 줄을 제외한 기존 PR 구현 중복 제출은 0개다.
기존 측정은 손 표현 Val 자모 CER .308333 / Test 진단 .269841이다. 이번 정리는 모델 재학습·새 CER 측정을 포함하지 않는다.
수행자 독립·라벨 의미 검증은 미완료다. 현재 Base는 검토용 스택이며 main 제출 전 실제 병합된 main에서 패치 재적용과 diff 재검증이 필요하다.
