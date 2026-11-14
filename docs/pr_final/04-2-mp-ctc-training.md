# [feat] #2 MediaPipe CTC 학습·평가 실행과 보존 감사

Refs #2

Base `feat/2-mp-dataset-preparation` → Head `feat/2-mp-ctc-training`, 검토 Head `4211eae14afd020fc5975e77c4691fe4b16d1b86`.
변경 규모는 3파일 / +258 / -8다. 비교는 실제 Base...Head다.
변경 목적은 `tools/run_mediapipe_ctc.py`에 해당하는 기존 구현을 검토 가능한 단계로 제공하는 것이다.
원래 검증된 함수·테스트를 재사용하고 소스·데이터·checkpoint의 동작을 보존한다.
선행 관계는 현재 Base의 코드다. 검토 순서는 기존 스택의 4번째이며 새 PR 번호는 생성 후 부여한다.

| 변경 파일 | 실제 이전→이후 줄 | 변경 내용 |
|---|---|---|
| `docs/pr_split/04-2-mp-ctc-training.md` | -0,0 +1,19 | 검증·재현·PR 설명 기록 |
| `tools/audit_mediapipe_ctc.py` | -0,0 +1,85 | 해당 기능 구현과 안전 조건 |
| `tools/run_mediapipe_ctc.py` | -37 +36,0<br>-42 +40,0<br>-53 +50,0<br>-102 +98,0<br>-117 +112,0<br>-130 +124,0<br>-142 +135,0<br>-285 +277,0<br>-309,0 +302,154 | 영상 매핑·특징·복구·학습의 해당 단계 |

기존 브랜치 전체 회귀 `441 passed, 2 warnings, 254 subtests passed in 109.22s (0:01:49)`.
이번 실제 병합 시뮬레이션의 단계 회귀 `51 passed, 2 warnings in 9.37s`.
테스트 명령·로그는 .tmp-issue2/dependency-final/mediapipe-04-tests.log 및 기존 git-split 로그에 있다.
기존 PR 12개 구현 모듈은 Base에서 상속한다. 초기화 설명문 한 줄을 제외한 기존 PR 구현 중복 제출은 0개다.
기존 측정은 손 표현 Val 자모 CER .308333 / Test 진단 .269841이다. 이번 정리는 모델 재학습·새 CER 측정을 포함하지 않는다.
수행자 독립·라벨 의미 검증은 미완료다. 현재 Base는 검토용 스택이며 main 제출 전 실제 병합된 main에서 패치 재적용과 diff 재검증이 필요하다.
