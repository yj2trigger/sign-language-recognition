# [feat] #2 전역 손목 이동을 보존하는 손 상대좌표 비교

Refs #2

Base `feat/2-mp-data-expansion` → Head `feat/2-mp-hand-features`, 검토 Head `cc17262ed1583cb0aa1daffef2279a4f0bb5672a`.
변경 규모는 6파일 / +218 / -3다. 비교는 실제 Base...Head다.
변경 목적은 `src/fingerspelling_hand_features.py`에 해당하는 기존 구현을 검토 가능한 단계로 제공하는 것이다.
원래 검증된 함수·테스트를 재사용하고 소스·데이터·checkpoint의 동작을 보존한다.
선행 관계는 현재 Base의 코드다. 검토 순서는 기존 스택의 10번째이며 새 PR 번호는 생성 후 부여한다.

| 변경 파일 | 실제 이전→이후 줄 | 변경 내용 |
|---|---|---|
| `docs/pr_split/10-2-mp-hand-features.md` | -0,0 +1,22 | 검증·재현·PR 설명 기록 |
| `src/fingerspelling_hand_features.py` | -0,0 +1,38 | 해당 기능 구현과 안전 조건 |
| `src/fingerspelling_runtime.py` | -14,0 +15<br>-63,0 +65,6<br>-66 +73<br>-79,0 +87 | 공개 추론·스키마·해당 단계 표현 연결 |
| `tests/test_fingerspelling_hand_features.py` | -0,0 +1,84 | 해당 기능의 정상·경계·오류 회귀 |
| `tools/experiment_mediapipe_performance.py` | -148,0 +149,2<br>-163 +165<br>-218 +220 | Train 표준화·export·Validation 선택 |
| `tools/train_expanded_mediapipe.py` | -0,0 +1,61 | 해당 기능 구현과 안전 조건 |

기존 브랜치 전체 회귀 `494 passed, 2 warnings, 254 subtests passed in 110.16s (0:01:50)`.
이번 실제 병합 시뮬레이션의 단계 회귀 `104 passed, 2 warnings in 10.32s`.
테스트 명령·로그는 .tmp-issue2/dependency-final/mediapipe-10-tests.log 및 기존 git-split 로그에 있다.
기존 PR 12개 구현 모듈은 Base에서 상속한다. 초기화 설명문 한 줄을 제외한 기존 PR 구현 중복 제출은 0개다.
기존 측정은 손 표현 Val 자모 CER .308333 / Test 진단 .269841이다. 이번 정리는 모델 재학습·새 CER 측정을 포함하지 않는다.
수행자 독립·라벨 의미 검증은 미완료다. 현재 Base는 검토용 스택이며 main 제출 전 실제 병합된 main에서 패치 재적용과 diff 재검증이 필요하다.
