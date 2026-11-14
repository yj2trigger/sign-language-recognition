# [feat] #2 전역 손목 이동을 보존하는 손 상대좌표 비교

Refs #2

Base `feat/2-mp-data-expansion` → Head `feat/2-mp-hand-features`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 10번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 전역 손목 이동을 보존하는 손 상대좌표 비교이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `src/fingerspelling_hand_features.py`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `src/fingerspelling_hand_features.py` | -0,0 +1,38 | 검증된 신규 파일 |
| `src/fingerspelling_runtime.py` | -14,0 +15<br>-63,0 +65,6<br>-66 +73<br>-79,0 +87 | 관련 기능 단계 반영 |
| `tools/experiment_mediapipe_performance.py` | -148,0 +149,2<br>-163 +165<br>-218 +220 | 관련 기능 단계 반영 |
| `tools/train_expanded_mediapipe.py` | -0,0 +1,61 | 검증된 신규 파일 |
| `tests/test_fingerspelling_hand_features.py` | -0,0 +1,84 | 검증된 신규 파일 |
| `docs/pr_split/10-2-mp-hand-features.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `494 passed, 2 warnings, 254 subtests passed in 110.16s (0:01:50)`. 로그 `hand-features-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
기존 관측 성능은 확장 손 표현 Val 자모 CER .308333, 기존 Test 진단 .269841, resync 한글 Val .695652/Test 진단 .64다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
