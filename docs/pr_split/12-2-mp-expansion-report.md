# [feat] #2 143개 확장 실험의 측정 결과와 사용 안내

Refs #2

Base `feat/2-mp-video-label-review` → Head `feat/2-mp-expansion-report`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 12번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 143개 확장 실험의 측정 결과와 사용 안내이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `docs/mediapipe_data_expansion.md`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `docs/mediapipe_data_expansion.md` | -0,0 +1,181 | 검증된 신규 파일 |
| `docs/fingerspelling_handoff.md` | -2,0 +3,8 | 관련 기능 단계 반영 |
| `docs/mediapipe_performance.md` | -2,0 +3,3 | 관련 기능 단계 반영 |
| `docs/pr_split/12-2-mp-expansion-report.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `499 passed, 2 warnings, 254 subtests passed in 110.25s (0:01:50)`. 로그 `expansion-report-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
기존 관측 성능은 확장 손 표현 Val 자모 CER .308333, 기존 Test 진단 .269841, resync 한글 Val .695652/Test 진단 .64다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
