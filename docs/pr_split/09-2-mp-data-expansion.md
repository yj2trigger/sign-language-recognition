# [feat] #2 기존 특징 재사용과 Train 희소 자모 영상 확장

Refs #2

Base `feat/2-mp-partial-rendering` → Head `feat/2-mp-data-expansion`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 9번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 기존 특징 재사용과 Train 희소 자모 영상 확장이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `tools/expand_mediapipe_dataset.py`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `tools/run_mediapipe_ctc.py` | -48 +48,9<br>-116 +124<br>-122,0 +131,42<br>-155,0 +206,5<br>-161,2 +216,2<br>-170 +225<br>-176,0 +232,5<br>-235,0 +296<br>-429,0 +491,5<br>-432,0 +499,4 | 관련 기능 단계 반영 |
| `tools/expand_mediapipe_dataset.py` | -0,0 +1,173 | 검증된 신규 파일 |
| `tests/test_expand_mediapipe_dataset.py` | -0,0 +1,123 | 검증된 신규 파일 |
| `docs/pr_split/09-2-mp-data-expansion.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `482 passed, 2 warnings, 254 subtests passed in 109.25s (0:01:49)`. 로그 `data-expansion-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
기존 관측 성능은 확장 손 표현 Val 자모 CER .308333, 기존 Test 진단 .269841, resync 한글 Val .695652/Test 진단 .64다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
