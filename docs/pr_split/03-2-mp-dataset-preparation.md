# [feat] #2 개별 영상 CRC·PTS 전처리와 중단 복구

Refs #2

Base `feat/2-mp-dataset-contracts` → Head `feat/2-mp-dataset-preparation`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 3번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 개별 영상 CRC·PTS 전처리와 중단 복구이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `tools/run_mediapipe_ctc.py`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `tools/run_mediapipe_ctc.py` | -142,0 +143,143 | 관련 기능 단계 반영 |
| `tests/test_mediapipe_fingerspelling.py` | -33 +32,0<br>-46 +44,0<br>-56 +53,0<br>-62 +58,0<br>-72 +67,0<br>-80 +74,0<br>-90 +83,0<br>-100 +92,0<br>-107 +98,0<br>-113 +103,0<br>-120 +109,0<br>-133 +121,0<br>-143 +130,0<br>-155 +141,0<br>-160 +145,0<br>-168 +152,0<br>-176 +159,0<br>-188 +170,0<br>-193 +174,0<br>-202 +182,0<br>-220 +199,0<br>-233 +211,0<br>-244 +221,0<br>-253 +229,0<br>-258 +233,0<br>-267 +241,0<br>-270,0 +245,66 | 관련 기능 단계 반영 |
| `docs/pr_split/03-2-mp-dataset-preparation.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `441 passed, 2 warnings, 254 subtests passed in 106.32s (0:01:46)`. 로그 `dataset-preparation-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
이 단계는 기본 파이프라인 또는 초기 개선 범위다. 최초 CER Train .8095/Val .85/Test 진단 .8730과 이후 결과는 기존 실험 기록을 따른다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
