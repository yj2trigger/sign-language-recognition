# [feat] #2 토큰을 보존하는 한글 부분 조합과 전달 API

Refs #2

Base `feat/2-mp-model-evaluation` → Head `feat/2-mp-partial-rendering`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 8번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 토큰을 보존하는 한글 부분 조합과 전달 API이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `src/fingerspelling/partial_compose.py`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `src/fingerspelling_runtime.py` | -46 +46<br>-47,0 +48,4<br>-56 +60,3<br>-80 +86<br>-84 +90<br>-86 +92<br>-94 +100 | 관련 기능 단계 반영 |
| `tests/test_mediapipe_performance.py` | -22 +21,0<br>-29 +27,0<br>-36 +33,0<br>-42 +38,0<br>-59 +54,0<br>-65 +59,0<br>-74 +67,0<br>-86 +78,0<br>-97 +88,0<br>-102 +92,0<br>-111 +100,0<br>-127 +115,0<br>-135,0 +124,40 | 관련 기능 단계 반영 |
| `docs/fingerspelling_handoff.md` | -0,0 +1,76 | 검증된 신규 파일 |
| `src/fingerspelling/partial_compose.py` | -0,0 +1,87 | 검증된 신규 파일 |
| `tools/compare_mediapipe_rendering.py` | -0,0 +1,57 | 검증된 신규 파일 |
| `docs/pr_split/08-2-mp-partial-rendering.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `468 passed, 2 warnings, 254 subtests passed in 108.79s (0:01:48)`. 로그 `partial-rendering-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
이 단계는 기본 파이프라인 또는 초기 개선 범위다. 최초 CER Train .8095/Val .85/Test 진단 .8730과 이후 결과는 기존 실험 기록을 따른다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
