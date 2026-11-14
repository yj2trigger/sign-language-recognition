# [feat] #2 MediaPipe 53점 공개 추론과 PTS 영상 처리

Refs #2

Base `integration/2-existing-pr-baseline` → Head `feat/2-mp-runtime`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 1번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 MediaPipe 53점 공개 추론과 PTS 영상 처리이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `src/fingerspelling_runtime.py`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `src/fingerspelling_runtime.py` | -0,0 +1,94 | 53점 특징·스키마 검증과 checkpoint 기반 공개 추론 |
| `src/fingerspelling_video.py` | -0,0 +1,58 | 실제 디코딩 PTS 검사와 중복 없는 프레임 선택 |
| `src/fingerspelling/__init__.py` | -1 +1 | 패키지 설명문을 기존 MediaPipe snapshot과 일치 |
| `tests/test_mediapipe_fingerspelling.py` | -0,0 +1,172 | 해당 기능의 정상·경계·예외 회귀 |
| `.gitattributes` | -0,0 +1,5 | 특징 소스 raw SHA를 위한 LF 고정 |
| `docs/pr_split/01-2-mp-runtime.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `420 passed, 2 warnings, 254 subtests passed in 108.62s (0:01:48)`. 로그 `runtime-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
이 단계는 기본 파이프라인 또는 초기 개선 범위다. 최초 CER Train .8095/Val .85/Test 진단 .8730과 이후 결과는 기존 실험 기록을 따른다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
