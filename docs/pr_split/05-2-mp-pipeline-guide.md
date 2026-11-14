# [feat] #2 MediaPipe 기본 학습 재현 기록과 모듈 출처

Refs #2

Base `feat/2-mp-ctc-training` → Head `feat/2-mp-pipeline-guide`. 기존 PR #11/#12/#13/#15/#16/#17/#18/#19/#20 기준 코드 위에 쌓는다.
검토·생성·Merge 순서는 기준 PR 승인 후 이 스택의 5번째다. 기준 브랜치는 PR 제출용이 아니다.
구현 목적은 MediaPipe 기본 학습 재현 기록과 모듈 출처이며 검증된 기존 구현을 원래 소스에서 재사용했다.
호출·검토 진입점은 `docs/mediapipe_fingerspelling.md`다. 전처리·학습 결과는 기존 D 자료를 사용한다.

| 파일 | 변경 전후 실제 줄 | 변경 내용 |
|---|---|---|
| `docs/mediapipe_fingerspelling.md` | -0,0 +1,344 | 검증된 신규 파일 |
| `src/fingerspelling/provenance.md` | -0,0 +1,16 | 검증된 신규 파일 |
| `docs/pr_split/05-2-mp-pipeline-guide.md` | 신규 문서 전체 | PR 제목·의존성·실측 검증 기록 |

전체 회귀 `441 passed, 2 warnings, 254 subtests passed in 108.33s (0:01:48)`. 로그 `pipeline-guide-tests.log`.
명령은 지정 Conda Python으로 `import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())`와 `-q -p no:cacheprovider --basetemp <D 임시 경로>`다.
이 단계는 기본 파이프라인 또는 초기 개선 범위다. 최초 CER Train .8095/Val .85/Test 진단 .8730과 이후 결과는 기존 실험 기록을 따른다.
이번 Git 분리는 재학습·재전처리를 포함하지 않는다. 수행자 독립 평가·라벨 전사 검증은 미완료다.
