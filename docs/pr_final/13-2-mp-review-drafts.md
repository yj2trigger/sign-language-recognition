# [feat] #2 MediaPipe 구현·확장 PR 검토 기록

Refs #2

Base `feat/2-mp-expansion-report` → Head `feat/2-mp-review-drafts`, 검토 Head `4308ab32ef67414601809fcbf98f6dc47ce09c98`.
변경 규모는 3파일 / +303 / -0다. 비교는 실제 Base...Head다.
변경 목적은 `docs/issue2_expansion_pr_draft.md`에 해당하는 기존 구현을 검토 가능한 단계로 제공하는 것이다.
원래 검증된 함수·테스트를 재사용하고 소스·데이터·checkpoint의 동작을 보존한다.
선행 관계는 현재 Base의 코드다. 검토 순서는 기존 스택의 13번째이며 새 PR 번호는 생성 후 부여한다.

| 변경 파일 | 실제 이전→이후 줄 | 변경 내용 |
|---|---|---|
| `docs/issue2_expansion_pr_draft.md` | -0,0 +1,101 | 검증·재현·PR 설명 기록 |
| `docs/issue2_mediapipe_pr_draft.md` | -0,0 +1,183 | 검증·재현·PR 설명 기록 |
| `docs/pr_split/13-2-mp-review-drafts.md` | -0,0 +1,19 | 검증·재현·PR 설명 기록 |

기존 브랜치 전체 회귀 `499 passed, 2 warnings, 254 subtests passed in 109.88s (0:01:49)`.
이번 실제 병합 시뮬레이션의 단계 회귀 `109 passed, 2 warnings in 10.58s`.
테스트 명령·로그는 .tmp-issue2/dependency-final/mediapipe-13-tests.log 및 기존 git-split 로그에 있다.
기존 PR 12개 구현 모듈은 Base에서 상속한다. 초기화 설명문 한 줄을 제외한 기존 PR 구현 중복 제출은 0개다.
기존 측정은 손 표현 Val 자모 CER .308333 / Test 진단 .269841이다. 이번 정리는 모델 재학습·새 CER 측정을 포함하지 않는다.
수행자 독립·라벨 의미 검증은 미완료다. 현재 Base는 검토용 스택이며 main 제출 전 실제 병합된 main에서 패치 재적용과 diff 재검증이 필요하다.
