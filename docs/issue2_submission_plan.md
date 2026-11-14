# Issue #2 최종 PR 제출 경로

Refs #2

기존 PR 9개는 새 조회에서 모두 open이다. 현재 실제 main에는 아직 통합 결과가 없다.
기존 16개 구현 브랜치의 Base/Head는 검토용 스택으로 유효하고, 전체 3-way 연결과 테스트를 확인했다.
기존 17개 Push 브랜치는 그대로 사용한다. 코드가 같은 새 16개 브랜치를 지금 중복 생성하지 않았다.

## 실제 제출 선행 조건

1. 기존 PR의 독립 그룹 #11/#12/#15/#16/#17/#18/#20은 각각 검토할 수 있다.
2. #13은 #12 ancestry와 기본 인코딩의 #11을 확인한다. #19는 #18 뒤에 검토한다.
3. #11과 #12의 초기화 설명문 충돌 해결안을 리뷰한다. 기존 PR의 400줄 초과는 별도 기록했고 원본 PR을 수정하지 않았다.
   #13/#19의 기존 Base는 부모 기능 브랜치다. 부모의 main 병합 후 소유자 승인으로 Base를 조정하거나 실제 main에서 새 제출 경로를 만든다. Squash 시 ancestry를 별도 확인한다.
4. MediaPipe의 코드상 기반은 #11/#13/#15/#16/#17 및 #13의 Base #12다. #18/#19/#20을 이 기능의 필수 import 선행 조건으로 추가하지 않는다.
5. 담당자의 실제 PR 승인·병합 뒤 원격 main을 다시 조회한다. 현재의 비교 기준 브랜치는 main에 제출하지 않는다.
6. 실제 병합된 main에서 새 제출용 첫 브랜치를 만든다. 기존 첫 Base...Head 패치만 안전하게 재적용하고 테스트한다.
7. 이후 기존 각 Base...Head 패치를 앞의 새 제출용 브랜치에 차례로 적용한다. 기존 브랜치·Commit은 보존한다.
8. 각 새 실제 Base와 diff를 다시 계산하고 10파일/+400줄을 검사한다. 기존 PR 소스가 새 diff에 포함되면 제출을 보류하고 원인을 확인한다.
9. 기존 스택 순서로 검토·PR 생성한다. 독립한 기존 PR 순서를 불필요하게 고정하지 않는다.

위 5단계의 실제 PR Merge는 이번 허용 범위 밖이며 외부 승인 단계로 남아 있다.
Base 이름만 main으로 바꾸면 공통 조상이 과거 main에 남을 수 있다. 이번 시뮬레이션에서 tree 일치와 Commit 그래프 차이를 함께 확인했다.
현재 main을 대상으로 기존 비교 기준의 +7852줄을 하나의 PR로 제출하지 않는다.

## 16개 실제 검토 diff와 단계 검증

| PR 번호 | Base | Head | 파일 | + | - | 이번 단계 테스트 | 최종 본문 |
|---|---|---|---:|---:|---:|---|---|
| 미생성 | `integration/2-existing-pr-baseline` | `feat/2-mp-runtime` | 6 | 352 | 1 | 30 통과 | `docs/pr_final/01-2-mp-runtime.md` |
| 미생성 | `feat/2-mp-runtime` | `feat/2-mp-dataset-contracts` | 3 | 283 | 0 | 47 통과 | `docs/pr_final/02-2-mp-dataset-contracts.md` |
| 미생성 | `feat/2-mp-dataset-contracts` | `feat/2-mp-dataset-preparation` | 3 | 228 | 26 | 51 통과 | `docs/pr_final/03-2-mp-dataset-preparation.md` |
| 미생성 | `feat/2-mp-dataset-preparation` | `feat/2-mp-ctc-training` | 3 | 258 | 8 | 51 통과 | `docs/pr_final/04-2-mp-ctc-training.md` |
| 미생성 | `feat/2-mp-ctc-training` | `feat/2-mp-pipeline-guide` | 3 | 379 | 0 | 51 통과 | `docs/pr_final/05-2-mp-pipeline-guide.md` |
| 미생성 | `feat/2-mp-pipeline-guide` | `feat/2-mp-ctc-conditioning` | 3 | 299 | 0 | 70 통과 | `docs/pr_final/06-2-mp-ctc-conditioning.md` |
| 미생성 | `feat/2-mp-ctc-conditioning` | `feat/2-mp-model-evaluation` | 3 | 327 | 5 | 70 통과 | `docs/pr_final/07-2-mp-model-evaluation.md` |
| 미생성 | `feat/2-mp-model-evaluation` | `feat/2-mp-partial-rendering` | 6 | 294 | 18 | 78 통과 | `docs/pr_final/08-2-mp-partial-rendering.md` |
| 미생성 | `feat/2-mp-partial-rendering` | `feat/2-mp-data-expansion` | 4 | 391 | 5 | 92 통과 | `docs/pr_final/09-2-mp-data-expansion.md` |
| 미생성 | `feat/2-mp-data-expansion` | `feat/2-mp-hand-features` | 6 | 218 | 3 | 104 통과 | `docs/pr_final/10-2-mp-hand-features.md` |
| 미생성 | `feat/2-mp-hand-features` | `feat/2-mp-video-label-review` | 3 | 221 | 0 | 109 통과 | `docs/pr_final/11-2-mp-video-label-review.md` |
| 미생성 | `feat/2-mp-video-label-review` | `feat/2-mp-expansion-report` | 4 | 212 | 0 | 109 통과 | `docs/pr_final/12-2-mp-expansion-report.md` |
| 미생성 | `feat/2-mp-expansion-report` | `feat/2-mp-review-drafts` | 3 | 303 | 0 | 109 통과 | `docs/pr_final/13-2-mp-review-drafts.md` |
| 미생성 | `feat/2-mp-review-drafts` | `feat/2-mp-keypoint-experiment` | 3 | 346 | 0 | 123 통과 | `docs/pr_final/14-2-mp-keypoint-experiment.md` |
| 미생성 | `feat/2-mp-keypoint-experiment` | `feat/2-mp-keypoint-report` | 2 | 335 | 0 | 123 통과 | `docs/pr_final/15-2-mp-keypoint-report.md` |
| 미생성 | `feat/2-mp-keypoint-report` | `docs/2-mediapipe-pr-stack` | 3 | 149 | 5 | 123 통과 | `docs/pr_final/16-2-mediapipe-pr-stack.md` |

전체 통합 회귀 `513 passed, 2 warnings, 254 subtests passed in 114.71s (0:01:54)`.
각 단계에서는 해당 시점에 존재하는 MediaPipe·키포인트 관련 테스트를 모두 실행했다. 기존 PR 통합 전체 회귀와 최종 전체 회귀는 별도로 실행했다.
합성 모델 최적화는 기존 회귀 테스트의 일부다. 실제 143개 데이터 학습이나 영상 전처리는 실행하지 않았다.
최종 tree는 기존 16개 스택 최종 상태와 동일하며 95d53b3의 40개 변경 파일도 바이트가 일치한다.
실제 MediaPipe checkpoint 로딩·공개 API의 입력 유한성·CTC·자모·손 표현·복구 조건은 관련 회귀와 최종 추론 대조 기록을 따른다.

## 문서 정리 브랜치

`docs/2-dependency-audit-final`는 기존 docs/2-mediapipe-pr-stack 위에 감사 보고서와 앞 8개 최종 PR 본문을 추가한다.
`docs/2-pr-submission-final`는 그 위에 이 제출 계획과 뒤 8개 본문을 추가한다.
두 브랜치는 문서만 변경한다. Python blob 전체가 기존 최종 스택과 같은지 검사하고 실제 diff 제한을 확인한다.
각 브랜치의 자체 PR 초안은 docs/pr_final/에 있다. 이번 작업에서도 PR 생성과 실제 원격 Merge는 수행하지 않았다.
이 문서의 main 제출 절차는 선행 PR이 실제 통합된 뒤 사용할 실행 계획이다. 지금 제출 가능한 main 기반 구현 브랜치라고 표시하지 않는다.
