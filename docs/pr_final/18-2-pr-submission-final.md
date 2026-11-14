# [docs] #2 MediaPipe PR 제출 순서와 최종 본문

Refs #2

Base `docs/2-dependency-audit-final` → Head `docs/2-pr-submission-final`.
기존 비교 기준과 실제 PR 병합 결과를 구분하고 제출 시 사용할 근거·본문을 제공한다.
별도 merge-tree 시뮬레이션과 테스트 결과를 집계했다. 구현·학습 결과는 원래 자료를 보존한다.
수정 파일과 실제 줄 범위는 아래와 같다.

- `docs/issue2_submission_plan.md` 이전 없음 → 이후 1–59행
- `docs/pr_final/09-2-mp-data-expansion.md` 이전 없음 → 이후 1–23행
- `docs/pr_final/10-2-mp-hand-features.md` 이전 없음 → 이후 1–25행
- `docs/pr_final/11-2-mp-video-label-review.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/12-2-mp-expansion-report.md` 이전 없음 → 이후 1–23행
- `docs/pr_final/13-2-mp-review-drafts.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/14-2-mp-keypoint-experiment.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/15-2-mp-keypoint-report.md` 이전 없음 → 이후 1–21행
- `docs/pr_final/16-2-mediapipe-pr-stack.md` 이전 없음 → 이후 1–22행

검증은 기존 PR 통합 전체 회귀390, 최종 전체 회귀513, 각 단계 관련 회귀 및 문서 메타데이터 검사다.
순서는 기존 검토 스택 뒤의 문서 갱신이다. main 제출의 선행 조건과 외부 승인 항목은 보고서에 명시했다.
기존 17개 브랜치와 실제 PR을 수정하지 않았다.

`docs/pr_final/18-2-pr-submission-final.md`은 이전 없음 → 이후 1–25행이다.
실제 변경량은 10파일 / +264 / -0이다.
