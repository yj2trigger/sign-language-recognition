# [docs] #2 기존 PR 의존성과 병합 검증

Refs #2

Base `docs/2-mediapipe-pr-stack` → Head `docs/2-dependency-audit-final`.
기존 비교 기준과 실제 PR 병합 결과를 구분하고 제출 시 사용할 근거·본문을 제공한다.
별도 merge-tree 시뮬레이션과 테스트 결과를 집계했다. 구현·학습 결과는 원래 자료를 보존한다.
수정 파일과 실제 줄 범위는 아래와 같다.

- `docs/issue2_dependency_audit.md` 이전 없음 → 이후 1–133행
- `docs/pr_final/01-2-mp-runtime.md` 이전 없음 → 이후 1–25행
- `docs/pr_final/02-2-mp-dataset-contracts.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/03-2-mp-dataset-preparation.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/04-2-mp-ctc-training.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/05-2-mp-pipeline-guide.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/06-2-mp-ctc-conditioning.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/07-2-mp-model-evaluation.md` 이전 없음 → 이후 1–22행
- `docs/pr_final/08-2-mp-partial-rendering.md` 이전 없음 → 이후 1–25행

검증은 기존 PR 통합 전체 회귀390, 최종 전체 회귀513, 각 단계 관련 회귀 및 문서 메타데이터 검사다.
순서는 기존 검토 스택 뒤의 문서 갱신이다. main 제출의 선행 조건과 외부 승인 항목은 보고서에 명시했다.
기존 17개 브랜치와 실제 PR을 수정하지 않았다.

`docs/pr_final/17-2-dependency-audit-final.md`은 이전 없음 → 이후 1–25행이다.
실제 변경량은 10파일 / +340 / -0이다.
