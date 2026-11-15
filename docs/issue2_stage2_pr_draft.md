# PR 제목

[feat] #2 다중 seed 검증과 시간 증강으로 Validation 자모 CER .225 재현

# PR 본문 초안

Refs #2. Base `feat/2-performance-20261115` (`62f7b6e`) → Head `feat/2-performance-stage2`.
PR 생성·Merge는 별도 지시 전까지 수행하지 않는다. 기존 PR #11~#57은 수정하지 않았다.

## Why

PR57의 .233333이 단일 seed 최선이어서 다중seed 안정성과 제한된 추가 개선을 실제 확인해야 했다.
추가3seed 평균은 .261111로 악화됐지만 시간 증강에서 27/120=.225000을 얻고 재현했다.
개선은 1자모뿐이며 .20 미만이나 안정적인 일반화 성능을 달성했다고 주장하지 않는다.

## What

기존 학습기에 선택적 dropout/Train증강 인자를 추가하고, 기존 CTC 학습·평가·누수검사를 재사용한다.
seed42/123/2026, 학습후보2개, blank보정1개, 동일설정 재현을 제한된 phase로 실행한다.
실험 결과·검증·보호·한계를 집계 문서에 기록한다. 원본·가중치·개별예측·학습결과JSON은 커밋하지 않는다.

## How

모든 학습은 동일 원본 Baseline에서 lr.001/20epoch/fresh AdamW/batch8로 독립 시작한다.
기존 Train123/Validation12/Test8 분할과 평가를 유지하고 Test 배열은 로드하지 않는다.
dropout .1 또는 Train raw xy/mask의 .9~1.1배 단조 시간 재표본화 중 하나만 바꾼다.
시간 증강 후 모든 정규화/손상대좌표/delta를 다시 계산하며 정답과 CTC 최소길이를 보존한다.
blank 후보는 최선 temporal 모델의 blank logit bias만 −.3으로 바꾼다. 후보 수를 확대하지 않았다.

## Where / 실제 변경 전후 줄 범위

비교 기준은 PR57 head `62f7b6e`이며 다음 외의 기존 파일 변경은 없다.

| 파일 | 이전 → 이후 | 내용 |
|---|---|---|
| tools/run_issue2_performance_batch.py | 39→39, 44→44–47, 52뒤→56, 61→65–66 | 선택적 dropout/증강과 설정 기록 |
| tools/run_issue2_stage2.py | 없음→1–198 | seed·후보·재현 phase |
| tests/test_issue2_stage2.py | 없음→1–42 | 순서·끝점·반복CTC길이·재현·비변조 |
| docs/issue2_performance_stage2.md | 없음→1–131 | 실제 명령·측정·검증·한계 |
| docs/issue2_stage2_pr_draft.md | 없음→1–78 | PR초안 |

## 실험 결과

| 설정 | Validation CER | D/I/S | 선택epoch |
|---|---:|---|---:|
| 기존 seed20261114 | .233333 | 13/2/13 | 14 |
| seed42 | .275000 | 17/2/14 | 13 |
| seed123 | .266667 | 16/2/14 | 18 |
| seed2026 | .241667 | 8/3/18 | 16 |
| dropout .1 | .258333 | 13/2/16 | 16 |
| 시간증강 | .225000 | 11/4/12 | 17 |
| 시간증강+blank −.3 | .225000 | 10/4/13 | 17 |

4seed 평균 .254167, 표본표준편차 .019837(모집단 .017180).
시간증강은 기존보다 절대 .008333/상대3.571% 개선했다. blank 자체의 추가 CER 이득은 없다.
동률규칙 CER→epoch→경로로 blank 파일을 선택했고 보정 없는 temporal도 보존했다.
학습120epoch(추가seed60+후보40+재현20) 후 종료했다.

## 테스트·독립 리뷰

전체 회귀162 passed/2 warnings. 실제 학습·평가·테스트 명령은 보고서에 있다.
CPU/GPU 개별예측 일치, 시간증강/blank 모델 재학습 weights bitwise 일치,
raw Reader12개, Train123개 배율1 전처리 동일성, 독립편집거리 재집계 통과.
별도 Codex 리뷰 에이전트가 누수·CTC·형상·seed·로드·측정·실제결과를 검토해 blocker 없음.
기존 미추적35개와 동결 데이터/기존결과/원본checkpoint172항목 SHA 일치.

## 의존성·한계·미완료

추가패키지 없음. 지정 D드라이브 sign-language Python3.12.16/torch2.10.0+cu128을 사용했다.
PR57과 미병합 Issue2 CTC/MediaPipe 구현에 의존한다. main통합 전 선행구현·재검증이 필요하다.
Validation12개 반복선택편향, 증강후 다중seed 효과, 수행자 독립CER·라벨 의미 검증은 미완료다.
시간증강의 의미보존은 설계 가정이며 전문가 영상전사로 입증되지 않았다. 기존 최선도 보존했다.
향후 phase CLI/manifest 대조와 재사용 학습기SHA 고정 방어 강화가 가능하다. Issue #2를 닫지 않는다.

## AI 사용 내역

Codex가 구현·실제GPU실험·회귀·문서작성, 별도 Codex에이전트가 독립 코드·결과 리뷰를 수행했다.
인간 전문가의 라벨·영상 검증 및 외부 인간 리뷰는 이번 작업에 포함되지 않았다.
