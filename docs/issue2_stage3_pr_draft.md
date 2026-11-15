# PR 제목

[feat] #2 wd.03으로 Validation CER .1917 재현 및 대응 4seed 검증

# GitHub Markdown 본문 초안

Refs #2. Base `feat/2-performance-stage2` (`1bd4705`) → Head `feat/2-performance-stage3`.
신규 PR생성·기존 PR수정·Merge·Issue종료는 수행하지 않았다.

## Why

기존 시간증강 .225가 여러 seed에서도 개선되는지 확인하고 실제 오류에 근거한 제한 실험이 필요했다.
시간증강만의 4seed 평균은 미증강보다 나빴다. 과적합 격차를 근거로 wd만 바꿔 .191667을 얻었다.
단일seed .20미만 성과와 다중seed 안정성을 구분하고 실패 후보도 기록한다.

## What

기존 학습기에 weight_decay 선택과 실행시간 기록, 기존 시간증강기에 강도 선택을 추가한다.
기존 오류정렬·beam 코드를 원문 그대로 재사용하고 원본 미추적 파일은 보호한다.
phase별 경로/구현/데이터SHA를 고정하는 실행기, 로컬 음향진단, 회귀와 실측 보고서를 추가한다.
원본 데이터·가중치·개별정답/예측·학습결과JSON은 Git에 포함하지 않는다.

## How

동일 원본 Baseline에서 독립 시작, lr.001/20epoch/fresh AdamW/batch8/hidden64×2를 고정했다.
Train123/Validation12/Test8 분할을 유지하고 Test 배열은 로드하지 않는다.
시간증강 seed42/123/2026을 먼저 검증하고 seed20261114에서 최대3후보만 비교했다.
후보는 증강범위[.95,1.05], wd.03, 사전 없는 beam10이다. 정답은 디코더에 전달하지 않는다.
최선 wd.03을 추가3seed와 동일seed 재학습으로 확인했다. 총180epoch로 종료했다.
동률이면 기존 모델을 유지하고 안정성검증 seed는 후보 설정선택 pool에 넣지 않았다.

## Where / 실제 변경 전후 줄 범위

기준은 Stage2 정확한 최종커밋 `1bd4705d64fc7492a170d43833a732f62d02821e`다.
신규파일은 Git상 이전 범위가 없다. 원본 미추적 모듈/테스트의 재사용 출처는 보고서에 기록했다.

| 정확한 파일 경로 | 이전 → 이후 | 역할 |
|---|---|---|
| tools/run_issue2_performance_batch.py | 39→39,49→49,53→53,79뒤→80 | wd인자·설정·시간 |
| tools/run_issue2_stage2.py | 33→33–35,44→46 | 동일 기본값 유지·증강범위인자 |
| tests/test_issue2_stage2.py | 26뒤→27 | 명시 초기화 fixture |
| tools/run_issue2_stage3.py | 없음→1–218 | seed/분석/후보/안정성/재현 |
| tools/inspect_stage3_acoustics.py | 없음→1–38 | 정답 없는 posterior진단 |
| src/ctc_postprocessing.py | 없음→1–184 | 기존 decoder 원문 재사용 |
| src/fingerspelling/ctc_alignment.py | 없음→1–34 | 기존 align_tokens 함수 재사용 |
| tests/test_ctc_postprocessing.py | 없음→1–117 | 기존 acoustic exhaustive oracle·의존성정리 |
| tests/test_issue2_stage3.py | 없음→1–57 | 경로/코드/데이터변조·정렬·CTC 회귀 |
| docs/issue2_performance_stage3.md | 없음→1–179 | 실측·명령·SHA·Git·한계 |
| docs/issue2_stage3_pr_draft.md | 없음→1–93 | PR초안 |

## 실측 결과

| 설정 | seed | Validation CER | D/I/S | epoch |
|---|---:|---:|---|---:|
| 기존 시간증강 | 20261114 | .225000 | 11/4/12 | 17 |
| 동일 | 42 | .266667 | 14/1/17 | 11 |
| 동일 | 123 | .266667 | 12/3/17 | 12 |
| 동일 | 2026 | .266667 | 11/2/19 | 12 |
| 약한 증강 | 20261114 | .275000 | 19/1/13 | 7 |
| wd.03 | 20261114 | **.191667** | **2/6/15** | 13 |
| beam10 | 20261114 | .216667 | 8/5/13 | 17 |
| wd.03 | 42 | .258333 | 11/4/16 | 16 |
| wd.03 | 123 | .266667 | 14/2/16 | 16 |
| wd.03 | 2026 | .233333 | 11/3/14 | 16 |

분모120자모. 미증강 mean/sampleSD .254167/.019837,
기존 시간증강 .256250/.020833, wd.03 .237500/.033679.
wd.03은 대응3seed 개선/1동률, 평균.018750 개선이나 변동성은 증가했다.
최저값은4자모 감소(절대.033333,상대14.815%)이며 다른3seed는.20미만에 도달하지 못했다.
최선 경로·각seed SHA/Train CER/시간·실제명령은 보고서에 있다. 기존 최선도 그대로 보존했다.

## 테스트·독립 리뷰·보호

전체197 passed/2 warnings, 수정후 관련42 passed. 초기 미추적 의존성6실패를 기록하고 수정했다.
decoder exhaustive-path oracle을 유지하고 범위 밖 다른 미추적 실행기의 n-best1테스트는 가져오지 않았다.
선택 가중치 bitwise/개별예측 재현, 모델·beam CPU/GPU 일치, raw Reader12개 일치 확인.
별도 Codex 에이전트가 코드/실측SHA/통계/실제 replay weights를 검토해 최종 blocker 없음.
원본영상/ZIP/NPZ/PT/JSON과 미추적35개 등10,366항목 SHA256 전후일치. 기존 Git refs 보존.
공개 API·웹캠·기존브랜치·PR #11~#57은 수정하지 않았다.

## 선행 PR 의존성·미완료

PR57의 정확한head62f7b6e와 그 자식Stage2 1bd4705에 의존한다. 선형이력·로컬/원격일치를 확인했다.
기존 미병합 CTC/MediaPipe 기반이 있어 main통합 전 선행 구현 반영과 최종 diff/회귀 재검증이 필요하다.
추가 패키지 없음. 지정 D드라이브 sign-language Python3.12.16/torch2.10.0+cu128을 사용했다.
12개 Validation의 반복선택편향, performer/라벨의미·자모별시간 미검증이 남는다.
motion/클립길이/posterior는 실제 수형·속도·정답시각 대리값이며 독립 의미검증이 아니다.
통계적 유의성·모든seed .20미만·실시간 연속입력·수행자독립CER는 미완료다. Issue를 닫지 않는다.

## AI 사용 내역

Codex가 기존코드 재사용·실제GPU실험·검증·보고를 수행했고 별도 Codex에이전트가 독립 리뷰했다.
사람 전문가의 영상전사·수행자검증 및 외부 인간 리뷰는 수행하지 않았다.
