# PR 제목

[feat] #2 Stage3 모델의 독립 평가 연결 및 과거 실험 재사용 차단

# GitHub Markdown 본문 초안

Refs #2. Base `feat/2-performance-stage3` (`655d958`) → Head `feat/2-independent-evaluation-stage4`.
PR 생성·기존PR 수정·Merge·Issue종료는 수행하지 않았다.

## Why

Stage3 .191667은 독립 성능이 아니다. 기존 평가 도구에 이 모델을 실제 연결하고
수행자·전사·영상 독립성에 근거한 평가 가능 여부를 확인해야 했다.
실제 결과는 BLOCKED/평가0/CER null이며 차단 원인을 숨기지 않는다.

## What

기존 PR23 실행기를 재구현하지 않고 고정 Git object에서 재사용하는 호환 연결을 추가한다.
Stage3 manifest/selection/result/checkpoint를 묶고 과거 학습·선택·관찰Test 영상 재사용을 선행 차단한다.
오류·변조·과거Val/Test·legacy형식 회귀와 실제 실행/공식자료/보호 결과를 기록한다.
기존 문의 초안은 복제하지 않고 Stage4 보완 문서만 추가한다.

## How

Base는 정확한 Stage3 원격 HEAD655d958에서 만들었다. PR57→59→60 선형 이력을 확인했다.
PR21~23은 별도 이력이라 Merge/Cherry-pick하지 않고 PR23 ece5a90의 소스3개를 그대로 로드한다.
기존 eligibility·근거SHA·중복·PTS·CTC·CER·Reader를 호출하고 승인 정보는 새로 만들지 않는다.
이력 preflight는 현재123/12/8과 과거full-fit1606/AB253·54·53을 합쳐 알려진1,607개를 제외한다.
원본 CSV·모델·학습/전처리/공개API는 수정하지 않는다. 새 파생CSV·provenance·상세결과는 로컬 전용이다.

## Where / 실제 변경 전후 줄 범위

비교 기준은 `655d958df1fc6be203d89f6ce2890aa89ed66030`이다. 모두 신규 파일이다.

| 정확한 파일 경로 | 이전 → 이후 | 역할 |
|---|---|---|
| tools/evaluate_stage4.py | 없음→1–135 | 기존실행기 연결·모델해시 결합·이력 배제 |
| tests/test_stage4_evaluation.py | 없음→1–104 | provenance·누수·형식·손상 회귀 |
| docs/issue2_independent_evaluation_stage4.md | 없음→1–166 | 실제 감사/평가/명령/보호/한계 |
| docs/issue2_provider_inquiry_stage4_addendum.md | 없음→1–36 | 기존 미발송 문의 초안의 추가항목 |
| docs/issue2_stage4_pr_draft.md | 없음→1–83 | PR초안 |

## 실측 결과

| 항목 | 결과 |
|---|---|
| 공식 수행자 매핑 | 미확보, Train123/전체매핑2000개 미확인 |
| 기존 검토 상태 | VERIFIED0/MISMATCH0/UNCERTAIN3/UNREVIEWED516 |
| 완전 독립 전사 | 0 |
| 알려진 실험 영상 합집합 | 1,607 |
| 원본2,000개 중 기록상 미사용 | 393, unseen signer 증명 아님 |
| 검토519개 선행 배제/잔여 | 513/6 |
| 독립 평가 | BLOCKED, 승인0영상/0수행자, 자모·한글CER null |
| D/I/S·분모·완전일치 | 신뢰 가능한 평가 입력이 없어 미측정 |
| 실제 Test 평가·추가 학습 | 없음 |

잔여6개도 매핑·전사·특징계약이 없고 Train123 인물 목록도 미확인이다.
393개는 기존 원본의 후보이며 새 평가자료 확보393개가 아니다. .191667은 기존 Validation 참고값이다.
공식 schema의 표제어/2023라벨수정 및 설명서 접근 한계는 보고서와 문의 보완에 기록했다.
추가 연락·모집·다운로드·519개 검토자료 재생성은 하지 않았다.

## 테스트·독립 리뷰

Stage4 전체214 passed/2 warnings, 기존PR23 전체272 passed/2 warnings.
중복된 공통테스트가 있어486개 고유테스트로 합산하지 않는다. 실제 명령은 보고서에 있다.
Stage3 checkpoint CPU/GPU·PR23 predictor·Reader·공개API 12개 일치, 빈입력PASS.
이는 기존 Validation 입력의 기술 호환성 확인이며 독립 성능 평가가 아니다.
별도 Codex 리뷰가 이전Validation 재사용 위험을 지적해 preflight·회귀를 추가했고 최종 blocker 없음.
보호 대상16,174항목 SHA256 전후일치, 원본/가중치/미추적35개/검토CSV 유지.
기존 local/remote refs 보존을 최종 확인한다. PR11~41·57·59·60을 수정하지 않았다.

## 선행 의존성·미완료

실제 PR Base는Stage3/PR60이며 PR57·59와 그 기반을 따른다. 별도 실행 의존성은 PR21~23의 ece5a90 Git object다.
해당 Git 이력이 없는 shallow checkout에서는 명시적으로 실패하며 자동 fetch/merge하지 않는다.
추가패키지·환경변경 없음. 지정 D드라이브 Conda Python만 사용했다.
공식 익명 수행자/원본구간 관계/독립전사·검수출처/릴리스 대응 근거가 필요하다.
performer 독립CER·라벨 의미 인증·Issue2 완료 조건은 미충족이다. 추가 튜닝보다 평가 근거 확보가 우선이다.

## AI 사용 내역

Codex가 기존구현/공식자료 감사·최소 연결·실제 실행·테스트·보고, 별도 Codex에이전트가 독립 리뷰했다.
제공기관 또는 숙련 수어 검토자의 승인·전사를 대신하지 않는다. 원본·개별정답/예측·모델은 업로드하지 않는다.
