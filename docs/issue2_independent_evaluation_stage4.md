# Issue #2 Stage4 독립 평가 가능성 감사 — 2026-11-15

**실제 평가 실행 결과는 BLOCKED, 승인된 평가0영상/0수행자, 자모·한글 CER null이다.**
기존 최선 .191667을 독립 CER로 재사용하지 않았다. 학습·가중치 변경·새 하이퍼파라미터 선택은 없다.
사람/전사 근거를 만들거나 추측하지 않고 실제 차단 사유와 기술 검증을 완료했다.

## A. Git·PR 감사

시작 로컬 HEAD는95d53b362c4ce117ce6bdb15ae012d710aec0edb였으며 사용자 추적 수정은 없었다.
`git ls-remote --heads origin`으로 실제 전체 원격 refs를 별도 파일에 저장했다.
신규 Stage4는 로컬·원격에 없었다. 아래 로컬·원격 SHA가 일치함을 확인한 후 생성했다.

| 브랜치 | 전체 SHA | 실제 부모 |
|---|---|---|
| feat/2-mediapipe-fingerspelling | 95d53b362c4ce117ce6bdb15ae012d710aec0edb | 651983003edb29389dfaaa6836e6d5ca715f7690 |
| feat/2-performance-20261115 | 62f7b6e3bbb16eeeae3ece1956085e878f5e1d6d | 95d53b362c4ce117ce6bdb15ae012d710aec0edb |
| feat/2-performance-stage2 | 1bd4705d64fc7492a170d43833a732f62d02821e | 62f7b6e3bbb16eeeae3ece1956085e878f5e1d6d |
| feat/2-performance-stage3 | 655d958df1fc6be203d89f6ce2890aa89ed66030 | 1bd4705d64fc7492a170d43833a732f62d02821e |

Stage4 Base는 정확한 원격 Stage3 HEAD `655d958df1fc6be203d89f6ce2890aa89ed66030`이다.
브랜치는 `feat/2-independent-evaluation-stage4`, Worktree는 `.tmp-issue2/independent-evaluation-stage4/worktree`다.
기존 세 브랜치...Stage3의 ahead/behind는 각각0/3,0/2,0/1이고 merge-base는 각 선행 HEAD다.
PR57 base95d53b3/head62f7b6e, PR59 base62f7b6e/head1bd4705, PR60 base1bd4705/head655d958이다.
PR11~41의 Base/Head/상태도 API로 확인해 로컬 `pr11-41-audit.json`에 기록했다.31개 모두 open/미병합이다.
PR21→22→23의 마지막 head는ece5a90dcaea65ecb0dc297f43f55e9b78651a83이다.
Stage3와 PR23의 merge-base는95d53b3이므로 PR23을 새 PR Base로 쓰면 범위가 부정확해진다.
새 PR Base는 Stage3가 맞다. 재사용 코드의 별도 의존성은 PR21~23으로 명시한다.
Merge/Rebase/Reset/Cherry-pick 없이 PR23 Git object를 읽어 재사용했다. 기존 브랜치를 수정하지 않았다.
변경 커밋은 신규 Stage4 커밋1개이며 정확한 Head/부모/원격일치는 최종 Git 검사 결과에 기록한다.
팀원 실시간·웹캠 파일 변경0개, 학습·특징·공개 API 변경0개다.

## B. 기존 구현 재사용과 최소 보완

| 기능 | 기존 구현 | Stage4 처리 |
|---|---|---|
| 53점/424차원·손 상대좌표·BiGRU/CTC·한글 | Stage3 Reader/runtime와 기존 모듈 | 그대로 사용 |
| 수행자·전사 승인 선언 검사 | PR21/22 eligibility | 그대로 호출 |
| 근거SHA·영상/구간/특징 중복·PTS·CTC 길이·CER | PR23 evaluate_independent_cer | 원문 Git object 재사용 |
| Stage3 모델 provenance | manifest/selection/result/checkpoint에 분산 | 해시 결합 호환 레코드만 새 로컬 출력 |
| 기존 Validation/Test와 과거 실험 재사용 배제 | PR23은 Train 중복만 차단 | 얇은 이력 preflight 추가 |

재사용 소스는 process_crowd_blocks/build_independent_review_handoff/evaluate_independent_cer 세 파일이다.
고정 commit ece5a90에서 메모리로 로드하고 실제 소스 SHA를 결과에 기록한다. 파이프라인 복제는 없다.
Stage3 training-config/manifest/selection/result/가중치 SHA·epoch·seed·wd·lr를 확인해 구형 실행기 입력에 연결한다.
이는 기술적 provenance 연결이지 수행자·전사 승인 생성이나 외부 증빙 진위 인증이 아니다.
모델선택에 이미 사용한 자료는 CSV 원본을 바꾸지 않고 새 파생 CSV에서 제외한다.
정답과 예측은 분리되고 기존 PR23 checkpoint_predictor가 Stage3 hand-local 변환도 적용한다.

## C. 공식 자료·메타데이터 확인

기존 기관 문의 초안, 공식 다운로드 응답, ZIP 추가 메타데이터 조사,14개 관측 분석을 먼저 읽었다.
2,000개 라벨이나671,745개 키포인트 JSON을 무의미하게 전부 재파싱하지 않았다.
보유 ZIP 중앙 목록에는 별도 수행자/세션 매핑 파일이 없다는 기존 결과를 보존하고 원본 SHA로 연결했다.
person_id=-1과 파일명/CROWD번호의 수행자 의미 미확인 상태도 그대로다.

[현재 AI-Hub 공식 페이지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)를 실제 재조회했다.
공개 설명은 studio20명/5각도와 CROWD21명 수집을 구분한다. 이는 보유 영상별 사람 대응표가 아니다.
공개 schema에서 attributes.name은 형태소 표제어로 설명된다. 완전한 독립 지문자 전사라는 보장은 없다.
v1.1/2023-12-11 라벨 수정 이력도 있어 보유 ZIP과 공식 릴리스의 버전 대응 근거가 추가로 필요하다.
[설명서7969](https://aihub.or.kr/file/down.do?dataMnlFileSn=7969&dataSetSn=103&fileSn=7969),
[가이드18941](https://aihub.or.kr/file/down.do?cnstcPrcuseFileSn=18941&dataSetSn=103&fileSn=18941)는 이번에도 HTML 응답으로 PDF를 확보하지 못했다.
이전 저장 응답은79바이트 접근 오류 스크립트였다. 이번에 PDF 본문을 읽었다고 주장하지 않는다.
추가로 확인한 [교육데이터 포털](https://data.edmgr.kr/dataView.do?id=aihub_103)도 동일한 두 AI-Hub 파일로 연결됐다.
새 공식 수행자 대응표·독립 전사 파일은 확보하지 못했다. 접근권한 우회나 제공기관 문의 발송은 없다.
real_sen 수행자 번호를 CROWD에 적용하지 않았다. 공개 인원수만으로 개인을 분류하지 않았다.

## D. 전체 실험 이력과 평가 후보 분류

현재 모델은 MediaPipe Train123에서 학습됐다. 과거 별도 OpenPose 모델은 full-fit-v1 Train1,606으로 학습됐다.
두 모델의 가중치 계보를 혼동하지 않는다. Stage3가1,606개를 학습했다는 뜻은 아니다.
다만 새 평가 자료의 실험 노출 이력을 보수적으로 배제하기 위해 과거 실험도 합집합에 넣었다.
기존 A/B는 Train253/Val54/Test53이었다. 전체 적합 실험에서 기존 heldout도 학습에 들어갔다.

| 범주 | 실제 확인 |
|---|---|
| 현재 Train | 123개, 공식 수행자 매핑 누락123 |
| 현재 Validation | 12개, 반복 모델선택에 사용, 새 평가에서 제외 |
| 현재 Test | 8개, 과거 진단 평가 이력 있음, 이번 실제 Test 평가/선택 없음 |
| 과거 full-fit | 1,606개, 현재Train122/123·Val12/12·Test8/8과 영상 겹침 |
| 알려진 전체 실험 영상 합집합 | 1,607개 |
| 기록상 미사용 후보 | 기존2,000개 중393개, 새 촬영/다운로드가 아님 |
| 기존 검토519개 중 이력 배제 | 513개 제외,6개 잔여 |
| unseen signer | 입증0개; 실제 보유 수행자 수·Train과 비중복 여부 미확정 |
| Unverified | 매핑2,000개 모두UNCONFIRMED, 완전 독립 전사0 |

393개는 기록상 영상 단위 후보일 뿐이다. 정답 정책·특징 품질·수행자·전사가 승인된 평가393개가 아니다.
영상 ID와 동일 문자열·동일 수행자는 별개다. 같은 문자열만으로 사람을 합치거나 파일명으로 사람을 추정하지 않았다.
현재/legacy manifest의 .mp4 표기를 기존 canonical ID로 연결했고 aggregate SHA를 개별 특징SHA로 오인하지 않았다.

## E. 실제 검토 상태와 평가 실행 결과

기존519개 검토 페이지/CSV를 재생성하지 않았다. VERIFIED0/MISMATCH0/UNCERTAIN3/UNREVIEWED516이다.
새 완전 전사·전문가 승인0개다. 기존 원본 RGB·PTS 관측과14개 제한 수형 분석은 이미 검토된 근거로 재사용했다.
이번에 새 맹검 자모열·자모 시각·수형 정답을 판정하지 않았다. 손 관측과 부분 crop만으로 전사를 확정하지 않는다.
원본에서 손이 관찰됐다는 기술 근거와 전체 지문자 구간·정답 의미 인증을 구분한다.

Stage3 최선 checkpoint의 실제 SHA는
`bd3a010ff4f2eb95393ce7503939f006b154f41acf1500fc47818236e9f3113a`다.
경로는 `.tmp-issue2/performance-stage3/run-01/decay03/improved.pt`이며 가중치를 바꾸지 않았다.
처음 PR23을 연결한 evaluation-01은519개 전부 자격 미달로 BLOCKED였다.
독립 리뷰 후 이력 검사를 보완한 최종 evaluation-final은513개를 선행 배제하고6개를 기존 gate에 전달했다.
최종6개 모두 다음 네 사유로 차단됐다.

- TRAIN_PERFORMER_MAPPING_INCOMPLETE: Train123 수행자 매핑 전부 미확인
- UNVERIFIED_PERFORMER_MAPPING: 후보의 공식 수행자 근거 없음
- INDEPENDENT_TRANSCRIPT_UNVERIFIED: 완전 독립 전사·전문가 승인 없음
- FEATURE_CONTRACT_UNVERIFIED: 검증된 구간·특징 계약 없음

| 측정 항목 | 실제 결과 |
|---|---|
| 상태 | BLOCKED |
| 독립 평가 영상/수행자 | 승인된0영상/0수행자; 실제 총 수행자 수는 미확정 |
| 자모 CER / 한글 CER | null / null |
| D/I/S·정답 자모 분모·완전 일치·수행자별 분포 | 신뢰 가능한 평가 입력이 없어 미측정 |
| 새로운 평가 자료의 모델 선택/학습 사용 | 없음 |
| Test 사용 | 이번에 읽거나 평가하지 않음; 과거 관찰 이력은 기록 |

이 상태에서는 실제 독립 CER를 강행할 수 없다. .191667은 기존 작은 Validation 참고값으로만 유지한다.
실행 중 legacy의 .mp4 ID/aggregate SHA 형식 차이2건을 발견해 최소 수정·회귀를 추가했다.
실패한 두 호출은 평가 출력 디렉터리 생성 전 실패했고 기존 결과를 덮어쓰지 않았다.

## F. 실제 실행·테스트 명령

아래 인자로 실행했다. 새 재실행은 output과basetemp를 새 이름으로 지정한다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$area="$repo\.tmp-issue2\independent-evaluation-stage4"
$crowd='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$source="$crowd\outputs\mediapipe-ctc-performance\data-expansion-01"
$pack="$repo\.tmp-issue2\evidence-handoff\review-pack-final"
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'
Set-Location "$area\worktree"
& $py tools/evaluate_stage4.py --training-config "$source\training-config.json" --stage3 "$repo\.tmp-issue2\performance-stage3\run-01" --checkpoint "$repo\.tmp-issue2\performance-stage3\run-01\decay03\improved.pt" --mapping-csv "$pack\performer-mapping-template.csv" --review-csv "$pack\blind-transcription-template.csv" --feature-root "$source\features" --evidence-root "$repo\.tmp-issue2\evidence-handoff" --history-config "$crowd\outputs\full-fit-v1\manifest.json" --history-config "$crowd\outputs\ab-hands-body-v1\manifest.json" --output "$area\evaluation-final"
& $py -m pytest -q -p no:cacheprovider --basetemp "$area\pytest-release"
Set-Location "$repo\.tmp-issue2\independent-execution\worktree"
& $py -m pytest -q -p no:cacheprovider --basetemp "$area\pytest-pr23"
& $py "$area\compatibility_local.py"
& $py "$area\inventory_local.py"
& $py "$area\protect_local.py" after
```

최종 Stage4 전체 **214 passed/2 warnings/111.31초**, 기존 PR23 전체 **272 passed/2 warnings/138.84초**다.
중복된 공통 테스트가 있으므로486개 고유 테스트로 합산하지 않는다. 경고는 기존protobuf deprecation이다.
초기 Stage4 208통과 후 리뷰 보완,213통과 후 손상NPZ 회귀를 추가해 최종214통과했다.
PR23의 합성 정상/빈입력/누수/수행자겹침/잘못된구간/특징·근거변조/VFR/CTC길이 등을 재실행했다.
새 회귀는 모델 provenance 변조·과거Val/Test/영상·source·특징 재사용·legacy 형식·손상NPZ를 확인한다.
실제 Stage3 모델의 PR23 predictor/CPU/GPU/Reader/read_fingerspelling을 기존 Val12개에서 대조해 모두 일치했다.
빈 입력 API도 통과했다. 이는 알려진 입력의 호환성 검사이며 새로운 독립 CER가 아니다.
별도 Codex 리뷰가 기존Val 재사용 위험을 지적해 이력 preflight를 보완했고 최종 blocker 없음이다.

## G. 보호·후속 조치·완료 판단

원본ZIP/MP4/NPZ/PT/JSON과 기존검토CSV/HTML/PNG 및 미추적35개 등16,174항목(약25.3GB)의
시작/종료 SHA256가 모두 일치했다. 새 파생CSV·결과·provenance는 Stage4 별도 경로에만 저장했다.
과거 모든temp·모든확장자를 감사했다고 주장하지 않는다. 명시 보호범위의 전후해시다.
기존 local refs와 전체remote heads 스냅샷을 최종 Push 후 비교한다. 기존 브랜치·PR·Commit은 수정하지 않는다.
관련 결과는 `evaluation-final/evaluation.json`, `inventory-final.json`, `compatibility.json`, `protection-after.json`이다.
개인별정보·정답·예측·원본·가중치·검토CSV·상세결과JSON은 GitHub에 업로드하지 않는다.
기존 문의 초안은 복제하지 않고 별도 Stage4 addendum만 작성했다. 문의 발송·모집·추가 다운로드는 없다.
필요 외부자료는 공식 익명 수행자 매핑/원본구간 관계/완전 독립전사/릴리스 대응 근거다.
직접 해결 가능한 연결·호환성·이력배제·회귀는 완료했다. 검토자 배정·전사검수·새 평가 모집절차는 팀 협의가 필요하다.
추가 성능 튜닝보다 유효한 평가 근거 확보가 우선이다. Issue #2의 수행자 독립 성능·정답 검증 조건은 미충족이다.
AI사용: Codex가 조사·최소 연결·실제 실행·검증·문서, 별도 Codex에이전트가 독립 코드/결과 리뷰를 수행했다.
사람 수어 전문가의 승인이나 제공기관 확인을 대신하지 않는다. Commit·일반Push만 하고 PR생성·Merge·Issue종료는 하지 않는다.
