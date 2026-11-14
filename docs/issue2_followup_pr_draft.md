# PR 제목

[test] #2 validate realtime segment compatibility and saved-result review readiness

Base `feat/2-independent-evidence-handoff`

Head `feat/2-realtime-evaluation-readiness`

Refs #2. Draft 제출을 권장한다.

## What

기존 실시간 버퍼 형식의 지문자 API 어댑터, 기존 519개 관측 영상의 검토 우선순위,
저장된 CTC 결과 분석과 독립 평가 원본 구간 중복 차단을 추가한다.
실제 웹캠 루프 연결과 수행자 독립 CER는 미완료다.

## Why

기존 API와 기술적 특징 추출은 확보됐지만 실시간 입력 연결 지점, 사람 검토 순서,
기존 예측 오류의 재현 가능한 집계가 필요했다. 수행자·독립 전사 근거가 없는 상태에서
참고 CER나 기술적 검증을 Issue #2 완료로 확대하지 않도록 준비 상태를 명시한다.

## How

- `realtime.py`의 timestamp/xy/mask 버퍼를 별도 어댑터가 읽고 공개 API를 호출한다.
- 기존 특징과 접촉표 해시·PTS·selected index를 검사하고 검토 순위를 생성한다.
- PR #21의 비노출 HTML·CSV·매핑 검사와 기존 CER/추론 구현을 재사용한다.
- 저장 결과의 원래 split ID·특징 해시·Test 선택 모델/체크포인트를 확인한다.
- 원본 영상 미확인과 같은 원본의 겹치는 평가 구간을 선언 검사에서 차단한다.

## How I tested

Conda `sign-language`의 지정 Python만 사용했다. 다음 명령은 후속 worktree에서 실행했다.
TEMP/TMP는 D드라이브 저장소 `.tmp-issue2`로 설정했다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp','D:/UNIVERSITY/UAI_project2/FILE2/sign-language-recognition/.tmp-issue2/followup/regression-submission','--junitxml','D:/UNIVERSITY/UAI_project2/FILE2/sign-language-recognition/.tmp-issue2/followup/regression-submission.xml']))"
```

240 passed, 0 failed, 0 skipped, 2 warnings in 110.23s.
기존 protobuf deprecation 경고 2건이며 19개 신규 테스트가 포함된다.
정상 버퍼·빈 구간·shape·mask·NaN/Inf·clock·VFR/저FPS 관측·부분 블록·누락 index·
중복 alias·원본 구간 겹침·누락 source·특징 변조·Test checkpoint 불일치를 검사했다.
실제 체크포인트 API 계약 12종과 CPU/CUDA3개 실제 입력도 재실행했다.
비공개 데이터 실행 명령과 출력 경로는 `docs/issue2_followup_validation.md`에 있다.

## Rejected Alternatives

- 팀원 realtime 경로 직접 수정은 소유권과 통합 검토가 필요하므로 호출 어댑터를 제공했다.
- 새 검토 페이지/CSV/매핑 검사기는 중복을 만들기 때문에 PR #21 구현을 재사용했다.
- 기존 모델의 새 학습은 이번 검증 범위 밖이며 저장 결과 분석으로 진행했다.
- 모델 출력의 정답 채택은 독립 검증 근거가 없어 제외했다.

## Verifiable Evidence

| 검증 | 실제 결과 |
|---|---|
| 전체 회귀 | 240 통과 / 실패0 / Skip0 / 경고2 |
| 실제 버퍼 재생 | 3개 / CPU·CUDA·직접 API 문자열 동일 |
| 기존 API 계약 검사 | 12종 통과 |
| 검토 우선순위 | 519개 / 파일 해시·PTS 검사 실패0 |
| 독립 평가 | 0개 / 독립 CER 미측정 |
| 의미 상태 | VERIFIED0 / MISMATCH0 / UNCERTAIN3 / UNREVIEWED516 |
| 보호 해시 | ZIP3·산출물149·checkpoint346·학습JSON15·미추적35·블록파일11784·config 일치 |

선정 모델의 제공 라벨 기준 자모 CER는 Train0.0800, Validation0.3083,
관찰 Test0.2698이다. 기본 모델은 Train0.0322, Validation0.3250이다.
Validation 분모는 120자모이며 모델간 오류 차이는 2개다. 일반화 우위를 확정하지 않는다.
현재 모델의 실사용 인식 성능은 검증되지 않았다.

## Which Line / File

| 파일 | 변경 전 → 변경 후 실제 범위 | 변경 내용 |
|---|---|---|
| `src/fingerspelling_realtime.py` | 없음 → 1–44 | 버퍼 구간 선택·clock 검사·공개 API 호출 |
| `tools/audit_fingerspelling_followup.py` | 없음 → 1–151 | 기존 파일 검증·순위·저장 예측 오류 집계 |
| `tools/build_independent_review_handoff.py` | 74–75 사이 및 75–76 사이 → 75 및 77–83 추가 | source 미확인·평가 구간 겹침 차단 |
| `tests/test_fingerspelling_followup.py` | 없음 → 1–98 | 정상·경계·오류·무결성 신규 회귀19개 |
| `docs/issue2_followup_validation.md` | 없음 → 1–145 | 실제 결과·해석·호환성·재현·잔여 조건 |
| `docs/issue2_followup_pr_draft.md` | 신규 파일 전체 | 이 제출 초안 |

## Compare Before & After

| 항목 | Before | After |
|---|---|---|
| 실시간 입력 연결 지점 | API 계약 및 개별 입력 검증 | 버퍼 어댑터와 실제 저장 시계열 재생 검증 |
| 검토 자료 | 519개 비노출 페이지 | 같은 페이지를 기술 관측 지표로 정렬 |
| 예측 오류 | 개별 저장 보고서 | 동일 split·특징 검증 후 S/D/I·길이·blank·품질군 집계 |
| 평가 중복 검사 | Train 원본·영상·사람·특징 중복 | 평가 후보끼리의 원본 구간 겹침도 차단 |

## AI Use Disclosure

OpenAI Codex가 기존 코드 조사, 어댑터·검사 도구·테스트·문서 작성을 수행했다.
별도 Codex 리뷰 에이전트가 특징 해시와 Test 귀속 검사를 지적했고 수정 후 재검토했다.
사람 수어 판독자의 전사 검수는 수행하지 않았다. 제출자의 직접 diff 이해와 승인은 별도다.

## Improvements & Trade-offs

엄격한 clock 검사는 누락·VFR를 고정15fps처럼 처리하는 오류를 차단하지만 실제 웹캠
지터에서는 호출자가 명시적 시간축 정책을 결정해야 한다. 3개 입력의 단발 시간은
워밍업 영향을 포함하므로 실시간 처리량 증거로 일반화할 수 없다.
손 관측률 순위는 수형·흐림·라벨 의미를 인증하지 않는다.
품질군 비교에서 결측군은 0개여서 품질과 오류의 관련성은 미확정이다.
원본 영상 ID가 정확하다는 공식 대응 근거와 독립 정답이 추가로 필요하다.

## To-Do

- [ ] 팀원과 실제 webcam segmentation/clock 정책을 합의하고 호출 지점을 연결
- [ ] 우선순위 자료를 숙련 검토자가 라벨 비노출 상태로 독립 전사
- [ ] Train123과 평가 후보의 공식 비식별 수행자 매핑 확보
- [ ] 자격·실제 특징 검증 후 기존 모델에서 독립 CER 측정

## Checklist & Open Questions

- [x] 실제 데이터와 합성 테스트를 구분했다.
- [x] 원본·기존 결과·팀원 파일과 기존 브랜치를 보존했다.
- [x] 재학습 없이 검증했고 독립 CER를 임의 생성하지 않았다.
- [x] 별도 Codex 리뷰 지적을 수정하고 회귀를 실행했다.
- [ ] 제출자가 모든 diff를 직접 읽고 팀원에게 설명할 수 있는가?
- [ ] 실시간 구간 경계·누락 시각 정책은 어느 팀원이 승인할 것인가?
- [ ] 공식 수행자 매핑과 독립 전사 검수 담당자는 누구인가?

PR 생성과 Merge는 사용자 지시 범위 밖이다. 이 파일은 붙여넣기용 초안이다.
