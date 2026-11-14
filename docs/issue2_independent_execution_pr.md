# PR 제목

[feat] #2 connect independent evidence checks to CTC evaluation

Base `feat/2-realtime-evaluation-readiness`

Head `feat/2-independent-execution`

Refs #2. 전체 의미 검증과 실제 독립 CER가 미완료이므로 Draft 제출을 권장한다.

## Why

기존 자격 검사·CTC 추론·CER 계산은 각각 존재했지만 승인된 수행자·전사 자료를 입력해
실제 측정까지 실행하는 경로가 없었다. 기하 측정은 자모 판독과 원라벨 대조로 연결되지 않았다.
이 두 연결을 보완하고 현재 데이터에서 판독·평가가 차단되는 원인을 실행 결과로 남긴다.

## What

- 기존 블록의 양손 기하에서 제한된 자모 후보를 제시하고 불확실하면 판독을 보류한다.
- 승인된 완전 전사·근거 파일·동일 구간이 있을 때 원라벨과 비교한다.
- 기존 매핑·전사 CSV를 받아 자격·근거·특징·체크포인트를 검사하고 실제 CER를 계산한다.
- 기존 처리기·모델·자모·CER·검토 생성기·API는 재사용한다.

## How

판독 함수에 라벨과 모델 출력을 전달하지 않는다. 실제 PTS·마스크·tracking context를
보존하고 블록 경계의 같은 동작은 연결한다. 완전한 독립 전사가 없으면 UNCERTAIN이다.
후보5개를 전체 철자 판독 성공으로 간주하지 않는다.

평가 실행기는 모델 결과와 학습 설정·checkpoint SHA를 결합하고 기존 eligibility를 호출한다.
Train 사람·영상·원본·특징 중복과 평가 구간·특징 중복을 차단한다.
실제 NPZ와 근거 SHA·PTS·라벨 범위·정규화·CTC 길이를 검사한 뒤 기존 모델로 추론하고
기존 edit_counts로 자모·한글 CER, S/D/I, 분모와 완전 일치율을 집계한다.
매핑·전사 근거의 진위와 승인 주체는 외부 검증 과정이 확인해야 한다.

## Where / Compare Before & After

모두 기존 Base에 없었던 신규 파일이다.

| 파일 | Before → After 실제 줄 범위 | 변경 내용 |
|---|---|---|
| `tools/read_independent_handshapes.py` | 없음 → 1–177 | 제한 수형 후보·연속 구간·근거 있는 원라벨 비교 |
| `tools/evaluate_independent_cer.py` | 없음 → 1–177 | 승인 자료 입력부터 기존 모델 추론·CER까지 연결 |
| `tests/test_independent_execution.py` | 없음 → 1–171 | 합성 정상·경계·누수·근거·시간축 회귀32개 |
| `docs/issue2_independent_execution.md` | 없음 → 1–162 | 사전 구현 표·실행 결과·재현 명령·남은 조건 |
| `docs/issue2_independent_execution_pr.md` | 신규 전체 | 이 PR 초안 |

## How I tested / Verifiable Evidence

기존 D드라이브 Conda sign-language Python을 사용했다. 모든 임시 파일은 D에 저장했다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp','D:/UNIVERSITY/UAI_project2/FILE2/sign-language-recognition/.tmp-issue2/independent-execution/regression-final','--junitxml','D:/UNIVERSITY/UAI_project2/FILE2/sign-language-recognition/.tmp-issue2/independent-execution/regression-final.xml']))"
```

272 passed, 0 failed, 0 skipped, 2 warnings in 109.04s. 기존 protobuf 경고2건이다.
별도 Codex 리뷰 지적3건을 수정하고 최종 재검토를 마쳤다.

| 실제 실행 | 결과 |
|---|---|
| 기존 영상 분석 | 14개·1,329프레임 |
| 제한 수형 후보 | 5구간·블록 경계 연결2개 |
| 독립 확정 자모 / 라벨 일치 / 불일치 확정 | 0 / 0 / 0 |
| 판독 보류 | 14영상 |
| 원본 형태소 JSON 재조사 | 2,000개·수행자/세션 필드0 |
| 실제 평가 실행기 | Train123 매핑 미확인으로 BLOCKED·독립 평가0·CER null |
| 실제 checkpoint 호환성 | CPU/CUDA 각3개·기존 API 문자열 일치 |
| 기존 MP4 무결성 | 12개·ZIP 크기/CRC 일치 |
| 기존 보호 해시 | ZIP3·산출물149·checkpoint346·JSON15·미추적35·블록파일11784·학습설정 일치 |

합성 수행자·근거와 주입 예측기로 가→거의 자모CER0.5/한글CER1.0을 검증했다.
이 값은 합성 테스트이며 실제 수행자 독립 CER나 기존 모델 성능으로 해석하지 않는다.
실제 데이터 실행 명령과 14개 ID·근거·출력 위치는 검증 문서에 있다.

## 의존성

Base SHA `03e69ee6da4514cfe204a11a0561c3ded9aa0e8a`의 PR #22와 그 선행 PR #21을 따른다.
기존 CSV·eligibility·hand_measures·Reader·features·CTC decode·CER를 직접 재사용한다.
새 패키지와 모델 학습은 없다. 기존 PR #11~#22·원격 Commit은 수정하지 않는다.

## Rejected Alternatives / Improvements & Trade-offs

완전한 자모 분류기를 임의 규칙으로 확정하면 깊이·접촉·방향·동적 복합 수형의 모호성이
숨겨지므로 현재 지원 범위를 명시하고 판독을 보류한다. 기준 각도와 시간은 탐색적 설정이다.
모델 예측을 정답으로 쓰거나 파일명으로 수행자를 가정하는 방식은 독립 근거가 없어 제외했다.
기존 템플릿·페이지 생성기·CER 구현의 중복을 피하고 실제 실행 경로만 추가했다.

## AI Use Disclosure

Codex가 공식 자료 대조, 코드·합성 테스트·실제 실행·문서 작성을 수행했다.
별도 Codex 리뷰 주체가 정규화 사전 검사, tracking context, 빈 라벨 처리를 지적했고 수정했다.
사람 수어 판독자의 독립 전사 승인은 아직 없다.

## 미완료 사항 / To-Do

- [ ] 숙련 검토자 또는 공식 자료의 완전한 자모 전사·원본 구간 확보
- [ ] Train과 평가 영상의 공식 익명 수행자 매핑 확보
- [ ] 검증된 실제 자료를 이번 실행기에 입력해 독립 CER 측정
- [ ] 제한 후보 판독의 실제 자모 정확도 검증과 지원 범위 결정

## Checklist & Open Questions

- [x] 기존 구현을 재사용하고 신규5파일만 추가했다.
- [x] 실제 실행과 합성 테스트를 구분했다.
- [x] 근거 부족을 VERIFIED나 실제 CER로 바꾸지 않았다.
- [x] 전체 회귀와 별도 Codex 리뷰를 수행했다.
- [ ] 제출자가 모든 diff를 이해하고 설명할 수 있는가?
- [ ] 승인된 수행자·전사 자료를 누가 확보하고 검수할 것인가?
