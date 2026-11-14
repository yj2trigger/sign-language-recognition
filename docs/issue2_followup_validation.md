# Issue #2 후속 검증

2026-11-14 KST. Refs #2.

## 작업과 의존성

원래 저장소는 `feat/2-mediapipe-fingerspelling`, HEAD `95d53b362c4ce117ce6bdb15ae012d710aec0edb`다.
기존 미추적 35개와 원래 브랜치는 보존했다. 별도 worktree의
`feat/2-realtime-evaluation-readiness`는 Draft PR #21 Head
`f72a44a6929efd4c8ec9a72a71a523aa96f9662b`에서 출발한다.
PR #21의 검토 페이지·CSV·매핑 검사와 선행 공개 API를 재사용한다.
기존 PR #11~#20의 자모·타깃·모델·평가·데이터 기능을 다시 구현하지 않는다.
PR #21의 Base는 `feat/2-final-submission-validation`의 `6455d7d...`다.
후속 PR 초안의 Base는 `feat/2-independent-evidence-handoff`다.

## 실시간 입력 연결

팀원 소유 `src/realtime.py`의 classify는 `(timestamp, xy53, mask53, ...)` 버퍼를 사용한다.
새 `src/fingerspelling_realtime.py`는 이 형식을 받아 구간을 선택하고 기존 공개 API를 호출한다.
원래 일반 수어 분류 경로와 팀원 파일은 보존했다. 실제 웹캠 루프 연결은 미완료다.

```python
from fingerspelling_realtime import read_segment

# Existing segment boundaries and buffer from the caller.
text = read_segment(buf, start, end)
```

`FINGERSPELLING_CHECKPOINT`는 기존 MediaPipe 체크포인트의 절대 경로로 설정한다.
CPU가 기본이며 CUDA는 `FingerspellingReader(checkpoint, device='cuda')`를
`reader=` 인자로 전달한다. Reader는 초기화 후 재사용한다.
비어 있는 선택 구간은 빈 문자열이다. 잘못된 shape·mask·NaN·Inf는 기존 검증기로 거부한다.
증가하지 않는 시각과 15fps에서 2ms 넘게 벗어난 프레임 간격은 거부한다.
실제 웹캠의 누락·지터나 VFR는 명시적 시간축 처리 정책을 먼저 합의해야 한다.
프레임을 합성하거나 불규칙 시각을 15fps 관측으로 바꾸지 않는다.

기존 특징 3개를 63·97·45프레임 버퍼로 재생했다. CPU/CUDA 어댑터 결과는 각 직접 호출과
같았고 CPU/GPU 문자열도 3개 모두 같았다. RTX 5060 Laptop GPU, torch 2.10.0+cu128이다.
단일 호출 CPU 시간은 56.1·27.5·14.5ms, CUDA는 403.5·32.0·16.9ms였다.
초기화·워밍업 영향이 있는 3회 관측이며 실시간 처리량 벤치마크로 일반화할 수 없다.
기존 API 실제 계약 검사 12종도 재실행해 통과했다. 예측은 정답 근거로 사용하지 않았다.

## 사람 검토 우선순위

PR #21의 기존 비노출 페이지 519개를 그대로 연결했다. 저장된 블록 NPZ와 RGB 접촉표
해시를 모두 확인한 뒤 완전한 한 손의 관측 비율, 관절 관측 비율, 양손 완전 관측 비율,
최대 PTS 간격, 영상 ID 순서로 결정적으로 정렬한다. 무작위 선택은 사용하지 않는다.
519개 모두 순위 생성에 성공했고 누락된 selected index나 해시 오류는 0건이었다.
마지막 부분 블록과 저FPS·VFR의 실제 시각도 보존한다.

| 우선순위 | 영상 ID | 저장된 전체 시각 범위(초) |
|---|---|---|
| 1 | NIA_SL_FS0047_CROWD18_F | 0–7.95 |
| 2 | NIA_SL_FS0050_CROWD18_F | 0–7.95 |
| 3 | NIA_SL_FS0078_CROWD19_F | 0–7.95 |
| 4 | NIA_SL_FS0153_CROWD18_F | 0–8.016667 |
| 5 | NIA_SL_FS0208_CROWD18_F | 0–7.933333 |

실제 라벨 구간 시작·끝, 블록 수, PTS 범위, 선정 이유와 기존 비노출 페이지 링크는
비공개 `review-priority.csv`에 있다. 상위 20개 세부 항목은 `audit.json`에도 기록했다.
순위는 전체 영상의 기술적 관측 지표다. 라벨 구간 내 손의 선명도·움직임 흐림·수형의
의미·굽힘 판독 가능성을 인증하지 않는다. 관절 검출 성공과 RGB 가시성은 별도로 검토한다.
원래 라벨과 모델 출력은 순위 계산에 사용하지 않는다. 의미 상태는
VERIFIED 0, MISMATCH 0, UNCERTAIN 3, UNREVIEWED 516으로 유지한다.

## 기존 예측 오류

학습을 반복하지 않고 기존 `result.json`과 `heldout-diagnostic.json`의 영상별 S/D/I 및
예측 자모열을 집계했다. 원래 training-config의 분할 ID와 특징 SHA를 대조한다.
Test는 저장된 selected 이름과 체크포인트 SHA가 일치하는 모델에만 귀속한다.
기본 모델의 Test 결과는 해당 저장 파일이 없어 미측정이다.
143개 특징의 분할은 Train123, Validation12, 관찰된 Test8이다.
과거 OpenPose 1,606개 full-fit 실험과 별개의 실험이다.

| 모델·분할 | D | I | S | 정답 자모 수 | 제공 라벨 기준 자모 CER | argmax blank 비율 |
|---|---:|---:|---:|---:|---:|---:|
| 기본 Train123 | 19 | 2 | 12 | 1025 | 0.032195 | 0.781011 |
| 기본 Validation12 | 14 | 5 | 20 | 120 | 0.325000 | 0.777637 |
| 손 상대좌표 Train123 | 71 | 3 | 8 | 1025 | 0.080000 | 0.683197 |
| 손 상대좌표 Validation12 | 21 | 3 | 13 | 120 | 0.308333 | 0.726811 |
| 손 상대좌표 관찰 Test8 | 7 | 0 | 10 | 63 | 0.269841 | 0.722467 |

빈 자모 예측은 모든 집계군에서 0개다. mean_p_blank, 평균 출력 길이, 영상별 입력 프레임·
정답/출력 자모 길이·오류·손 관측률과 자모 출력 빈도는 비공개 오류 JSON에 기록했다.
Train 출력 상위 자모는 두 모델 모두 ㅇ·ㄹ·ㅗ·ㄱ·ㅏ다. 이는 출력 빈도이며 학습 라벨의
분포를 보정한 편향 추정은 아니다. Blank 비율 감소가 삭제 감소를 보장하지 않는다.
Validation에서 손 상대좌표 모델의 D는 7개 늘었고 S는 7개, I는 2개 줄었다.
전체 오류 차이는 2개이므로 이를 일반화 성능 우위로 확정하지 않는다.

모든 143개가 ‘95% 이상 프레임에서 최소 한 손 전체 관측’ 군에 속했다.
반대 품질군은 0개여서 이 구분으로 결측과 CER의 관련성을 추정할 수 없다.
실제 수형·흐림과 오류의 관계도 독립 판독 자료가 없어 미확정이다.
제공 라벨 자체의 의미가 미검증이므로 모든 CER는 참고 지표다.

## 독립 평가 준비

PR #21의 매핑·전사 CSV와 자격 검사기를 재사용했다. 부족했던 두 항목만 보완했다.
원본 영상 ID가 없으면 선언상 준비 완료를 차단한다. 같은 canonical 원본에서
평가 구간이 겹치면 양쪽 후보 모두 차단한다. 끝점만 맞닿는 구간은 겹침으로 보지 않는다.
Train에 같은 원본이 있으면 기존의 더 보수적인 전체 원본 제외 규칙이 적용된다.
이 규칙은 다른 파일명으로 재포장된 원본을 자동 식별하지 못한다. 공식 출처 대응이 필요하다.

공식 매핑·독립 전사·출처 구간·실제 특징이 확보되면 다음 순서로 연결한다.

1. 기존 `eligibility`로 Train 수행자·영상·특징·원본 중복과 전사 근거 선언을 검사한다.
2. 근거 파일 해시와 제공 주체를 사람이 확인하고 실제 특징의 원본·구간·시간축을 검증한다.
3. 기존 `FingerspellingReader`와 recognizer의 자모 토큰 출력을 사용해 추론한다.
4. 기존 `fingerspelling.cer_metrics.edit_counts`로 독립 전사 대비 S/D/I와 분모를 합산한다.
5. 자모 CER·한글 CER·완전 일치율과 수행자·영상 수를 함께 기록한다.

현재 실제 수행자 매핑·독립 전사는 0건이다. 실제 독립 평가 0개, 독립 CER 미측정이다.
이번 작업으로 이 두 종료 조건을 완료할 수는 없다. 기존 미발송 기관 문의 초안과
사람 검토 CSV가 후속 입력 경로다. 새 연락이나 파일 전송은 수행하지 않았다.

## 재현과 보호

작업 폴더는 저장소 `.tmp-issue2/followup`이며 Git에는 코드·테스트·문서만 포함한다.
`analysis-final/review-priority.csv`, `analysis-final/audit.json`, `actual-01.json`,
`errors-verified.json`, `guard-after.json`은 D드라이브 비공개 산출물이다.
오류 집계의 최종 입력 검증은 `errors-final-checked.json`에 다시 기록한다.

```powershell
$env:TEMP='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2'
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
& $py tools/audit_fingerspelling_followup.py --blocks "$data\outputs\crowd-block15-audit-v1" --pack "$repo\.tmp-issue2\evidence-handoff\review-pack-final" --models "$data\outputs\mediapipe-ctc-performance\expanded-models-01" --features "$data\outputs\mediapipe-ctc-performance\data-expansion-01\features" --output "$repo\.tmp-issue2\followup\analysis-new"
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" -q -p no:cacheprovider --basetemp "$repo\.tmp-issue2\followup\tests-new"
```

출력 경로는 새 경로여야 한다. 기존 결과는 덮어쓰지 않는다.
원본 ZIP3개·이전 산출물149개·checkpoint346개·학습 JSON15개·미추적35개·
블록 결과11,784개와 training-config 해시가 작업 전 보호 기록과 모두 일치했다.
보호 검사의 Git 접근이 sandbox에서 실패하여 동일한 읽기 전용 명령을 승인받아 재실행했다.
새 패키지 설치·학습·영상 추출·MediaPipe 재처리는 수행하지 않았다.

별도 Codex 리뷰 주체가 특징 해시 미대조와 Test 체크포인트 귀속 미검사를 발견했다.
두 항목을 수정하고 변조 차단 회귀를 추가했다. 리뷰 주체는 사람 수어 판독자가 아니다.
최종 회귀는 240 passed, 0 failed, 0 skipped, 2 warnings, 110.23초다.
19개 신규 회귀가 포함되며 경고는 기존 protobuf deprecation 2건이다.
`regression-submission.xml`에 JUnit 결과를 보존했다. 최종 별도 리뷰에서 추가 필수 결함은 없었다.
Commit/Push 증거는 제출 보고서에 기록한다.
