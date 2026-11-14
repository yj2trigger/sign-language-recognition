# Issue #2 독립 평가 근거 확보 및 검토 인계

2026-11-14 KST. Refs #2. 수행자 독립 CER와 독립 철자 정답은 아직 미확보다.
기술 테스트·추론 성공을 이 두 조건의 완료로 대체하지 않는다.
이번 작업은 기존 자료·공개 제공 경로의 추가 조사와 실제 검토 입력 자료 생성이다.

## 1. 필수 조건 A: 수행자 독립 CER

이전 URL 추정을 반복하지 않고 공식 문서의 실제 응답, ZIP 추가 파일, 구축기관과 연구자
공개 자료를 조사했다. 기존 2,000개 형태소 JSON·그룹당 키포인트 1프레임 2,000개 조사
기록도 보존한다. 형태소에는 수행자 필드가 없고, 표본 키포인트 person_id는 모두 -1이다.
파일명·URL·CROWD18/19나 얼굴·신체 특징을 사람 식별자로 채택하지 않았다.

| 경로 | 이번에 실제 확인한 결과 | 판정 한계 |
|---|---|---|
| 공식 설명서 7969 | GET 200, text/html, 79바이트 | PDF 본문 아님 |
| 공식 가이드 18941 | GET 200, text/html, 79바이트 | PDF 본문 아님 |
| 보유 원본 ZIP 3개 | MP4·JSON 외 추가 메타데이터 파일 0개 | JSON 내부 필드의 완전 부재를 뜻하지 않음 |
| 키포인트 ZIP 중앙 목록 | JSON 671,745개 모두 canonical 영상 ID 포함 | JSON 내용을 이번에 전수 재분석한 것은 아님 |
| 구축기관 공개 설명 | 수집·가공 플랫폼과 수어 과제 설명 | CROWD 영상별 수행자 매핑 없음 |
| KSL-Guide 저자 프로젝트 | 공개 루트에 README 1개, AI-Hub 103 링크 | 개인별 대응표·독립 자모 전사 파일 미확보 |

두 공식 파일 응답은 ‘정상적인 접근이 아닙니다’라는 스크립트다. 이를 로그인 필요라고
단정하거나 페이지 승인 조건을 우회하지 않았다.
[설명서](https://aihub.or.kr/file/down.do?fileSn=7969&dataMnlFileSn=7969&dataSetSn=103),
[가이드](https://aihub.or.kr/file/down.do?fileSn=18941&cnstcPrcuseFileSn=18941&dataSetSn=103).

[AI-Hub 103](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)은 수집군의 언어제공자
21명과 형태소 name이 표제어임을 설명한다. 이 설명만으로 현재 2,000개 영상의 사람이나
실제 자모 순서를 복원할 수 없다.
[구축기관 과제 소개](https://blog.aiworkx.ai/nia-sign-language-ai-data-sets/)는 구축·검수
체계의 맥락이며 영상별 인증 매핑 자료가 아니다.
[현재 blackolive 소개](https://aiworkx.ai/contents/blackolive.html?lan=ko)에서도 과제 전용
매핑을 확보하지 못했다.
[KSL-Guide 프로젝트](https://github.com/ChelseaGH/KSL-Guide)는 저자 측 연구 경로지만
이번에 읽은 README에는 CROWD 대응표가 없었다. 연결된 IEEE DOI의 본문은 도구로
열람하지 못해 논문 전체 내용에 대한 부재 주장은 하지 않는다.

**수행자 매핑 확보 0건, 확인된 비식별 ID 0개, 실제 사람 수 미확정.**
기존 Train123의 수행자를 모두 확인할 수 없어 평가 자격을 입증한 영상은 0개다.
자모 CER·한글 CER·완전 일치율·S/D/I 및 정답 글자 수는 미측정이다. 분모 0의 CER를
0으로 표시하지 않는다. 현재 자료의 숫자·의복·외형으로 임의 holdout을 만들지 않았다.

## 2. 필수 조건 B: 영상·정답 의미

기존 519개 성공 영상·4,590개 블록과 앞서 공식 지문자 도판을 대조한 3개 사례를 재사용한다.
이전 결과의 18확인시점·384관측, 공식 수형 대조 및 한계는 선행 보고서에 있다.
이 작업에서도 CTC 출력은 정답이나 VERIFIED 판정 근거로 사용하지 않았다.

새 인계 자료는 519개 라벨 비노출 HTML과 독립 전사 CSV다. 영상별 페이지에는 원래
start/end와 모든 저장 블록의 PTS·기존 RGB 접촉표를 보여 주며 원래 name은 숨긴다.
기존 원본 MP4 12개는 크기·CRC가 ZIP 멤버와 일치할 때만 로컬 재생 링크로 연결했다.
새 MP4 추출이나 MediaPipe 검출·특징 재생성은 하지 않았다.

원본 MP4가 이미 있는 기존 검토 사례 1개도 직접 열람했다. 저장 PTS와 대조한 원본
프레임 18·25·33·47의 1.266667·1.766667·2.466667·3.700000초에서 손 부분을 원본
해상도로 확인했다. source는 1080×1920이며 관찰용 crop과 직사각형 위치를 별도 보존했다.
이는 새 특징 추출이 아닌 기존 MP4의 시각 검토다. 정적 수형의 차이는 더 잘 보였으나
한 시점의 움직임 흐림과 비맹검 선택·독립 전사 부재 때문에 전체 철자 확정으로 승격하지
않았다. 검토 주체는 Codex 시각 검토이며 숙련 수어 검토자의 인증을 뜻하지 않는다.

| 상태 | 현재 수 |
|---|---:|
| VERIFIED | 0 |
| MISMATCH | 0 |
| UNCERTAIN | 3 |
| NOT_FINGERSPELLING | 0 |
| UNREVIEWED | 516 |

누적 의미 검토 시도는 3영상이며 이번 원본 RGB 재확인은 그중 1영상이다.
이 표의 0 MISMATCH는 라벨 정확성 인증이 아니다. 519개 기술 성공을 의미 성공으로
확대하지 않고, 새 사람 검토가 들어오기 전까지 원본 라벨·학습 타깃을 바꾸지 않는다.

## 3. 실제 생성한 비공개 인계 자료

프로젝트 `.tmp-issue2/evidence-handoff/review-pack-v2`에 다음을 생성했다. Git에는 넣지 않는다.

- `blind-index.html`, 영상별 519개 HTML: 원문 라벨·모델 예측을 숨긴 순차 검토 자료.
- `blind-transcription-template.csv`: 영상 ID, 원래 구간, 5종 상태, 독립 전사, 근거·검수자,
  근거 해시, 특징 계약 확인 칸. 사람 판독 전에 라벨 대조 파일을 열지 않는다.
- `adjudication-private-do-not-show-before-transcription.csv`: 독립 전사 이후에만 대조할 원래 라벨.
- `performer-mapping-template.csv`: 전체 2,000개 영상 ID와 미확인 매핑 칸. 실명·생체정보를 요구하지 않는다.
- `readiness.json`: Train123 매핑 미확인, 선언상 평가 준비 0, 실제 독립 평가 0, CER null.

공식 자료가 제공되거나 독립 검토자가 전사한 뒤 근거 파일·필드·구간·검토 주체·신뢰
수준을 함께 남긴다. 원래 name에 맞추어 전사를 고치지 말고 먼저 자체 전사를 기록한다.
그 후 지문자 여부와 원래 라벨의 일치를 대조해 상태를 결정한다. 불일치 확인과 읽지
못한 경우를 구분하고, 이름·원본 좌표·인물 정보는 GitHub로 재배포하지 않는다.

## 4. 평가 자격 선언 검사

`tools/build_independent_review_handoff.py`는 중복·별칭 영상 ID, Train 인물·영상·원본
겹침, 동일 특징 해시, 근거 없는 VERIFIED, 모델 예측 기반 전사를 차단한다.
Train 사람 매핑이 일부라도 없으면 평가 후보를 준비 완료로 선언하지 않는다.
수행자 코드는 문자열·미식별값 검사를 하고 앞뒤 공백을 제거해 중복을 비교한다.

이 검사는 **입력된 근거 선언의 형식·정합성 검사**다. 제공기관 응답 진위나 실제 특징
배열을 자동 인증하지 않으며 평가를 실행한 것도 아니다. declared_ready와 실제 평가 수는
별도다. 적격 근거가 생기면 근거 파일 검증 → 실제 특징/구간/시간축·checkpoint 계약
검증 → 기존 모델 추론 → 자모/한글 S+D+I 및 분모·수행자별 집계를 실제 실행해야 한다.
현재는 첫 단계가 성립하지 않아 모델을 독립 평가 목적으로 호출하지 않았다.

## 5. 외부 자료 요청의 구체적 다음 단계

현재 공개 조사로는 자료 제공 가능 여부를 확인받지 못했다. 자료가 존재하지 않는다고
결론내리지도 않았다. 정상 열람 방법, 비식별 매핑, 전사 정의·검수 자료를 질의할
[미발송 문의 초안](issue2_provider_inquiry_draft.md)에 실제 공개 연락처와 최소 파일 스키마를
작성했다. 사용자 승인 없이 이메일·문의 폼 제출·파일 전송을 하지 않았다.

완료에 필요한 것은 다음 세 자료다.

1. 실제 언어제공자를 구분하는 인증된 익명 코드와 Train123 및 후보 영상의 대응표.
2. 평가 영상의 독립 자모 순서·구간·지문자 유형 및 전사/검수 근거.
3. 원본 클립·겹침 관계와 기존 15fps/53점 특징의 호환성을 입증할 자료.

매핑 확인 결과 모든 보유 수행자가 Train에 있으면, 새로운 비학습 수행자 자료가 별도로
필요하다. 이번 범위에서는 이를 해결하려고 재학습하거나 새 대량 전처리를 시작하지 않는다.

## 6. 재현·검증·보호

기존 Conda Python만 사용하며 출력은 새 D 경로여야 한다. 기존 출력 폴더는 거부한다.

```powershell
$env:TEMP='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2'
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
& $py tools/build_independent_review_handoff.py --blocks "$data\outputs\crowd-block15-audit-v1" --previous "$data\outputs\mediapipe-ctc-performance\data-expansion-01" --output 'D:\new-results\independent-review'
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" -q -p no:cacheprovider --basetemp 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\evidence-handoff\regression-final'
```

최종 전체 회귀는 **221 passed, 2 warnings in 108.50s**다. 신규 검토·선언 검사 30개를
포함하며 기존 CER·checkpoint 테스트도 실행했다. 경고는 기존 protobuf 경고다.
독립 리뷰의 수행자 코드 공백·타입·미식별값 및 공백 근거 허용 문제를 수정하고 재검증했다.
추가 필수 결함은 발견되지 않았다.
원본 ZIP 3개, 기존 블록 결과 파일 11,784개, 선정 산출물 149개, 기존 미추적 35개,
발견된 checkpoint 346개, 확장 학습 JSON 15개 및 원래 training-config 해시는 이전 보호
기준과 완전히 일치했다. 라벨 비노출 페이지 519개에 대한 원래 라벨 문자열 노출 검사도
0건이었다. 실제 Commit/Push SHA는 작업 종료 보고서에 기록한다.
기존 checkpoint 호환성·CER 구현은 선행 실제 API 검증과 현재 회귀를 재사용한다.
원래 `feat/2-mediapipe-fingerspelling`과 `95d53b3...`, 기존 PR·16개 브랜치는 보존한다.
새 브랜치는 `feat/2-independent-evidence-handoff`, Base는 `6455d7d...`의
`feat/2-final-submission-validation`이다. PR 생성·Merge는 하지 않는다.

**Issue #2 최종 완료: 아니오.** 수행자 독립 평가와 독립 철자 전사 자료가 확보되지 않았다.

## 7. 제출용 PR 초안

제목: `[test] #2 독립 평가 근거 검사와 라벨 비노출 검토 인계`

본문:

Refs #2. 수행자 비중복·독립 전사 근거가 없는 상태에서 CER를 확정하지 않도록 선언 검사를
추가하고, 기존 관측 자료를 사람이 라벨 비노출 상태로 검토할 수 있게 한다. 기관 자료
요청 초안을 작성했으며 발송하지 않았다. 이름·원본·좌표·검토 CSV는 Git에 넣지 않는다.

Base: `feat/2-final-submission-validation` (`6455d7d...`).
Head: `feat/2-independent-evidence-handoff`. 기존 PR·16개 브랜치는 변경하지 않는다.

| 파일 | 이전 → 이후 줄 | 변경 이유·방식 |
|---|---|---|
| `tools/build_independent_review_handoff.py` | 없음 → 1–163 | 기존 파일만 참조하는 검토 페이지·CSV와 평가 자격 선언 검사 |
| `tests/test_independent_review_handoff.py` | 없음 → 1–112 | 중복·공백 ID·미검증 전사·시간·원본 CRC·덮어쓰기 차단 |
| `docs/issue2_provider_inquiry_draft.md` | 없음 → 1–77 | 실제 공개 문의처·필요 파일·질문, 미발송 상태 |
| `docs/issue2_independent_evidence_handoff.md` | 없음 → 1–173 | 공식 자료 접근 결과·실제 측정·한계·이 인계 절차 |

How: 원본/기존 결과를 읽고 별도 D 경로에만 검토 자료를 생성한다. CTC 예측을 정답으로
쓰지 않는다. 기존 원본 MP4 12개는 크기·CRC 확인 후 링크만 제공한다.
Tests: 위 재현 명령으로 221개 통과, 경고 2개. 보호 해시 완전 일치, 독립 리뷰 완료.
Limitations: VERIFIED 0, UNCERTAIN 3, UNREVIEWED 516, 실제 독립 평가 0, CER 미측정.
declared_ready는 증거 선언 형식 검사이며 외부 자료 진위나 모델 평가 완료가 아니다.
기존 기능 구현·공개 API 검증과 이번 두 필수 조건의 미완료를 구분한다. PR 생성·Merge 없음.
