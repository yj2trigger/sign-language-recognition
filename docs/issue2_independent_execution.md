# Issue #2 독립 판독 후보와 CER 실행 연결

2026-11-14 KST. Refs #2. 전체 지문자 철자 판독과 실제 수행자 독립 CER는 미완료다.
이번에 실행 가능하게 만든 것은 제한된 수형 후보 분석, 근거 있는 전사와 원라벨의 비교,
검증 자료를 입력받아 기존 체크포인트 추론과 CER 계산까지 이어지는 실행 경로다.

## 코드 작성 전 확인

| 기능 | 기존 구현 | 실제 실행 가능성 | 이번 보완 |
|---|---|---|---|
| 독립 지문자 판독 | 각도·거리 측정과 수동 도판 대조 | 완전한 자모 판독기 없음 | 제한된 수형 후보와 판독 보류 |
| 실제 영상·라벨 비교 | 수동 사례 기록 | 승인 전사 입력의 자동 비교 경로 없음 | 동일 구간 전사·근거 검사 후 대조 |
| 판독 근거 기록 | JSON·PTS·RGB·해시 | 가능 | 기존 자료·측정 함수 재사용 |
| 공식 수행자 매핑 로딩 | CSV·매핑 검사 | 파일 근거·평가 실행 연결 필요 | 기존 CSV 로딩과 근거 해시 연결 |
| 수행자 독립 분할 검사 | eligibility 구현 | 가능 | 그대로 호출 |
| 독립 CER 계산 연결 | 추론·CER 개별 구현 | 자격 검사부터 실제 측정까지 실행기 없음 | 기존 구성요소를 하나의 CLI로 연결 |

MediaPipe·블록 처리·재시도·자모·BiGRU/CTC·디코딩·공개 API·실시간 어댑터·
검토 페이지·CSV 생성기·CER 알고리즘·오류 분석은 변경하지 않았다.
새 브랜치 `feat/2-independent-execution`은 PR #22 Head
`03e69ee6da4514cfe204a11a0561c3ded9aa0e8a`를 Base로 별도 D worktree에 생성했다.
PR #11~#22와 원래 루트 브랜치·미추적 파일을 보존한다.

## 독립 수형 후보

국립국어원 교재의 인쇄면 186–187, 표8·표9를 확인했다. 자모별 수형 도판과
복합 표현을 제공하지만 MediaPipe 좌표의 각도 임계값이나 CROWD 영상 정답은 제공하지 않는다.
[공식 한국수어 교육 교재](https://www.korean.go.kr/common/download.do?file_path=etcData&c_file_name=c7f3fcaa-016a-4b72-8803-b597ab5ca52b_0.pdf).

새 도구는 기존 `hand_measures`로 관절 각도를 계산하고 영상 좌표의 검지 방향과
손가락 굽힘·펴짐을 조합한다. 수직 검지 수형에서 ㄱ/ㅜ 후보, 엄지까지 굽은 위쪽 검지에서
ㅣ 후보만 다룬다. 다른 자모·깊이·가림·손바닥 방향을 배제할 수 없어 항상 UNCERTAIN이다.
155°/125° 각도, 방향 성분 0.8, 최소 3관측·0.12초, 최대 간격 0.25초는 탐색적 구현
설정이다. 공식 표준 임계값이나 검증된 분류 성능으로 해석하지 않는다.
손목·관절 측정은 원래 53점 중 양손만 사용하며 좌우 교환·반전은 하지 않는다.

각 관측의 실제 PTS, 마스크, block 및 tracking context를 보존한다.
같은 context에서 이어지는 후보를 블록 경계 너머로 연결하고 context 변경이나 긴 관측
공백에서는 끊는다. 블록을 자모로 간주하거나 존재하지 않는 프레임을 보간해 판독하지 않는다.
후보 함수에는 원래 라벨과 CTC 출력을 전달하지 않는다. 라벨은 후보 분석 후에만 읽는다.

`compare_labels`는 기존 검토 CSV에서 정확한 같은 구간의 완전한 독립 전사를 받으면
기존 `review_valid`, 근거 파일 해시, 자모 처리기를 호출해 원라벨과 비교한다.
일치하면 VERIFIED, 다르면 MISMATCH를 반환한다. 전사·근거가 없거나 구간이 다르면
UNCERTAIN이다. 파일 해시는 승인된 근거 자료와의 일치 검사이며 검수자·자료 진위의
인증은 제공기관·검토자 승인 과정에서 확인해야 한다. 기하 후보만으로 전사를 확정하지 않는다.

## 실제 실행 결과

기존 원본 MP4가 남아 있는 12사례와 앞선 대조 사례 2개를 추가해 총14개를 분석했다.
기존 MP4 12개는 ZIP 멤버 크기·CRC와 모두 일치했다. 새 영상 추출·특징 생성은 없다.
전체 1,329프레임에서 양손 관측 2,658회를 처리했다.
손 관절 누락 117회, 지원 규칙 밖 2,496회, 제한된 후보 45회다.
후보 hold는 5개이며 그중 2개가 블록 경계를 넘는다. 모든 hold는 ㄱ/ㅜ의 복수 후보다.

| 영상 ID | 후보 PTS(초) | 연결 블록 |
|---|---|---|
| NIA_SL_FS0485_CROWD19_F | 4.083333–4.416667 | 3 |
| NIA_SL_FS0735_CROWD18_F | 0.966667–1.266667 | 1,2 |
| NIA_SL_FS0495_CROWD19_F | 8.200000–8.683333 | 6 |
| NIA_SL_FS0001_CROWD18_F | 3.733333–4.066667 | 4,5 |
| NIA_SL_FS0001_CROWD19_F | 3.466667–3.866667 | 4 |

기존 RGB crop도 공식 도판과 시각 대조했다. 검지·엄지 방향, 펴진 손가락과 굽은
손가락이 관찰되지만 단일 crop으로 전체 구간의 연속 철자를 입증하지 못했다.
Codex가 수행했으며 숙련 수어 판독자의 독립 전사 검수는 아니다. 이전 라벨을 이미 본
사례이므로 이 시각 대조를 새 맹검 정답으로 채택하지 않았다.

독립 확정 자모0, 라벨 일치 확인0, 불일치 확정0, 판독 보류14영상이다.
후보 유무나 정답 자모가 후보 집합에 포함되는지만으로 라벨 일치를 확정하지 않았다.
현재 범위는 완전한 한국어 지문자 판독기가 아니다. 제한 규칙·2D 투영·깊이·접촉·방향·
복합 수형 및 연속 자모 경계의 불확실성이 차단 원인이다.
원래 검토 CSV의 UNCERTAIN3/UNREVIEWED516은 수정하지 않았다. 새14개 결과는
별도 파일의 자동 분석 결과이며 원래 검토 상태와 구분한다.

실제 분석 ID 전체는 다음과 같다.

```text
NIA_SL_FS0450_CROWD19_F NIA_SL_FS0476_CROWD19_F NIA_SL_FS0486_CROWD19_F
NIA_SL_FS0485_CROWD19_F NIA_SL_FS0490_CROWD19_F NIA_SL_FS0482_CROWD19_F
NIA_SL_FS0735_CROWD18_F NIA_SL_FS0646_CROWD18_F NIA_SL_FS0492_CROWD19_F
NIA_SL_FS0501_CROWD19_F NIA_SL_FS0495_CROWD19_F NIA_SL_FS0483_CROWD19_F
NIA_SL_FS0001_CROWD18_F NIA_SL_FS0001_CROWD19_F
```

## 수행자 근거와 평가 실행

실제 형태소 JSON2,000개를 다시 읽었다. 필드 경로는 metaData 및 url/name/duration/
exportedOn, data 및 start/end/attributes/name이다. performer/signer/speaker/provider/
session 계열 필드는 0개다. 이름·URL·CROWD18/19를 사람 식별자로 사용하지 않았다.
기존 ZIP 추가 메타데이터 조사와 키포인트 person_id=-1 표본 결과를 재사용했다.
공식 설명의 CROWD 언어제공자21명은 현재 영상별 수행자 대응표를 제공하지 않는다.
공식 name은 형태소 표제어로 설명된다. [AI-Hub103](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103).
기존 접근 실패 문서 URL을 반복 요청하거나 승인 조건을 우회하지 않았다.

새 실행기는 다음을 연결한다.

1. 학습 설정 SHA와 inference checkpoint SHA를 기존 모델 결과에 대조한다.
2. 기존 매핑·전사 CSV와 eligibility로 Train 사람·영상·원본·특징 및 평가 구간 겹침을 검사한다.
3. 승인 매핑·독립 전사의 근거 파일을 evidence-root 안에서 읽어 SHA를 확인한다.
4. 후보별 실제 NPZ·PTS·라벨 구간·마스크·정규화·CTC 최소 길이를 검사한다.
5. 평가 후보 간 동일 특징 SHA도 제외한 뒤 기존 Reader/CTC 디코더로 추론한다.
6. 기존 edit_counts로 자모·한글 S/D/I/분모·CER·완전 일치율을 집계한다.

모든 후보의 전처리 검사를 모델 초기화 전에 수행한다. VFR/저FPS NPZ가 고정15fps
계약과 맞지 않으면 명시적 시간축 정책이 필요하다는 사유로 제외한다.
CSV의 mapping_source/evidence_source에는 evidence-root 기준 근거 파일 경로를 넣는다.
기존 CSV 열을 재사용하며 새 템플릿을 생성하지 않는다. 승인 주체·근거·hash·VERIFIED
상태가 실제 외부 검증 결과여야 한다. 일반 텍스트 파일을 해시한 것만으로 승인될 수 없다.
원본 ID 대응과 클립 경계의 진위도 제공기관의 신뢰 가능한 자료가 필요하다.

실제2,000매핑·519검토행을 입력해 실행했고 Train123의 매핑이 모두 미확인이라
BLOCKED, 실제 독립 평가0, 자모/한글 CER null을 기록했다.
별도로 새 predictor 경로의 CPU/CUDA 각3개 기존 특징 추론이 기존 API와 문자열 일치했다.
이는 모델 호환성 검사다. 실제 독립 CER로 기록하지 않았다.

합성 수행자·근거·특징과 주입된 예측기로도 연결 경로를 실행했다.
합성 가→거 사례는 기존 CER로 자모0.5, 한글1.0, 치환1을 계산했다.
이 수치는 합성 테스트의 기대값이며 실제 모델이나 수행자의 성능 수치가 아니다.

## 재현

모든 명령은 신규 worktree에서 기존 Conda Python으로 실행한다. 출력은 새 D 경로로 지정한다.

```powershell
$repo='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$data='D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd'
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$env:TEMP="$repo\.tmp-issue2"
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$ids='NIA_SL_FS0450_CROWD19_F NIA_SL_FS0476_CROWD19_F NIA_SL_FS0486_CROWD19_F NIA_SL_FS0485_CROWD19_F NIA_SL_FS0490_CROWD19_F NIA_SL_FS0482_CROWD19_F NIA_SL_FS0735_CROWD18_F NIA_SL_FS0646_CROWD18_F NIA_SL_FS0492_CROWD19_F NIA_SL_FS0501_CROWD19_F NIA_SL_FS0495_CROWD19_F NIA_SL_FS0483_CROWD19_F NIA_SL_FS0001_CROWD18_F NIA_SL_FS0001_CROWD19_F'.Split(' ')
$videoArgs=@()
foreach($id in $ids){$videoArgs+=@('--video-id',$id)}
& $py tools/read_independent_handshapes.py --blocks "$data\outputs\crowd-block15-audit-v1" @videoArgs --output "$repo\.tmp-issue2\independent-execution\reading-new"
$model="$data\outputs\mediapipe-ctc-performance\expanded-models-01\expanded-hand-local"
$previous="$data\outputs\mediapipe-ctc-performance\data-expansion-01"
$pack="$repo\.tmp-issue2\evidence-handoff\review-pack-final"
& $py tools/evaluate_independent_cer.py --training-config "$previous\training-config.json" --model-result "$model\result.json" --checkpoint "$model\inference.pt" --mapping-csv "$pack\performer-mapping-template.csv" --review-csv "$pack\blind-transcription-template.csv" --feature-root "$previous\features" --evidence-root "$repo\.tmp-issue2\evidence-handoff" --output "$repo\.tmp-issue2\independent-execution\evaluation-new.json"
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" -q -p no:cacheprovider --basetemp "$repo\.tmp-issue2\independent-execution\tests-new"
```

승인된 전사와 라벨 비교에는 판독 도구의 `--reviews <기존CSV> --evidence-root <근거폴더>`를 추가한다.
현재 실제 결과는 `.tmp-issue2/independent-execution`의 `reading-final/readings.json`,
`actual-execution.json`, `evaluation-final.json`, `regression-final.xml`에 보존한다.
원본 좌표·라벨·영상·개인별 상세 매핑·예측·체크포인트와 이 비공개 결과를 Git에 넣지 않는다.

## 테스트와 독립 리뷰

최종 전체 회귀272 passed, 0 failed, 0 skipped, 2 warnings, 109.04초다.
신규32개는 정상 연결, 부분 결측, VFR/저FPS, 블록 경계·context 변경, 잘못된 시각,
손 누락, 라벨 불일치, 미확인/모델 기반 전사, 수행자·원본·구간·특징 중복과 근거 변조를 포함한다.
경고는 기존 protobuf deprecation이다. 테스트 데이터의 수행자와 전사는 모두 합성이다.
별도 Codex 리뷰가 어깨 정규화 불가 입력의 사전 차단, context 경계 보존, 빈 라벨 처리를
지적했다. 세 항목을 수정하고 회귀를 추가했으며 최종 리뷰에서 추가 필수 결함은 없었다.
원본 ZIP3개·특징 등149개·checkpoint346개·학습JSON15개·미추적35개·블록파일11,784개와
학습 설정의 보호 해시가 이전 기준과 모두 일치했다. 원래 worktree·기존 PR·브랜치를 보호했다.

남은 필수 자료는 공식 익명 수행자와 Train/평가 영상의 대응표, 숙련 검토자 또는 공식
자료의 완전한 독립 자모 전사, 원본 클립·구간 대응이다. 이 자료가 없으면 전체 의미 검증과
실제 수행자 독립 CER를 완료할 수 없다. 새 실행 경로는 그 자료를 입력받아 검사·추론·측정한다.
