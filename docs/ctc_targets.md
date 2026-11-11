# CTC 정답 생성 및 검증 (2026-11-11)

## 확인 범위와 의존성

작업 브랜치 `feat/2-ctc-targets`의 시작 HEAD는 `2f10cda`로
`feat/2-label-preparation`과 동일하다. 기존 미추적 `.vscode/`는 수정하지 않았다.
PR #11과 #12는 조회 시 open / merged=false였다.

- [Issue #2](https://github.com/yj2trigger/sign-language-recognition/issues/2)는 37종을 언급하지만 목록·ID·숫자 정책의 근거를 제공하지 않는다.
- [PR #11](https://github.com/yj2trigger/sign-language-recognition/pull/11)의 검토한 commit은 `8f23314c7f924081f38d2b34b65996931398bb7c`다.
  `decompose_hangul(text, preserve_boundaries=False)`는 NFC 후 호환 자모 list를 반환한다.
  초성 19, 중성 21, 종성 27개(역할 간 중복 있음), 고유 자모는 51개다.
  겹받침·쌍자음·복합 모음은 분리하지 않는다. 고립 결합 자모·옛한글·숫자·공백은 거부한다.
  Unicode 자모는 실제 수어 지문자 클래스와 같은 개념이 아니다.
- [PR #12](https://github.com/yj2trigger/sign-language-recognition/pull/12)의 `read_crowd_zip`과
  `CrowdReport.records[*].target.text`를 그대로 재사용한다. 기존 파서·어댑터는 수정하지 않았다.
- 저장소·README·설계 문서·Issue·두 PR에서 공식 37종 목록을 확인하지 못했다.
  공식 출처의 목록 및 클래스별 대응표를 확보해야 한다. 51종 진단 목록도 공식 목록이 아니다.

현재 브랜치에는 jamo.py가 없다. 복제·신규 구현·cherry-pick하지 않았다.
기본 encoder는 해당 모듈을 지연 import하며 없으면 `missing_pr11_dependency`로 실패한다.
PR #11 반영 후 기본 연결을 사용할 수 있고, 반영 전에는 검토된 분리 함수를 명시적으로 주입할 수 있다.

## API와 정책

`TokenVocabulary(token_ids, blank_id=...)`는 호출자가 지정한 token→ID를 복사한다.
등록되지 않은 토큰, 중복 ID, 음수·bool ID, blank 충돌을 거부한다.
순서로 ID를 추측하지 않으며 sparse ID의 logits 폭은 `num_classes`로 제공한다.
모델과 vocabulary 및 정책을 함께 버전 관리해야 한다.

`TargetEncoder(vocabulary, decompose=None, literal_mapping=None).encode(text, input_length=None)`:

- 원문과 NFC 문자열을 별도로 보존하고 NFKC를 적용하지 않는다.
- 초성·종성의 같은 호환 자모는 같은 명시적 ID로 간다. 복합 자모의 추가 분해는 하지 않는다.
- 숫자 읽기·공백 삭제·미지원 문자 삭제는 하지 않는다. 기본은 이유 코드로 오류 처리한다.
- 숫자·공백은 NFC 문자별 `literal_mapping={"1": ("digit-one",), " ": ("space",)}`처럼
  명시한 정책이 있을 때만 변환한다. 출력 토큰은 vocabulary에 등록되어야 한다.
- 빈 입력은 `empty_text` 오류다. 기존 라벨 파서도 빈/공백뿐인 문자열을 거부한다.
- CTCTarget의 ids에는 blank를 넣지 않는다. 반복 토큰을 축약하지 않는다.
- 최소 모델 출력 시점 수 = 정답 길이 + 인접한 동일 ID 쌍의 수.
  `input_length`는 원본 영상 프레임 수가 아니라 subsampling 이후 모델 출력 시점 수다.
  생략하면 시간 길이는 미검증이며 **학습 투입 전 반드시 실제 길이를 제공해야 한다**.
- 기본 자모열은 음절 경계·초종성 역할을 잃는다. 경계 마커를 정답에 넣지 않는다.
  추후 decoding/CER는 이 제약과 PR #11 조합 계약을 고려해야 한다.

`audit_crowd(report, encoder)`는 개인정보 없는 집계만 반환한다.
정답 원문·토큰·ID 배열은 로컬 비공개 자료이며 직렬화하거나 로그로 출력하면 안 된다.
빈 문자열 통계는 정상 adapter records만 대상으로 하며 파싱 실패 내 원문은 검사하지 않는다.
미지원 유형은 NFC 문자 출현 수이며 라벨 수가 아니다. 미등록 자모 토큰은 별도 오류 코드로 집계한다.

## 검증

환경: `D:/UNIVERSITY/DOWNLOAD/MINI_CONDA/REAL_MINICONDA/envs/sign-language/python.exe`.
추가 패키지는 설치하지 않았다. PyTorch가 필요한 학습·실시간 테스트는 이번 검증 범위 밖이다.

- 신규 두 테스트 + 기존 라벨/CROWD: 67 passed, 1 skipped, 76 subtests passed.
  skip은 현재 브랜치에 PR #11 모듈이 없는 통합 테스트다.
- PR #11 Git 코드를 메모리에서 연결한 신규 테스트: 33 passed.
  통합 테스트는 현대 음절 11,172개 전체와 NFC/NFD, 겹받침·복합 모음·옛 자모 거부를 확인한다.
- 기존 preprocess/extract_full/select_classes: 18 passed, 기존 protobuf deprecation 경고 2개.

실제 ZIP은 기존 문서의 CROWD Validation ZIP을 읽기 전용으로 처리했다.
`tools/audit_ctc_targets.py <로컬 ZIP 경로> --reviewed-pr11-ref 8f23314c7f924081f38d2b34b65996931398bb7c`
명령은 신뢰한 Git ref 코드만 메모리 실행한다. 파일 복제·ZIP 추출은 하지 않는다.
PR #11의 JAMO를 정렬한 **51개 Unicode 진단 토큰**, blank=0,
숫자·공백 기본 거부 정책으로 검사했다. ZIP SHA-256 전후 일치.

| 항목 | 수 |
|---|---:|
| 전체 JSON / 정상 파싱 | 2,000 / 2,000 |
| 파싱 실패 | 0 |
| 진단 변환 성공 | 1,616 |
| 변환 불가능 | 384 |
| 숫자 포함 / 성공 / 거부 | 382 / 0 / 382 |
| 빈 문자열(정상 파싱 대상) | 0 |
| 미지원 문자 출현: decimal / Ll / Lu | 596 / 24 / 2 |
| 최초 오류: 숫자 정책 필요 / 기타 문자 | 382 / 2 |
| 동일 FS 원문 불일치 그룹 | 3 |

문자 유형별 수는 모든 미지원 문자를 세며, 최초 오류 라벨 수와 합계가 다르다.
숫자 포함 라벨 382개는 Issue의 192개와 범위가 다를 수 있어 동일 통계로 단정하지 않는다.
변환 성공은 **공식 37종 호환성·인식 성능·영상 정합성·학습 가능성 완료 판정이 아니다**.
실제 영상/모델 출력 길이가 없어 시간 길이 검증은 미수행이다.

## 독립 리뷰 및 다음 순서

구현 후 별도 리뷰 에이전트가 읽기 전용으로 검토했다. 발견한 NFC로 변하는 literal key의
도달 불가능 문제를 거부 검증으로 수정했다. 옛 호환 자모/채움 문자 거부가 문자 유형 통계에
누락되는 문제도 수정했다. 각각 회귀 테스트를 추가했다. 빈 문자열 집계 범위와 시간 길이
미검증 조건을 문서에 명시했다. 외부 사람의 리뷰는 수행하지 않았다.

1. 공식 37종 목록·출처·ID 및 종성/복합 자모·숫자·공백 대응 정책을 확정한다.
2. PR #11·#12를 검토·반영하고 이 브랜치에서 기본 연결 통합 테스트를 다시 실행한다.
3. 불일치 3그룹과 미지원 라벨을 비공개 검토하고 실제 영상 목록·내용을 대조한다.
4. 확정 vocabulary를 주입해 재검증하고 모델 출력 길이를 연결한다.
5. CTC 학습·CER·실시간 연결을 진행한다. Issue #2 전체 완료는 아직 아니다.

커밋·Push·PR 생성·Merge는 수행하지 않았다.
