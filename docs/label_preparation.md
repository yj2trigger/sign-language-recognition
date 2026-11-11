# 지문자 정답 준비 분석 (2026-11-11)

## CROWD 실제 자료 검증 및 어댑터 확장

이 절은 아래 최초 조사 당시의 미확인 사항을 갱신한다. 이제 확보한 CROWD 라벨의
영상명과 문자열 대응은 확인했으며 `src/fingerspelling/crowd_labels.py`에 구현했다.
도로명 분류·자모 토큰화·영상 내용 대조는 여전히 수행하지 않았다.

요청 경로의 `D:/UNIVERS/`는 존재하지 않았다. 실제 파일은
`D:/UNIVERSITY/UAI_project2/sign-language-data/fingerspelling/crowd/수어 영상/2.Validation/[라벨]01_crowd_morpheme.zip`
이며 952,783바이트다. ZIP은 읽기 전용으로 열고 해제하지 않았다.

확인된 형식(값·실제 ID는 생략):

```text
morpheme/{2자리 ID}/NIA_SL_FS{4자리 ID}_CROWD{2자리 ID}_F_morpheme.json
metaData: {url: str, name: str, duration: number, exportedOn: str}
data: [{start: number, end: number, attributes: [{name: str}]}]
```

모든 2,000개 파일에서 구간·이름 속성은 각각 하나이고, 예상 mp4 이름은
`metaData.name`, `metaData.url`의 파일명 부분과 일치했다. 라벨 문자열은
`data[0].attributes[0].name`에 있다. FS 1,000종, CROWD 2종이며 모든 FS에 2개 영상명이
연결된다. 동일 FS의 문자열이 다른 그룹은 3개다. 공백/NFC로 통합하지 않으며
영상별 원문을 그대로 보존하고 검토 대상 ID는 메모리에만 제공한다.
동일 CROWD ID가 실세계에서 동일인임은 별도 명세 확인이 필요하다.

| 신규 API | 입력 | 출력 / 계약 |
|---|---|---|
| `label_preparation.decode_label_document` | UTF-8 bytes | dict. 기존 중복 키·인코딩 검증을 공통화. `decode_morphemes` 계약은 동일 |
| `expected_video_name` | ZIP 내부 JSON 경로 | 엄격한 ASCII 파일명·폴더 ID 일치 검증 후 mp4 basename |
| `parse_crowd_label` | 내부 경로, JSON bytes | `CrowdLabel(fs_id, crowd_id, target)`. URL은 HTTP(S)만 허용하며 접속하지 않음 |
| `find_conflicts` | CrowdLabel iterable | 원문이 다른 FS ID tuple. 비공개 로컬 검토용이며 공개 출력 금지 |
| `read_crowd_zip` | ZIP 경로/zipfile이 지원하는 파일 객체, 선택적 크기 제한 | `CrowdReport`. 정상 레코드·비공개 검토 ID·고정 오류 코드별 개수 |
| `CrowdReport.summary` | 없음 | 정수 집계 dict. 공개 가능한 출력은 이 집계로 제한 |
| `compare_video_names` | CrowdReport, 선택적 영상 basename iterable | `VideoCheck`: 예상/제공/대응/누락/추가/중복 개수 및 상태 |

`Target.signer_id`는 `CROWD` namespace와 ID를 연결한 로컬 그룹 키다. 실제 개인 정보가
아니며 다른 데이터셋과 동일인 여부를 추정하지 않는다. `Target.text`를 정규화·수정하지
않는다. `duration`, `exportedOn`은 매핑에 사용하지 않으며 메타데이터 전체의 유효성을
검사하는 기능은 아니다. 도로명 접미사 후보는 `로` 또는 `길`로 끝나는 문자열이며
확정된 클래스가 아니다. 한글 통계는 완성형 음절, 숫자 통계는 ASCII 0~9 포함 기준이다.

파일별 오류는 코드별로 집계하고 해당 레코드를 제외한다. 중복 ZIP JSON명은
첫/마지막 항목을 선택하지 않고 중복된 모든 항목을 제외한다. `json_count`에는 오류
파일도 포함되고, FS/문자 통계는 정상 레코드만 대상으로 한다. 비JSON 파일과 디렉터리는
파싱하지 않는다. 보고서에 오류가 있으면 영상 목록이 정상 부분집합과 같아도 `mismatch`다.

기본 제한은 JSON당 1MiB, 엔트리 100,000개, 선언된 전체 JSON 비압축 크기 128,000,000바이트다.
멤버를 순서대로 제한 읽기하며 ZIP 전체를 해제하지 않는다. 엔트리/전체 크기 제한 또는
ZIP 열기 실패는 고정 코드 예외로 전체 처리를 중단한다. 중앙 디렉터리는 Python zipfile이
먼저 메모리에 올리므로 임의의 초대형 비신뢰 ZIP에 대한 완전한 메모리 격리는 아니다.
정상 레코드·중복/불일치 집계는 O(파일 수 + 문자열 총량) 메모리를 사용한다.

영상 목록 `None`은 `not_provided`, 빈 목록은 제공된 빈 목록으로 취급한다. `matched`도
**공급된 이름 목록의 일치**만 뜻하며 파일 존재·영상 내용 검증을 뜻하지 않는다.
중복 개수는 첫 항목 이후의 추가 출현 수, 추가/누락 개수는 고유 이름 수다.
비교에는 경로가 아닌 엄격한 CROWD mp4 basename을 공급한다.
영상 목록을 자동 생성해 다시 비교한 결과를 실제 영상 검증으로 표시해서는 안 된다.

기존 `build_dataset.py`는 WORD/REAL 전용이므로 수정하거나 CROWD에 적용하지 않았다.
`landmarks.py`도 변경하지 않았다. 신규 모듈의 객체·정답·검토 ID를 직렬화하면
비공개 파생 자료가 되므로 공개 저장소·로그로 내보내지 않는다. CLI/출력 파일 생성은 없다.

검증 결과는 [실행 기록](label_preparation_validation.md)의 CROWD 절 참조.
다음 단계는 불일치 3그룹의 비공개 검토, 실제 영상 목록 대조, 공식 수행자/도로명 분류
정의·37종 목록·숫자 정책 확보, PR #11 main 반영 후 별도 자모 토큰화 연결이다.

## 결론과 근거 범위

아래는 CROWD ZIP 확보 전의 최초 조사 기록이며, 현재 확인 범위는 위 절을 우선한다.

이 변경은 **확인된 형태소 JSON 부분집합 파서와 내부 정답 표현**을 제공한다.
도로명 지문자 전용 스키마·한글 도로명 매핑은 확인되지 않았다. 도로명 학습용
정답을 실제로 생성·검증했다는 의미가 아니다. PR #11 코드 조회·복사·병합 없음.

공식 출처: [AI-Hub 수어 영상, 데이터셋 103](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103).
공개 설명은 형태소 JSON 및 시작·종료 시각, 표제어 필드를 설명한다.
이는 자모별 정렬이나 도로명 표기 규칙을 보장하지 않는다. 공식 설명서·활용가이드
링크도 열었으나 본문이 반환되지 않아 내용을 검증하지 못했다. 데이터 다운로드 없음.

## 저장소 및 로컬 자료 확인

- 작업 브랜치 `feat/2-label-preparation`; 기준 로컬 main `c627cf7`.
- `src/landmarks.py`: 상체 11점·양손 42점, 전체 영상 목표 15fps 추출 지원.
- `src/build_dataset.py`: WORD 정면 영상명에서 단어/수행자 번호 추출,
  `morpheme/{signer}/{base}_morpheme.json` 조회. 현재 첫 구간·첫 표제어만 사용.
- `src/select_classes.py`: WORD 라벨을 사용한 단어 분류용 클래스 선정.
  숫자 접미사 정리 로직은 단어 데이터 전용이므로 도로명에는 재사용하지 않는다.
- `src/dataset.py`, `configs/split-6fold.json`: 수행자 단위 단어 분류 분할.
  이 수행자 번호/분할을 지문자 데이터에 그대로 적용하지 않는다.
- `tools/manifest.py`, `manifests/`: 기존 단어 자료의 경로·해시·목록 관리.
- `tests/`: 합성 전처리·추출·클래스 선정·실시간·학습 테스트.
- README 기본 외부 자료 경로 `D:/sign-language-data`는 현재 존재하지 않는다.
- git 제외된 로컬 `src/youtube_test/labels/수어 영상/1.Training/[라벨]01_real_word_morpheme.zip`
  발견. ZIP 디렉터리만 집계하면 JSON 240,000개, WORD 이름 패턴 240,000개,
  FINGER/ROAD 이름 패턴 0개. 이것만으로 ZIP에 도로명 정보가 없다고 단정하지 않는다.
  JSON 한 개의 **키만** 확인: `metaData,data` → `start,end,attributes` → `name`.
  원문·수행자 값·영상은 출력/복사/추출하지 않았다. 도로명 지문자 실제 자료 검증은 미실행.

## 확인된 필드와 미확인 사항

| 항목 | 확인 결과 / 처리 |
|---|---|
| `data` | 기존 코드 및 로컬 단어 자료에서 배열 확인. 파서는 빈 배열 거부 |
| `start`, `end` | 공식 표에서 형태소 구간 시작/종료 시각. 숫자·십진 문자열 초 단위로 해석하는 기존 코드와 공식 예시에 대응 |
| `attributes[].name` | 공식 정의는 형태소 표제어. 도로명 완성 문자열이라고 단정 불가 |
| `metaData` | 로컬 최상위 키 존재 확인. URL 등 메타데이터로 영상/수행자를 자동 추론하지 않음 |
| 한글 도로명 얻기 | 도로명 ID→표기 매핑표, 전용 라벨 또는 파일명 규칙 필요. 현재 미확인 |
| 영상당 정답 하나 | 도로명 자료에서 미확인. 여러 형태소를 한 정답으로 결합하지 않음 |
| 자모 단위 시각 | 제공 여부 미확인. 형태소 시각을 자모 시각으로 사용하지 않음 |
| 한글 자모 37종 | 정확한 구성·출처 미확인. 37개 토큰이나 blank 포함 여부를 임의 정의하지 않음 |
| 숫자 포함 도로명 | 원문 그대로 보존. 삭제·한글 읽기 변환·자동 제외 없음. CTC 토큰화 정책 결정 전 학습 투입 금지 |
| 같은 정답 여러 영상 | 도로명 실제 빈도 미확인. 내부 설계는 서로 다른 영상의 같은 문자열 허용 |
| 영상-라벨 대응 | WORD에서 기존 코드의 base 이름 대응만 확인. 도로명 전용 대응 규칙은 미확인 |
| 수행자/분할 | WORD 코드의 REAL 번호만 확인. 지문자 번호 체계·공식 분할·동시점 영상 그룹은 미확인 |

## 내부 표현과 API

파일: `src/label_preparation.py`. Python 표준 라이브러리만 사용한다.

| API | 입력 | 출력 / 검증 |
|---|---|---|
| `Segment` | 시작 초, 종료 초, 문자열 | 불변 형태소 레코드. 직접 생성 시 검증하지 않으므로 파서 사용 권장 |
| `Target` | 영상 ID, 수행자 그룹 ID, 전체 문자열, 선택 구간 tuple | 불변 내부 정답. ID는 검증된 어댑터가 공급. 직접 생성 후 `validate_targets` 필수 |
| `parse_morphemes(document)` | dict, `data` 배열의 형태소 객체 | 모든 구간의 `tuple[Segment, ...]`; 첫 항목만 선택하지 않음 |
| `decode_morphemes(raw)` | UTF-8 bytes, 선택적 BOM | 위와 같은 tuple. 잘못된 인코딩·JSON·중복 JSON 키 거부 |
| `make_target(video_id=, signer_id=, text=)` | 검증된 외부 대응 관계, 문자열 세 개 | 구간 없는 `Target`. 도로명 출처 검증 함수가 아님 |
| `target_from_morphemes(document, video_id=, signer_id=)` | JSON dict 및 명시적 ID | 정확히 한 형태소일 때만 구간 있는 `Target`. 전체 전사인지 호출자가 별도 확인 |
| `validate_targets(targets)` | `Iterable[Target]` | tuple. 영상 ID 중복은 동일 행도 거부. 같은 text의 다른 영상은 허용 |
| `validate_signer_splits(splits)` | 분할명 → `Iterable[Target]` dict | 성공 시 None. 분할 간 영상/수행자 중복 거부. 분할 내 같은 수행자는 허용 |
| `LabelError` | 잘못된 필드/구간/중복 | ValueError 하위 예외. 메시지는 값 없이 필드 위치·오류 종류만 포함 |

지원 범위: 형태소당 이름 속성 하나, 양의 길이 구간, 시간순·겹침 없는 구간.
이는 안전하게 거부하기 위한 **이 파서의 제한**이며 실제 데이터 전체의 보편 스키마가 아니다.
미사용 메타 필드는 무시한다. Unicode 정규화·자모 분리·숫자 치환은 수행하지 않는다.
빈/공백뿐인 문자열, 제어/형식 문자·대체문자·surrogate를 거부하며 나머지 공백·숫자·기호는 보존한다.
긴 문자열을 임의 자르지 않는다. 이 API는 메모리에 있는 자료만 처리하므로
향후 파일/서비스 입력 어댑터에는 크기 제한을 별도 설계해야 한다.

합성 예시 (AI-Hub 실제 도로명 스키마 예시가 아님):

```python
from label_preparation import make_target, validate_targets, validate_signer_splits

train = validate_targets([
    make_target(video_id="synthetic-1", signer_id="group-a", text="가상누리12길"),
])
validate_signer_splits({"train": train, "validation": []})
```

CTC 토큰은 아직 없다. PR #11 main 반영 후 검토된 분리 함수를 `Target.text`에
명시적으로 적용하는 별도 단계로 연결한다. 현재 단어 분류 `dataset.py`로 직접 투입하지 않는다.

## 테스트와 별도 리뷰

신규 `tests/test_label_preparation.py`의 모든 fixture는 합성 데이터이다.
정상·빈 입력·누락·자료형·인코딩·중복 JSON 키/영상·기호/숫자·긴 문자열·시간 경계·
다중 구간·수행자 누수·오류 메시지 노출을 검사한다.

구현 완료 후 작성자가 별도 읽기 리뷰를 수행했다(외부 리뷰어나 별도 에이전트 리뷰 아님).
발견/수정:

1. `Target` 직접 생성으로 잘못된 구간이 배치 검증을 통과할 수 있음 → 구간 자료형,
   개수, 시간, 전체 text와의 일치 검사 추가. 회귀 테스트 추가.
2. Python JSON 정수 길이 제한에서 일반 ValueError가 외부로 나올 수 있음 →
   값 없는 LabelError로 변환. 5,000자리 합성 정수 회귀 테스트 추가.

실행 결과와 명령은 `label_preparation_validation.md`에 기록한다.
단위 테스트 통과는 AI-Hub 도로명 데이터와의 일치를 증명하지 않는다.

## 데이터 보호와 다음 작업

원본·단순 가공 영상/좌표/라벨/개인정보는 커밋하지 않는다. 기존 ZIP은 그대로 두었다.
로컬 정답/라벨은 git 제외 `data/`, `private_labels/`, `generated_targets/`에 보관한다.
오류 메시지·repr 억제는 보조 수단이며 Target 직렬화 결과도 공개하면 안 된다.

실제 학습 준비 전 필요한 순서:

1. 권한 있는 로컬 도로명 자료와 매핑표/명세에서 파일명, 한글명, 수행자, 시점·반복 그룹 검증.
2. 영상-라벨 결합의 누락·다대일·중복 집계 및 표본 전사 대조(원문은 비공개).
3. 37종 목록·숫자/기호/공백 정책 확정. 지원하지 않는 문자를 조용히 지우지 않기.
4. PR #11 반영 후 자모 분리 재사용, token↔ID, CTC blank, 반복 토큰 길이 조건 검증.
5. 원본 수행자/동일 촬영의 모든 시점을 같은 분할에 유지. 미등장 도로명 평가가 필요하면
   정답 문자열 그룹도 별도로 분리. seed·분할 기준·수행자 namespace 기록.
6. 15fps 특징 길이와 CTC 출력 길이, 결측 프레임, 빈 영상 및 정답 길이 검증.

현재 PR은 분석/기반 파서의 초안으로 검토 가능하다. 도로명 정답 생성 완료 PR로는
제출 준비가 끝나지 않았다. 실제 도로명 스키마 확인과 팀 리뷰가 남아 있다.
PR 본문은 이후 사용자가 채팅으로 제공한 템플릿 구조에 맞춰 작성했다.
