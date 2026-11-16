# Issue #2 — 팀원 2: 지문자 CTC 학습 환경 인계

## 범위
이 PR은 **다른 담당자가 지문자 모델을 학습할 수 있도록 하는 코드 기반**만 전달합니다. 실제 AI-Hub 영상 다운로드·추출, 실데이터 학습 실행, CER 성능 개선 실험, 학습에 없는 수행자 평가, 실시간 앱 통합은 인계받는 담당자의 후속 작업이며 이 PR의 완료 주장에 포함되지 않습니다.

## 기존 PR에서 선별한 코드
- #11: 한글 자모 분해·조합 및 단위 테스트
- #12: 읽기 전용 CROWD 라벨 매핑·파싱 및 테스트 (기존 README/.gitignore 수정은 가져오지 않음)
- #13: CTC 토큰/학습 정답 생성 및 테스트 (실제 데이터 감사 도구는 제외)
- #16: 관절 입력 검증, 결측·정규화, 시계열 특징, 가변 길이 배치 및 테스트
- #17: BiGRU CTC 모델, 학습 루프, 체크포인트 및 테스트 (별도 추론 API는 제외)

기존 PR 내용 중 추출·실험·성능 개선 결과 파일을 이 브랜치에 복사하지 않았습니다. 기존 PR #11~#17의 소스 파일은 그대로 두었습니다. **이 통합 PR과 원본 PR들을 각각 병합하면 중복되므로, 팀이 어느 경로로 병합할지 결정해야 합니다.**

## 데이터·입력 계약
- 기준 프레임: 기존 추출 담당자가 정하는 15 fps 특징
- raw 관절 배열: `xy (T,53,2)`, `mask (T,53)`
- 좌표 검증→선택적 보간·정규화→시계열 특징→CTC 배치
- 정답: 검증된 문자열→외부에서 지정한 토큰 목록→CTC target
- 실제 지문자 37종 목록, 수행자 식별 및 정답 의미 검증은 별도 검증 필요

## 로컬 확인
전용 Conda 환경 `sign-language`에서 NumPy, pytest, PyTorch 등을 설치한 뒤 다음 명령을 실행합니다.

```powershell
& 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe' -m pytest -q tests/test_jamo.py tests/test_label_preparation.py tests/test_crowd_labels.py tests/test_ctc_targets.py tests/test_fingerspelling_tokens.py tests/test_sequence_validation.py tests/test_sequence_preprocessing.py tests/test_sequence_features.py tests/test_ctc_batching.py tests/test_ctc_model.py tests/test_ctc_training.py tests/test_ctc_checkpoint.py
```

**주의:** 위 명령은 검증 방법이며, 본 통합 브랜치에서 실제 실행된 결과가 아닙니다. 원본 PR별 통과 기록은 원본 PR 본문을 참조하십시오. 본 통합 PR의 의존성·상호 호환성과 전체 회귀는 리뷰어가 실제 실행으로 확인해야 합니다.

## 변경·보존 원칙
- `main` 기반 새 브랜치. 원본 코드 수정·덮어쓰기·재작성 없음
- 기존 일반 수어 모델, `src/realtime.py`, 기존 원본 데이터, GPU 체크포인트 변경 없음
- AI 사용: 기존 PR 코드의 선별·의존성 분석·인계 문서 및 PR 작성에 ChatGPT 사용
- 이 PR은 Issue #2 전체 완료가 아닌 **학습 환경 인계**만 다룹니다. `Refs #2`
