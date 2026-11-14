# PR 제목

[test] #2 수행자 근거·연속 수형·공개 API 최종 검증

# PR 본문

## Why

기존 모델과 특징은 존재하지만 영상 ID 분리만으로 수행자 독립 평가를 주장할 수 없고,
일부 RGB 접촉표만으로 실제 철자를 확정할 수도 없다. 기존 데이터와 모델을 보존하면서
평가 자격·의미 판독 한계·실제 API 계약을 재현 가능한 검사로 남긴다. Refs #2.

## What / How

- 원본 형태소 2,000개와 실제 Train123/Val12/Test8, ZIP 경로, URL 패턴을 읽기 전용 조사한다.
  URL 후보를 사람으로 확정하지 않는다. 공식 수행자 매핑이 없어 독립 CER는 미측정이다.
- 기존 블록의 3영상·384관측·18확인시점을 연속 분석하고 각도·거리·투영 정보를 기록한다.
  공식 교육 도판과 비교했지만 완전한 독립 전사가 없어 3개 모두 UNCERTAIN이다.
- 기존 checkpoint와 실제 특징 3개에서 public CPU API 및 동일 CUDA reader를 호출한다.
  문자열 3/3 일치, 계약 사례 12개 통과다. 예측을 라벨 의미 검증 정답으로 쓰지 않는다.
- 원본 ZIP·블록 파일·기존 모델/특징·미추적 파일의 before/after 해시를 비교한다.
  기존 코드·라벨·분할·브랜치·PR을 변경하지 않고 재학습·새 영상 전처리를 하지 않는다.

## Where / Lines

비교 기준은 Base의 `80b006beea4671e7b4b6d8a38ab50e4a52c458f3`다.
모든 파일이 신규이므로 이전 줄 범위는 없음이다. 기존 production 파일 변경은 0개다.

| 파일 | 이전 → 이후 줄 | 구체적 변경 |
|---|---|---|
| `tools/validate_issue2_submission.py` | 없음 → 1–225 | 수행자 근거, 실제 API 계약 검사, 보호 해시 manifest |
| `tools/inspect_issue2_semantics.py` | 없음 → 1–120 | 연속 관절·18시점 관찰, GIF 누적 시간 보정·재읽기 |
| `tests/test_issue2_submission_validation.py` | 없음 → 1–116 | 후보 ID 오인 차단, 경계·극단·GIF·API 판정·CER 회귀 |
| `docs/issue2_submission_validation.md` | 없음 → 1–223 | 실제 결과, 불가능한 평가 조건, 재현·통합 접점 |
| `docs/issue2_submission_validation_pr.md` | 없음 → 1–74 | 이 PR 초안 |

## Tests

기존 Conda `sign-language`에서 실행했다. 새 브랜치 전체 **191 passed, 2 warnings**,
기존 PR 테스트를 포함한 detached 통합 검사 **554 passed, 2 warnings, 254 subtests passed**다.
production Python 48개는 두 검사 트리에서 바이트 일치한다. 실제 Merge를 수행한 결과는 아니다.

```powershell
$py='D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
$env:TEMP='D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2'
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
& $py -m pytest -q -p no:cacheprovider --basetemp 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\final-validation\regression-final2'
# 기존 PR 통합 테스트 복사본에서 실행:
& $py -c "import torch,pytest;torch.set_num_threads(2);raise SystemExit(pytest.main())" -q -p no:cacheprovider --basetemp 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2\final-validation\integrated-tests'
```

실제 API 호출·데이터 검사 명령과 실행 디렉터리는 검증 보고서에 있다.
독립 리뷰의 GIF 양자화·보호 범위 지적을 수정하고 재검증했다.
원본 ZIP 3개, 블록 결과 11,784파일, 선정 산출물 149개, 미추적 35개,
추가 발견 checkpoint 346개 및 확장 학습 JSON 15개가 해시 일치했다.

## Dependencies

- **Base:** `feat/2-crowd-block-validation` (`80b006b...`)
- **Head:** `feat/2-final-submission-validation`
- 선행 블록 도구 PR은 아직 생성하지 않았다. Base가 main에 병합됐다는 뜻이 아니다.
- 기존 #11/#12 → #13은 런타임 의존과 Git ancestry를 구분하며, 실제 #13 Base는 #12다.
- #18 → #19이며 #17과 #18 사이에 불필요한 선행 관계를 추가하지 않는다.
- 기존 16개 분할 브랜치·기존 PR을 수정하지 않는다. 이전 통합용 5개 로컬 브랜치도 보존한다.

## Limitations

수행자 독립 CER, 완전한 영상·라벨 독립 전사, 실사용 인식 품질은 미완료다.
기존 Test진단 CER는 수행자 독립 성능이 아니다. strict 출력에는 미조합 자모가 남을 수 있다.
실시간 루프는 수정하지 않았다. 실제 문자열·라벨·원본 영상·좌표·GIF·checkpoint는
이 PR에 포함하지 않는다. 연구용 검증 구현의 제출이며 Issue #2 전체 완료 선언은 아니다.

Refs #2
