# 팀원 A용 지문자 인터페이스 전달

> 최신 확장 실험은 [데이터 확장 보고서](mediapipe_data_expansion.md)를 따른다.
> 연구용 후보는 `mediapipe-ctc-performance/expanded-models-01/expanded-hand-local/inference.pt`,
> SHA `0ec9ded244354bf00ede986e66d7619b6b19d6b413996f6129f6b1815329e3f5`다.
> 아래 예제의 환경변수 경로만 이 파일로 바꾸면 동일 API를 사용한다.
> `resync`는 명시적으로 설정한다. Train123/Val12/Test진단8, 자모 CER .08/.308333/.269841,
> resync 한글 CER .235154/.695652/.64, Val 완전 일치0%다.
> 수행자 독립·의미 검증은 여전히 미완료다. 아래 59개 실험 수치는 이전 기록이다.

현재 기능은 실험용이다. 수행자 독립 성능과 라벨의 실제 철자 대응은 검증되지 않았고,
완전 일치율은 0%다. 실시간 번역 시스템의 검증된 기본 인식기로 사용하지 않는다.
`src/realtime.py`는 수정하지 않았다. 호출 측의 수집·실시간 루프는 팀원 A가 연결한다.

## 입력과 호출

MediaPipe `landmarks.process_frame()` 순서의 `(T,53,2)` xy와 boolean `(T,53)` mask를 사용한다.
상체 11 / 왼손 21 / 오른손 21, 좌우 반전 없음, 목표 15fps다.
검출·패딩·시간축 정책이 다른 입력을 동일하다고 가정하지 않는다.
빈 입력·1프레임·양손 미검출은 빈 문자열, 잘못된 차원·NaN/Inf는 오류다.
모델이 없으면 명시적 오류가 발생하며 임의의 인식 결과를 만들지 않는다.

```python
# src가 import 경로에 있는 환경
from fingerspelling_runtime import FingerspellingReader

reader = FingerspellingReader(checkpoint_path, device='cpu', render_mode='resync')
text = reader(xy, mask)  # reader는 시작 시 한 번 생성
```

최종 공개 함수도 그대로 사용할 수 있다.

```powershell
$repo = 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition'
$py = 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe'
Set-Location -LiteralPath $repo
$env:PYTHONPATH = "$repo\src"
$env:FINGERSPELLING_CHECKPOINT = 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance\controlled-01\train-affine\inference.pt'
$env:FINGERSPELLING_RENDER_MODE = 'resync'
```

```python
from fingerspelling_runtime import read_fingerspelling
text = read_fingerspelling(xy, mask)
```

위 환경변수는 실행 프로세스에서만 설정한다. 기본 렌더러는 `strict`다.
`resync`는 유효 음절을 조합하고 남은 자모를 보존하므로 반환 문자열에 자모가 남을 수 있다.
정답 교정·사전 기반 복원·누락 자모 추가를 하지 않는다.
체크포인트 경로·기기·렌더러별 캐시를 사용하므로 불변 모델 경로를 지정한다.

## 모델 확보와 검증

모델 파일은 GitHub에 포함하지 않는다. 권한 있는 팀 내 비공개 공유 위치에서 받거나,
각자의 AI-Hub 이용 권한으로 데이터를 확보해 재현 명령으로 학습해야 한다.
원본 영상·라벨·좌표 배열은 공개 재배포하지 않는다. 이 문서는 전송 자체를 수행하지 않는다.

현재 실험용 후보:

- 파일: `controlled-01/train-affine/inference.pt`
- SHA-256: `0df50fc0bc8a83b5d7ae11983c2a7203661b9d8fd7252d04563d388a73eeffa8`
- 입력 424차원, hidden64/2층 BiGRU, 52클래스, blank0, 기존 자모 순서 유지.
- 특징 표준화가 첫 Linear에 합쳐져 공통 특징 함수를 그대로 사용한다.
- CPU/GPU 실제 표본 예측 일치와 기존 함수 호출을 확인했다.
- **추론 전용:** 학습을 재개할 때에는 별도의 `training/last.pt` 및 해당 affine/config를 사용한다.

```powershell
Get-FileHash -LiteralPath $env:FINGERSPELLING_CHECKPOINT -Algorithm SHA256
```

받은 파일의 SHA와 코드 버전·전처리 스키마를 확인한다. 불일치 검사를 끄지 않는다.
세 전처리 소스 파일의 LF 줄바꿈은 raw SHA 계약의 일부여서 `.gitattributes`로 고정했다.
복사 과정에서 코드 줄바꿈을 임의로 변환하지 않는다.
다른 가상환경을 새로 만들지 말고 지정 Conda Python을 사용한다.

## 측정 결과와 남은 조건

원본 특징 59개는 Train39/Validation12/Test8로 나눴으며 수행자 신원은 미확정이다.
표준화 후보의 자모 CER는 Train0.5833/Val0.7583/기존 Test 진단0.8254다.
resync 한글 CER는 Train0.7050/Val0.8043/Test진단0.84, 완전 일치율0%다.
Test 한글 CER는 기존 모델과 같아서 실사용 품질 개선을 입증한 결과가 아니다.
입력 영상의 VFR 때문에 정확히 균일한 15fps라고 주장하지 않는다.
추가 성능 검증·전문가 라벨 검토·공식 수행자 매핑이 필요하다.
실험과 재현 방법은 `docs/mediapipe_performance.md`를 따른다.
