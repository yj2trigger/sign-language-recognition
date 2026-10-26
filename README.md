# sign-language-recognition

랜드마크 기반 딥러닝 모델을 활용한 실시간 수어 번역 시스템 (SW·AI융합연구개발 일반과제, 2026.10~2026.11)

MediaPipe Holistic으로 상체 11점과 양손 42점(총 53점)을 추출하고, LSTM/GRU와 1D-CNN(TCN)으로 단어를 인식한다.

## 환경

Python 3.12, CPU 전용. 가상환경은 레포 안 `.venv`(git 제외).

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 데이터 위치

원본 영상과 대용량 좌표 배열은 레포에 올리지 않고 로컬 `D:\sign-language-data`에 둔다. (AI-Hub 이용 조건: 팀 안에서만 사용, GitHub 파일당 100MB 한도)
레포에서 관리하는 것: 클래스 목록, 수행자 분할 파일, 데이터 목록과 해시, 추출·전처리 스크립트·설정, 실험 결과.

| 파일 | 내용 |
|---|---|
| `manifests/sources.csv` | 받은 AI-Hub 원본 zip(파일 키, 경로, 담긴 수행자·단어 범위, 크기, 앞 1MB SHA-256) |
| `manifests/features.csv` | 추출한 좌표 클립 전부(단어 번호, 수행자 번호, 프레임 수, 파일 SHA-256) |
| `manifests/models.csv` | 저장한 모델(SHA-256, 만든 커밋, 학습 수행자) |
| `configs/split-6fold.json` | 수행자 6폴드 분할(해시 `85477280ac`, 모든 실험 공통) |
| `configs/classes-*.json` | 클래스(단어 번호) 목록 |
| `runs/` | 실험 결과 json |

내 데이터가 목록과 같은지 확인: `python tools/manifest.py check` (약 6분, 마지막 줄 "결과: 모두 일치").
데이터를 새로 추출·학습한 뒤에는 `python tools/manifest.py build` 로 목록을 다시 만들어 커밋한다.
경로가 다르면 `--data <좌표·모델 폴더> --pre <미리 받은 라벨 폴더>` 로 지정한다.

## 기록

일지·시행착오·제약 표·그림 목록은 task-management-repository의 `logs/sign-language/`에 있다.
