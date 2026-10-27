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

원본 영상과 대용량 좌표 배열은 레포에 올리지 않고 로컬 `D:\sign-language-data`에 둔다. AI-Hub 이용 조건상 원본 데이터와 원본을 단순 가공한 데이터(추출한 좌표 배열 포함)는 공유·재배포할 수 없기 때문이다. 이 레포에는 코드, 데이터 목록·해시, 실험 결과(지표·예측 번호)만 둔다. 데이터를 쓰려면 각자 AI-Hub에서 내려받아 같은 코드로 추출한다.
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

## 데이터 출처

이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "수어 영상"과 "재난 안전 정보 전달을 위한 수어영상 데이터"를 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다.

This research used datasets from The Open AI Dataset Project (AI-Hub, S. Korea). All data information can be accessed through 'AI-Hub (www.aihub.or.kr)'.

## 라이선스

코드는 MIT 라이선스(`LICENSE`)를 따른다. AI-Hub 데이터는 이 라이선스에 포함되지 않으며 AI-Hub 이용 조건을 따른다.

## 기록

일지·시행착오·제약 표·그림 목록은 task-management-repository의 `logs/sign-language/`에 있다.
