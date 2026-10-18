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

## 기록

일지·시행착오·제약 표·그림 목록은 task-management-repository의 `logs/sign-language/`에 있다.
