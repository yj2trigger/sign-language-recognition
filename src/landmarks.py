"""영상 구간에서 상체 11점 + 양손 42점(총 53점)의 x, y 좌표와 검출 마스크를 뽑는다.

mediapipe==0.10.21 의 구형 API(mp.solutions.holistic)를 쓴다. 신청서의 사전 조사(속도 19.5~24fps,
제공 키포인트와의 손 좌표 오차 중앙값 0.08~0.10)가 이 버전으로 측정되었고 model_complexity 로
속도·정확도를 조절할 수 있어서다. MediaPipe 의존은 이 파일에만 둔다(신형 API로 옮길 때 이 파일만 바꾼다).

좌표는 이미지 정규화 좌표(0~1)다. 좌우 손은 MediaPipe 결과의 left/right를 그대로 쓰므로
영상을 좌우 반전하지 않은 입력을 전제로 한다(웹캠도 반전 전 프레임을 넣을 것).
점 순서는 preprocess.py 와 같다: 상체 11 + 왼손 21 + 오른손 21.
"""
import cv2
import mediapipe as mp
import numpy as np

POSE_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]   # 코, 눈2, 귀2, 어깨2, 팔꿈치2, 손목2
N_POINTS = 11 + 21 + 21
MIN_VISIBILITY = 0.5


def _frame(res):
    xy = np.zeros((N_POINTS, 2), np.float32)
    mask = np.zeros(N_POINTS, bool)
    if res.pose_landmarks:
        pose = res.pose_landmarks.landmark
        for k, idx in enumerate(POSE_IDX):
            xy[k] = (pose[idx].x, pose[idx].y)
            mask[k] = pose[idx].visibility >= MIN_VISIBILITY
    for start, hand in ((11, res.left_hand_landmarks), (32, res.right_hand_landmarks)):
        if hand:
            xy[start:start + 21] = [(p.x, p.y) for p in hand.landmark]
            mask[start:start + 21] = True
    return xy, mask


def extract_clip(video_path, start_s, end_s, margin_s=0.5, complexity=1):
    """[start_s - margin_s, end_s + margin_s] 구간의 프레임에서 좌표를 뽑는다.
    반환: (xy (T,53,2), mask (T,53), fps). 읽은 프레임이 없으면 None.
    클립마다 새 Holistic 을 만들어 앞 클립의 추적 상태가 섞이지 않게 한다."""
    cap = cv2.VideoCapture(str(video_path).replace("\\", "/"))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t0 = max(0.0, start_s - margin_s)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
        n = int(round((end_s + margin_s - t0) * fps)) + 1
        xys, masks = [], []
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity) as holistic:
            for _ in range(n):
                ok, bgr = cap.read()
                if not ok:
                    break
                xy, mask = _frame(holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                xys.append(xy)
                masks.append(mask)
    finally:
        cap.release()
    if not xys:
        return None
    return np.stack(xys), np.stack(masks), fps
