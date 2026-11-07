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
# 비수지 신호용 얼굴 메시 점(468점 중 33점). 53점과 따로 저장해 기존 점 순서를 바꾸지 않는다.
FACE_IDX = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336,    # 눈썹 오른쪽 5 · 왼쪽 5
            33, 133, 159, 145, 263, 362, 386, 374,             # 눈 오른쪽(꼬리 2, 위·아래 눈꺼풀) · 왼쪽
            61, 291, 0, 17, 37, 267, 84, 314, 13, 14, 78, 308,  # 입 바깥 8 · 안쪽 4
            1, 152, 10]                                        # 코끝, 턱, 이마(머리 방향)


def _frame(res):
    """반환: xy (53,2), mask (53,), z (53,). z 는 MediaPipe 의 깊이 값(손은 손목 기준, x 와 비슷한 화면 너비 단위)."""
    xy = np.zeros((N_POINTS, 2), np.float32)
    z = np.zeros(N_POINTS, np.float32)
    mask = np.zeros(N_POINTS, bool)
    if res.pose_landmarks:
        pose = res.pose_landmarks.landmark
        for k, idx in enumerate(POSE_IDX):
            xy[k] = (pose[idx].x, pose[idx].y)
            z[k] = pose[idx].z
            mask[k] = pose[idx].visibility >= MIN_VISIBILITY
    for start, hand in ((11, res.left_hand_landmarks), (32, res.right_hand_landmarks)):
        if hand:
            xy[start:start + 21] = [(p.x, p.y) for p in hand.landmark]
            z[start:start + 21] = [p.z for p in hand.landmark]
            mask[start:start + 21] = True
    return xy, mask, z


def _face(res):
    """반환: face (33,2), face_mask (33,). 얼굴이 안 잡힌 프레임은 0 과 False."""
    face = np.zeros((len(FACE_IDX), 2), np.float32)
    if not res.face_landmarks:
        return face, np.zeros(len(FACE_IDX), bool)
    lm = res.face_landmarks.landmark
    face[:] = [(lm[i].x, lm[i].y) for i in FACE_IDX]
    return face, np.ones(len(FACE_IDX), bool)


TARGET_ASPECT = 16 / 9   # 학습 데이터(1920x1080)의 화면비


def fit_16x9(bgr):
    """프레임을 16:9 틀 가운데에 넣고 남는 곳을 검은색으로 채운다. 이미 16:9면 그대로 돌려준다.
    MediaPipe 좌표는 x 를 화면 너비, y 를 화면 높이로 나눈 값이라 화면비가 다르면 손 모양이 다르게 찌그러진다(C-16).
    세로 핸드폰(9:16)·4:3 웹캠 등 어떤 비율이 들어와도 학습 데이터와 같은 좌표 단위가 되게 한다."""
    h, w = bgr.shape[:2]
    if abs(w / h - TARGET_ASPECT) < 1e-3:
        return bgr
    if w / h < TARGET_ASPECT:                      # 세로로 긴 화면: 좌우에 여백
        new_w = int(round(h * TARGET_ASPECT))
        pad = new_w - w
        return cv2.copyMakeBorder(bgr, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    new_h = int(round(w / TARGET_ASPECT))          # 가로로 더 긴 화면: 위아래에 여백
    pad = new_h - h
    return cv2.copyMakeBorder(bgr, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def center_crop(bgr, aspect):
    """가운데를 aspect(너비/높이) 비율로 잘라낸다. 다른 카메라 화면비를 흉내 내는 검증용이다(예: 9/16 = 핸드폰 세로)."""
    h, w = bgr.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        return bgr[:, x0:x0 + nw]
    nh = int(round(w / aspect)); y0 = (h - nh) // 2
    return bgr[y0:y0 + nh]


def process_frame(holistic, bgr, with_face=False):
    """MediaPipe 에 넣는 단일 경로: 16:9 틀에 넣고 → RGB 로 바꿔 → 추출. 학습 데이터 추출과 실시간·영상 파일 입력이 모두 이 함수를 쓴다.
    반환: (xy, mask, z, 처리한 프레임의 (너비, 높이)). with_face 이면 뒤에 (face, face_mask) 를 더한다."""
    fitted = fit_16x9(bgr)
    res = holistic.process(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB))
    xy, mask, z = _frame(res)
    wh = (fitted.shape[1], fitted.shape[0])
    if with_face:
        return (xy, mask, z, wh) + _face(res)
    return xy, mask, z, wh


def select_frames(n_frames, src_fps, target_fps, phase=0):
    """원본 n_frames 장 중 target_fps 카메라가 찍었을 프레임 번호. 원본 fps 가 목표의 정수배가 아니어도(29.97 등) 된다.
    phase 는 원본 프레임 단위의 시작 어긋남(0 ≤ phase < 원본/목표 비율). 실제 카메라 박자와 수어 시작이 무관한 것을 흉내 낸다."""
    step = src_fps / target_fps
    if step < 1:
        raise ValueError("목표 fps(%.2f)가 원본 fps(%.2f)보다 크다" % (target_fps, src_fps))
    idx, k = [], 0
    while True:
        i = int(round(phase + k * step))
        if i >= n_frames:
            return idx
        idx.append(i)
        k += 1


def extract_clip(video_path, start_s, end_s, margin_s=0.5, complexity=1, with_z=False, crop_aspect=None):
    """[start_s - margin_s, end_s + margin_s] 구간의 프레임에서 좌표를 뽑는다.
    반환: (xy (T,53,2), mask (T,53), fps). with_z 이면 (xy, mask, fps, z (T,53), (처리한 화면 너비, 높이)). 읽은 프레임이 없으면 None.
    crop_aspect 를 주면 프레임을 그 비율로 가운데 잘라낸 뒤 처리한다(다른 카메라 흉내, 검증용).
    클립마다 새 Holistic 을 만들어 앞 클립의 추적 상태가 섞이지 않게 한다."""
    cap = cv2.VideoCapture(str(video_path).replace("\\", "/"))
    wh = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t0 = max(0.0, start_s - margin_s)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
        n = int(round((end_s + margin_s - t0) * fps)) + 1
        xys, masks, zs = [], [], []
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity) as holistic:
            for _ in range(n):
                ok, bgr = cap.read()
                if not ok:
                    break
                if crop_aspect:
                    bgr = center_crop(bgr, crop_aspect)
                xy, mask, z, wh = process_frame(holistic, bgr)
                xys.append(xy)
                masks.append(mask)
                zs.append(z)
    finally:
        cap.release()
    if not xys:
        return None
    if with_z:
        return np.stack(xys), np.stack(masks), fps, np.stack(zs), wh
    return np.stack(xys), np.stack(masks), fps


def extract_video(video_path, target_fps=15.0, phase_seed=0, complexity=1):
    """영상 전체 시간에서 target_fps 카메라가 찍었을 프레임만 MediaPipe 에 넣는다(나머지는 디코딩만 하고 건너뜀).
    추적도 그 간격으로 이어지므로 실제 target_fps 카메라와 같은 조건이 된다.
    위상은 phase_seed % (원본/목표 비율, 반올림) 로 정한다(30→15fps 면 짝·홀 프레임 중 하나).
    반환: dict(xy (T,53,2), mask, z (T,53), face (T,33,2), face_mask, t (T,) 원본 기준 초, src_fps, phase, wh). 읽은 프레임이 없으면 None."""
    cap = cv2.VideoCapture(str(video_path).replace("\\", "/"))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        phase = phase_seed % max(1, int(round(src_fps / target_fps)))
        keep = set(select_frames(n, src_fps, target_fps, phase))
        out = {k: [] for k in ("xy", "mask", "z", "face", "face_mask", "t")}
        wh = None
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity) as holistic:
            for i in range(n):
                if i not in keep:
                    if not cap.grab():
                        break
                    continue
                ok, bgr = cap.read()
                if not ok:
                    break
                xy, mask, z, wh, face, fmask = process_frame(holistic, bgr, with_face=True)
                for k, v in zip(("xy", "mask", "z", "face", "face_mask", "t"), (xy, mask, z, face, fmask, i / src_fps)):
                    out[k].append(v)
    finally:
        cap.release()
    if not out["xy"]:
        return None
    res = {k: np.stack(v) for k, v in out.items()}
    res["t"] = res["t"].astype(np.float32)
    res.update(src_fps=float(src_fps), phase=phase, wh=wh)
    return res
