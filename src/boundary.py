"""실시간 구간 탐지와 문장 끝 판정: BoundaryDetector.update(t, xy, mask) -> (시작, 끝, 종류) 또는 None.

종류: 'word', 'fingerspelling', 'sentence_end'. sentence_end 의 (시작, 끝)은 문장 첫 단어 시작 ~ 마지막 단어 끝이다.
입력은 landmarks.py 와 같은 53점(상체 11 + 왼손 21 + 오른손 21) 정규화 좌표와 마스크, 15fps 기준.
얼굴 33점(landmarks.FACE_IDX 순서)은 face_cue=True 일 때 update(..., face=, face_mask=) 로 넣으면 문장 끝 보조 판단에 쓴다(없어도 동작).

구간 활동 판정은 두 가지다.
  - 모델: train_boundary.py 가 학습한 인과(causal) TCN/GRU 가 최근 window 프레임 특징으로 지금 프레임의 B/I/O 확률을 낸다.
  - 규칙(모델 없을 때): 손목이 어깨선 아래 rest_rise 어깨너비보다 높거나 손끝이 move_speed 보다 빠르면 활동,
    손을 든 채 hold_speed 보다 느린 상태가 word_pause_s 이어지면 단어 사이 멈춤으로 본다.
특징은 지금과 바로 앞 프레임만 쓰는 FeatureExtractor 하나로 만든다. 학습(batch_features)과 실시간이 같은 코드를 지난다.
torch 는 모델을 쓸 때만 필요하다(규칙·지표·로그 함수는 numpy 만 쓴다).
"""

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:          # 규칙 모드·지표만 쓰는 환경(테스트 등)
    torch = None

N_POINTS, N_FACE = 53, 33
L_SH, R_SH = 5, 6                      # preprocess.py 와 같은 어깨 위치
POSE_WRIST = (9, 10)                   # 상체 11점 안의 왼쪽·오른쪽 손목
HAND_STARTS = (11, 32)
TIPS = np.array([4, 8, 12, 16, 20])    # 손 21점 안의 손가락 끝
DEFAULT_CENTER, DEFAULT_WIDTH = np.array([0.5, 0.4], np.float32), 0.2   # 어깨가 한 번도 안 보였을 때 쓰는 값(화면 가운데, 보통 어깨너비)
FEAT_NAMES = [f"{h}_{k}" for h in ("L", "R") for k in ("present", "on_change", "speed", "acc", "finger", "wrist_x", "wrist_y", "wrist_ok")]
N_FEAT = len(FEAT_NAMES)
EPS = 1e-6                     # 시간 비교 여유: 15fps 3프레임 간격이 0.2 또는 0.19999.. 로 나와 같은 길이가 다르게 판정되지 않게
SPEED_SCALE, ACC_SCALE, CLIP =5.0, 50.0, 4.0     # 특징 값 크기를 1 안팎으로(어깨너비/초 단위를 나눈다)

# 얼굴 33점 안의 위치(landmarks.FACE_IDX 순서)
F_BROWS, F_EYES, F_LIP_IN, F_CHIN, F_FOREHEAD = slice(0, 10), slice(10, 18), (26, 27), 31, 32


def sanitize(xy, mask, n=N_POINTS):
    """모양을 확인하고 NaN·inf 이거나 화면 밖으로 크게 벗어난 점(정규화 좌표 -0.5~1.5 밖)은 마스크를 끈다.
    반환: (xy float32 (n,2) 복사본, mask bool (n,)). 모양이 틀리면 ValueError."""
    xy = np.array(xy, np.float32, copy=True)
    mask = np.array(mask, bool, copy=True).reshape(-1)
    if xy.shape != (n, 2) or mask.shape != (n,):
        raise ValueError("xy 는 (%d, 2), mask 는 (%d,) 이어야 한다: %s, %s" % (n, n, xy.shape, mask.shape))
    bad = ~np.isfinite(xy).all(1) | (xy < -0.5).any(1) | (xy > 1.5).any(1)
    mask &= ~bad
    xy[~mask] = 0.0
    return xy, mask


class FeatureExtractor:
    """프레임 하나를 받아 특징 벡터(N_FEAT,)를 낸다. 손마다: 검출 여부, 검출 켜짐(+1)/꺼짐(-1), 손끝 속도,
    가속도, 손목 기준 손가락 움직임, 손목 위치(x, y)와 그 유효 여부. 좌표는 어깨 중점 원점·어깨너비 1 로 정규화한다.
    어깨가 안 보이는 프레임은 마지막으로 본 어깨를 쓴다. 손이 앞 프레임에 없었으면 속도·가속도는 0 이다(튀는 값 방지)."""

    def __init__(self, fps=15.0):
        self.fps = fps
        self.reset()

    def reset(self):
        self.prev_t = None
        self.center, self.width = None, None
        self.prev_tips = [None, None]
        self.prev_rel = [None, None]
        self.prev_speed = [0.0, 0.0]
        self.prev_present = [False, False]
        self.last = {"speed": 0.0, "finger": 0.0, "wrist_speed": 0.0, "raised": False, "present": False, "visible": False}
        self.prev_wrist = [None, None]

    def step(self, t, xy, mask):
        xy, mask = sanitize(xy, mask)
        dt = t - self.prev_t if self.prev_t is not None and t > self.prev_t else 1.0 / self.fps
        self.prev_t = t
        if mask[L_SH] and mask[R_SH]:
            w = float(np.linalg.norm(xy[L_SH] - xy[R_SH]))
            if w > 1e-6:
                self.center, self.width = (xy[L_SH] + xy[R_SH]) / 2, w
        center = DEFAULT_CENTER if self.center is None else self.center
        width = DEFAULT_WIDTH if self.width is None else self.width
        n = (xy - center) / width
        out, speeds, fingers, wspeeds, raised, any_present, any_wrist = [], [], [], [], False, False, False
        for h, (start, pw) in enumerate(zip(HAND_STARTS, POSE_WRIST)):
            present = bool(mask[start])                  # MediaPipe 는 손 21점을 함께 켜고 끈다. 손목 점으로 판정
            on_change = float(present) - float(self.prev_present[h])
            speed = finger = acc = 0.0
            tips = rel = None
            if present:
                tips = n[start + TIPS]
                rel = tips - n[start]
                if self.prev_tips[h] is not None:
                    speed = float(np.linalg.norm(tips - self.prev_tips[h], axis=1).mean() / dt)
                    finger = float(np.linalg.norm(rel - self.prev_rel[h], axis=1).mean() / dt)
                    acc = (speed - self.prev_speed[h]) / dt
            self.prev_tips[h], self.prev_rel[h] = tips, rel
            self.prev_speed[h], self.prev_present[h] = speed, present
            if mask[pw]:
                wrist, wok = n[pw], 1.0
            elif present:
                wrist, wok = n[start], 1.0
            else:
                wrist, wok = np.zeros(2, np.float32), 0.0
            wsp = 0.0
            if wok and self.prev_wrist[h] is not None:
                wsp = float(np.linalg.norm(wrist - self.prev_wrist[h]) / dt)
            self.prev_wrist[h] = wrist.copy() if wok else None
            any_present |= present
            any_wrist |= bool(wok)
            raised |= bool(wok) and float(wrist[1]) < self.rest_rise
            speeds.append(speed); fingers.append(finger); wspeeds.append(wsp)
            out += [float(present), on_change, speed / SPEED_SCALE, acc / ACC_SCALE, finger / SPEED_SCALE,
                    float(wrist[0]), float(wrist[1]), wok]
        self.last = {"speed": max(speeds), "finger": max(fingers), "wrist_speed": max(wspeeds),
                     "raised": raised, "present": any_present, "visible": any_wrist}
        return np.clip(np.array(out, np.float32), -CLIP, CLIP)

    rest_rise = 1.2      # 손목 y(어깨 기준, 어깨너비 단위)가 이보다 작으면 '손 올림'. realtime.HandRaiseSegmenter 와 같은 값


def batch_features(t, xy, mask, fps=15.0):
    """영상 하나 전체의 특징 (T, N_FEAT). 실시간과 같은 FeatureExtractor 를 프레임 순서대로 돌린다."""
    fx = FeatureExtractor(fps)
    return np.stack([fx.step(float(t[i]), xy[i], mask[i]) for i in range(len(t))]) if len(t) else np.zeros((0, N_FEAT), np.float32)


def face_signals(face, face_mask):
    """얼굴 33점에서 (눈썹 올림, 입 벌림)을 얼굴 높이(이마~턱) 단위로. 필요한 점이 안 보이면 None."""
    if face is None or face_mask is None:
        return None
    face = np.asarray(face, np.float32)
    fm = np.asarray(face_mask, bool).reshape(-1)
    if face.shape != (N_FACE, 2) or fm.shape != (N_FACE,) or not fm.all() or not np.isfinite(face).all():
        return None
    h = float(face[F_CHIN, 1] - face[F_FOREHEAD, 1])
    if h <= 1e-6:
        return None
    brow = float(face[F_EYES, 1].mean() - face[F_BROWS, 1].mean()) / h
    mouth = abs(float(face[F_LIP_IN[1], 1] - face[F_LIP_IN[0], 1])) / h
    return brow, mouth


# ---- 모델: 지금 프레임의 출력이 과거 프레임에만 의존하는 인과 TCN / 단방향 GRU. 입력 (B, T, F) -> 로짓 (B, T, 3) ----

if torch is not None:
    class _FrameNorm(nn.Module):
        """프레임마다 채널 방향으로만 정규화한다(LayerNorm). GroupNorm·BatchNorm 은 시간축까지 섞어 미래 프레임이 새어 든다."""

        def __init__(self, c):
            super().__init__()
            self.ln = nn.LayerNorm(c)

        def forward(self, x):                            # x: (B, C, T)
            return self.ln(x.transpose(1, 2)).transpose(1, 2)

    class _CausalBlock(nn.Module):
        def __init__(self, c, dilation, dropout):
            super().__init__()
            self.pad = 2 * dilation                      # 커널 3: 왼쪽에만 (3-1)*dilation 을 채운다
            self.c1 = nn.Conv1d(c, c, 3, dilation=dilation)
            self.c2 = nn.Conv1d(c, c, 3, dilation=dilation)
            self.n1, self.n2 = _FrameNorm(c), _FrameNorm(c)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.drop(F.gelu(self.n1(self.c1(F.pad(x, (self.pad, 0))))))
            return F.gelu(x + self.n2(self.c2(F.pad(h, (self.pad, 0)))))

    class CausalTCN(nn.Module):
        """팽창 1·2·4·8, 커널 3 → 받아들이는 범위 1 + 2·2·(1+2+4+8) = 61 프레임(15fps 에서 약 4초)."""

        def __init__(self, n_feat=N_FEAT, n_out=3, channels=48, dropout=0.2):
            super().__init__()
            self.inp = nn.Conv1d(n_feat, channels, 1)
            self.blocks = nn.Sequential(*[_CausalBlock(channels, d, dropout) for d in (1, 2, 4, 8)])
            self.head = nn.Conv1d(channels, n_out, 1)

        def forward(self, x):
            return self.head(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    class CausalGRU(nn.Module):
        def __init__(self, n_feat=N_FEAT, n_out=3, hidden=64, dropout=0.2):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(dropout))
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            return self.head(self.rnn(self.inp(x))[0])


def receptive_field(model):
    """인과 TCN 의 지금 프레임 출력이 보는 과거 프레임 수(지금 포함). GRU 처럼 정해진 범위가 없으면 None."""
    if torch is None or not isinstance(model, CausalTCN):
        return None
    return 1 + sum(2 * blk.pad for blk in model.blocks)     # 블록마다 커널 3 합성곱 2개, 각각 2·dilation 만큼 늘린다


def model_window(model, train_crop):
    """실시간 입력 창 크기를 학습과 같게 정한다.
    TCN: 받아들이는 범위(이보다 길면 같은 값, 짧으면 학습 때 본 과거를 잘라 먹는다). 학습 조각이 범위보다 짧으면 ValueError.
    GRU: 학습 조각 길이(학습 때 조각 첫 프레임에서 빈 상태로 시작했으므로 실시간도 같은 길이를 빈 상태에서 돌린다)."""
    rf = receptive_field(model)
    if rf is None:
        return int(train_crop)
    if train_crop < rf:
        raise ValueError("학습 조각(%d)이 TCN 받아들이는 범위(%d)보다 짧다" % (train_crop, rf))
    return rf


def build_model(arch, n_feat=N_FEAT):
    if torch is None:
        raise ImportError("모델을 쓰려면 torch 가 필요하다(requirements.txt 의 CPU 빌드 설치 안내 참고)")
    return {"tcn": CausalTCN, "gru": CausalGRU}[arch](n_feat)


def load_model(path):
    """train_boundary.py 가 저장한 .pt 를 읽는다. 반환: (eval 모드 모델, 저장 정보 dict)."""
    if torch is None:
        raise ImportError("모델을 쓰려면 torch 가 필요하다")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("feat_names") != FEAT_NAMES:
        raise ValueError("특징 구성이 이 코드와 다르다(모델을 다시 학습할 것)")
    model = build_model(ck["arch"], ck["n_feat"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck
