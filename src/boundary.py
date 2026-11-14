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
import datetime
import math
import os
import re
from collections import deque
from dataclasses import dataclass

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


# ---- 실시간 탐지기 ----

@dataclass
class DetectorConfig:
    fps: float = 15.0
    window: int = 61               # 모델에 넣는 최근 프레임 수. 학습과 맞춘 값은 model_window(), .pt 에 저장된다
    on_frames: int = 2             # 활동이 이만큼 이어지면 구간 시작
    off_frames: int = 4            # 비활동이 이만큼 이어지면 구간 끝(15fps 약 0.27초). 손이 잠깐 안 잡혀도 끊지 않는다
    act_thr: float = 0.5           # 모델: P(B)+P(I) 가 이 이상이면 활동
    begin_thr: float = 0.3         # 모델: 구간 중 P(B) 가 이 이상이면 앞 단어를 닫고 새 단어 시작(실제 문장 검증에서 고름)
    move_speed: float = 0.6        # 규칙: 손끝 속도(어깨너비/초)가 이보다 빠르면 활동
    hold_speed: float = 0.5        # 규칙: 손을 든 채 이보다 느리면 '멈춤'(손목 속도와 손가락 속도 둘 다)
    word_pause_s: float = 0.25     # 규칙: 멈춤이 이만큼 이어지면 단어 끝
    min_seg_s: float = 0.2         # 이보다 짧은 구간은 버린다(잡음)
    pause_s: float = 0.8           # 문장 끝: 마지막 단어 뒤 멈춤이 최소 이만큼 이어져야 한다
    max_pause_s: float = 1.2       # 문장 끝: 멈춤이 이만큼이면 다른 단서와 무관하게 문장 끝(실제 문장 검증에서 고름)
    cue_thr: float = 0.5           # 문장 끝: pause_s~max_pause_s 사이에서는 단서 점수가 이 이상일 때 끝
    len_ratio: float = 1.3         # 마지막 단어가 앞 단어 중앙값의 이 배 이상이면 길이 단서 1(같으면 0.5)
    face_cue: bool = False         # 표정 단서를 쓸지. 실제 문장 검증에서 켜면 문장 끝을 더 놓쳐 기본은 끈다
    face_ref: float = 0.06         # 눈썹·입 변화(얼굴 높이 단위)가 이만큼이면 표정 단서 1
    face_tail_s: float = 0.5       # 마지막 단어 끝 이 초 전부터의 표정을 문장 평균과 비교
    fs_min_s: float = 0.6          # 지문자: 이 이상 길고
    fs_ratio: float = 2.0          #   손가락 움직임 / 손목 움직임이 이 배 이상이면 지문자


class BoundaryDetector:
    """프레임을 하나씩 받아 단어·지문자 구간과 문장 끝을 알린다.

    Args:
        model (str | torch.nn.Module | None): .pt 경로, 이미 만든 모델, 또는 None(규칙 모드).
        config (DetectorConfig | None): 문턱값. None 이면 기본값(모델 파일에 저장한 값이 있으면 그것).

    Examples:
        >>> det = BoundaryDetector()
        >>> det.update(0.0, np.zeros((53, 2)), np.zeros(53, bool)) is None
        True
    """

    @property
    def uses_face(self):
        """realtime.py 가 얼굴 33점을 뽑아 update(..., face=, face_mask=) 로 넘길지. 표정 단서를 켰을 때만."""
        return self.cfg.face_cue

    def __init__(self, model=None, config=None):
        saved, crop = {}, None
        if isinstance(model, str):
            model, ck = load_model(model)
            saved, crop = ck.get("detector", {}), ck.get("train_crop")
        self.model = None if model is None else model.eval()   # 학습 모드로 넘겨도 dropout 을 끈다(실시간 결과가 매번 같게)
        self.cfg = config or DetectorConfig(**saved)
        if model is not None:
            rf = receptive_field(model)
            if rf is not None and self.cfg.window < rf:
                raise ValueError("입력 창(%d)이 TCN 받아들이는 범위(%d)보다 짧아 학습 때와 출력이 달라진다" % (self.cfg.window, rf))
            if rf is None and crop is not None and self.cfg.window != crop:
                raise ValueError("GRU 입력 창(%d)이 학습 조각 길이(%d)와 다르다" % (self.cfg.window, crop))
        self.fx = FeatureExtractor(self.cfg.fps)
        self.reset()

    def reset(self):
        """새 영상·새 사람을 시작할 때 부른다. 버퍼와 진행 중 구간·문장을 모두 비운다."""
        self.fx.reset()
        self.feats = deque(maxlen=self.cfg.window)
        self.last_t = None
        self.start = self.cand = self.last_active = None
        self.run_on = self.run_off = 0
        self.still_since = None
        self.need_motion = False
        self.seg_wrist = self.seg_finger = 0.0
        self.sentence = []                   # 지금 문장의 단어 [(시작, 끝, 종류)]
        self.face_hist = deque(maxlen=int(self.cfg.fps * 30))   # (t, 눈썹, 입), 문장 시작부터
        self.last_probs = None

    # -- 판정 부품 --
    def _activity(self, still):
        """반환: (활동 여부, 새 단어 시작 확률 또는 None)."""
        if self.model is not None:
            x = torch.from_numpy(np.stack(self.feats)[None])
            with torch.no_grad():
                p = torch.softmax(self.model(x)[0, -1], 0).numpy()
            self.last_probs = p
            # 손(손 21점·포즈 손목)이 하나도 안 보이면 모델 출력과 무관하게 비활동. 학습 영상에는 이런 프레임이 거의 없어
            # 모델이 마지막 활동 상태를 이어 가는 일이 있었다(사람이 화면에서 사라진 영상). 짧은 미검출은 off_frames 가 흡수한다
            return self.fx.last["visible"] and float(p[1] + p[2]) >= self.cfg.act_thr, float(p[1])
        last = self.fx.last
        if self.need_motion and not still:
            self.need_motion = False
        active = last["present"] and (last["raised"] or last["speed"] >= self.cfg.move_speed)
        return active and not self.need_motion, None

    def _seg_kind(self, dur):
        if dur >= self.cfg.fs_min_s - EPS and self.seg_finger >= self.cfg.fs_ratio * max(self.seg_wrist, 1e-6):
            return "fingerspelling"
        return "word"

    def _close(self, end):
        start, self.start = self.start, None
        self.run_off = 0
        self.still_since = None
        dur = end - start
        if dur < self.cfg.min_seg_s - EPS:
            return None
        seg = (start, end, self._seg_kind(dur))
        self.sentence.append(seg)
        return seg

    def _open(self, t):
        self.start, self.last_active = t, t
        self.b_armed = False               # 새 단어를 연 B 확률이 한 번 내려가야 다음 B 로 자를 수 있다
        self.seg_wrist = self.seg_finger = 0.0
        self.run_off = 0
        if not self.sentence:              # 문장의 첫 단어: 표정 기준을 새로 잡는다(앞 문장 표정을 섞지 않게)
            while self.face_hist and self.face_hist[0][0] < t - 1.0:
                self.face_hist.popleft()

    def sentence_cues(self, t):
        """문장 끝 단서 점수 (길이, 표정). 계산할 수 없으면 그 값은 None."""
        cfg = self.cfg
        length = face = None
        if len(self.sentence) >= 2:
            durs = [e - s for s, e, _ in self.sentence]
            ratio = durs[-1] / max(np.median(durs[:-1]), 1e-6)
            length = float(np.clip(0.5 + 0.5 * (ratio - 1) / (cfg.len_ratio - 1), 0, 1))
        if cfg.face_cue and self.face_hist and self.sentence:
            arr = np.array([(ft, b, m) for ft, b, m in self.face_hist if ft >= self.sentence[0][0]])
            if len(arr) >= 4:
                tail = arr[arr[:, 0] >= self.sentence[-1][1] - cfg.face_tail_s]
                body = arr[arr[:, 0] < self.sentence[-1][1] - cfg.face_tail_s]
                if len(tail) and len(body):
                    change = np.abs(np.median(tail[:, 1:], 0) - np.median(body[:, 1:], 0)).max()
                    face = float(np.clip(change / cfg.face_ref, 0, 1))
        return length, face

    def _check_sentence_end(self, t):
        pause = t - self.sentence[-1][1]
        if pause < self.cfg.pause_s - EPS:
            return None
        cues = [c for c in self.sentence_cues(t) if c is not None]
        score = float(np.mean(cues)) if cues else 0.5       # 단서가 없으면 중립: 멈춤 pause_s 만으로 끝
        if score >= self.cfg.cue_thr or pause >= self.cfg.max_pause_s - EPS:
            out = (self.sentence[0][0], self.sentence[-1][1], "sentence_end")
            self.sentence = []
            return out
        return None

    # -- 공개 API --
    def update(self, t, xy, mask, face=None, face_mask=None):
        """프레임 하나를 넣는다.

        Args:
            t (float): 프레임 시각(초). 앞 프레임보다 크지 않으면 그 프레임은 무시한다(None).
            xy (np.ndarray): (53, 2) 정규화 좌표. NaN·화면 밖 점은 마스크를 끈 것으로 본다.
            mask (np.ndarray): (53,) 검출 여부. 모두 0 이어도 된다(손이 화면 밖).
            face, face_mask (np.ndarray | None): (33, 2), (33,) 얼굴 점. 문장 끝 보조 단서.

        Returns:
            tuple[float, float, str] | None: 구간이 끝난 프레임에서만 (시작, 끝, 종류).

        Raises:
            ValueError: t 가 유한한 수가 아니거나 xy·mask 모양이 틀릴 때.
        """
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t 가 유한한 수가 아니다: %r" % (t,))
        if self.last_t is not None and t <= self.last_t:
            return None
        self.last_t = t
        self.feats.append(self.fx.step(t, xy, mask))
        fs = face_signals(face, face_mask)
        if fs is not None:
            self.face_hist.append((t,) + fs)
        last, cfg = self.fx.last, self.cfg
        still = last["wrist_speed"] < cfg.hold_speed and last["finger"] < cfg.hold_speed
        active, p_begin = self._activity(still)

        if self.start is None:                                    # 쉬는 중
            if active:
                self.run_on += 1
                if self.run_on == 1:
                    self.cand = t
                if self.run_on >= cfg.on_frames:
                    self._open(self.cand)
                    self.run_on = 0
                return None
            self.run_on = 0
            return self._check_sentence_end(t) if self.sentence else None

        if active:                                                # 구간 안
            self.run_off = 0
            self.seg_wrist += last["wrist_speed"]
            self.seg_finger += last["finger"]
            if p_begin is not None:                               # 단어와 단어가 쉬지 않고 이어질 때: B 가 다시 올라오면 자른다
                if p_begin < cfg.begin_thr:
                    self.b_armed = True
                elif self.b_armed and t - self.start >= cfg.min_seg_s - EPS:
                    seg = self._close(self.last_active)
                    self._open(t)
                    return seg
            if self.model is None:                                # 규칙: 손을 든 채 멈추면 단어 끝
                if still:
                    self.still_since = self.still_since if self.still_since is not None else self.last_active
                    if t - self.still_since >= cfg.word_pause_s - EPS:
                        self.need_motion = True
                        return self._close(self.still_since)
                    return None
                self.still_since = None
            self.last_active = t
            return None
        self.run_off += 1
        if self.run_off >= cfg.off_frames:
            return self._close(self.still_since if self.still_since is not None else self.last_active)
        return None

    def flush(self):
        """스트림이 끝났을 때 진행 중인 단어와 문장을 닫는다. 반환: 이벤트 리스트(0~2개)."""
        out = []
        if self.start is not None:
            seg = self._close(self.still_since if self.still_since is not None else self.last_active)
            if seg:
                out.append(seg)
        if self.sentence:
            out.append((self.sentence[0][0], self.sentence[-1][1], "sentence_end"))
            self.sentence = []
        return out


# ---- 검증: 문장 영상 단위의 경계 정확도와 지연 ----

def run_stream(det, t, xy, mask, face=None, face_mask=None, flush=True):
    """영상 하나를 탐지기에 프레임 순서대로 넣는다. 반환: [(알린 시각, 시작, 끝, 종류)]. flush 로 닫힌 것은 마지막 프레임 시각."""
    det.reset()
    ev = []
    for i in range(len(t)):
        r = det.update(t[i], xy[i], mask[i], None if face is None else face[i], None if face_mask is None else face_mask[i])
        if r:
            ev.append((float(t[i]),) + r)
    if flush and len(t):
        ev += [(float(t[-1]),) + r for r in det.flush()]
    return ev


def segments_to_frames(t, segments):
    """구간 목록을 프레임별 활동 여부 (T,) bool 로."""
    t = np.asarray(t, float)
    act = np.zeros(len(t), bool)
    for s, e, *_ in segments:
        act |= (t >= s) & (t <= e)
    return act


def frame_f1(t, pred_segments, true_segments):
    """프레임 단위 활동(구간 안/밖) 정밀도·재현율·F1."""
    p, g = segments_to_frames(t, pred_segments), segments_to_frames(t, true_segments)
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(a[:2] == b[:2])


def match_segments(pred, true, iou_thr=0.5):
    """IoU 가 큰 쌍부터 1:1 로 맞춘다. 반환: [(예측 번호, 정답 번호, IoU)]."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(true)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); out.append((i, j, v))
    return out


def segment_scores(pred, true, iou_thr=0.5):
    """IoU ≥ iou_thr 를 맞힘으로 본 구간 정밀도·재현율·F1 과 맞힌 쌍의 평균 IoU."""
    m = match_segments(pred, true, iou_thr)
    prec = len(m) / len(pred) if pred else 0.0
    rec = len(m) / len(true) if true else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "mean_iou": float(np.mean([v for *_, v in m])) if m else 0.0, "n_pred": len(pred), "n_true": len(true)}


def evaluate_stream(events, t, true_segments, iou_thr=0.5):
    """run_stream 결과 하나를 정답과 비교한다.

    Returns:
        dict: frame(F1 등), segment(IoU 기준 F1), latency_s(맞힌 단어마다 알린 시각 - 정답 끝),
        sentence_end_latency_s(문장 끝을 알린 시각 - 정답 마지막 단어 끝, 못 알렸으면 None), n_sentence_end(알린 횟수).
    """
    words = [e for e in events if e[3] != "sentence_end"]
    pred = [e[1:] for e in words]
    m = match_segments(pred, true_segments, iou_thr)
    sent = [e for e in events if e[3] == "sentence_end"]
    true_end = max(e for _, e, *_ in true_segments) if true_segments else None
    return {"frame": frame_f1(t, pred, true_segments), "segment": segment_scores(pred, true_segments, iou_thr),
            "latency_s": [words[i][0] - true_segments[j][1] for i, j, _ in m],
            "sentence_end_latency_s": sent[0][0] - true_end if sent and true_end is not None else None,
            "n_sentence_end": len(sent)}


def summarize(results):
    """evaluate_stream 결과 여러 개를 합친다(프레임·구간 지표는 영상 평균, 지연은 전체 중앙값·p95)."""
    lat = [x for r in results for x in r["latency_s"]]
    slat = [r["sentence_end_latency_s"] for r in results if r["sentence_end_latency_s"] is not None]
    mean = lambda f: float(np.mean([f(r) for r in results])) if results else 0.0
    return {"n_videos": len(results),
            "frame_f1": mean(lambda r: r["frame"]["f1"]),
            "segment_f1": mean(lambda r: r["segment"]["f1"]),
            "mean_iou": mean(lambda r: r["segment"]["mean_iou"]),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "sentence_end_rate": mean(lambda r: float(r["n_sentence_end"] == 1)),
            "sentence_end_latency_median_s": float(np.median(slat)) if slat else None}


# ---- 실험 기록: logs/sign-language/trials.md 에 T-번호 항목을 덧붙인다 ----

def format_trial(tid, title, hypothesis, attempt, result, decision, date=None):
    """실험 기록 한 항목(markdown). 단어명·문장 원문은 넣지 말 것(지표·설정만)."""
    date = date or datetime.date.today().isoformat()
    return "\n".join(["## %s %s (%s)" % (tid, title, date), "",
                      "- 가설: " + hypothesis, "- 시도: " + attempt, "- 결과: " + result, "- 판단: " + decision, ""])


def next_trial_id(text):
    nums = [int(n) for n in re.findall(r"^## T-(\d+)", text, re.M)]
    return "T-%03d" % (max(nums) + 1 if nums else 1)


def append_trial(path, title, hypothesis, attempt, result, decision, date=None):
    """path 에 다음 T-번호로 항목을 덧붙이고 그 번호를 돌려준다. 파일·폴더가 없으면 만든다."""
    text = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    tid = next_trial_id(text)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if text and not text.endswith("\n\n"):
            f.write("\n" if text.endswith("\n") else "\n\n")
        f.write(format_trial(tid, title, hypothesis, attempt, result, decision, date))
    return tid
