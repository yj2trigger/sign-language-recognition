"""외부 수어 MP4 + CSV/JSON 구간 라벨을 기존 프로젝트의 NPZ로 변환.
CSV 컬럼: video,start,end,word_id[,signer,gloss]
JSON: 같은 키의 객체 배열 또는 {"video": "...", "segments": [{start,end,word_id,...}]}
word_id는 학습된 가중치의 원래 클래스 ID와 일치해야 한다.
예: python src/build_youtube_dataset.py --videos D:/test/videos --labels D:/test/labels.csv --out D:/test/features
"""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import landmarks


def describe(values, width=1.0):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    bins, counts = np.unique(np.floor(a / width).astype(int), return_counts=True)
    lower = float(bins[np.argmax(counts)] * width)
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                mode_interval=dict(lower=lower, upper=lower + width, count=int(counts.max())),
                min=float(a.min()), max=float(a.max()), std=float(a.std()),
                p95=float(np.percentile(a, 95)))


def read_labels(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("segments"), list):
            rows = [{**{k: v for k, v in data.items() if k != "segments"}, **seg}
                    for seg in data["segments"]]
        else:
            raise ValueError("JSON은 객체 배열 또는 {video,segments:[...]} 이어야 합니다")
    else:
        raise ValueError("CSV 또는 JSON 라벨만 지원합니다")
    if not rows:
        raise ValueError("라벨이 비었습니다")
    result = []
    for i, r in enumerate(rows, 1):
        video = str(r["video"]).strip()
        start, end = float(r["start"]), float(r["end"])
        if not video or not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"{i}행 시간/영상 정보 오류")
        result.append(dict(video=video, start=start, end=end, word_id=int(r["word_id"]),
                           signer=int(r.get("signer") or 0), gloss=str(r.get("gloss") or "")))
    return result


def extract_timed(path, start, end, margin, complexity):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {path}")
    xys, masks, depths, frame_ms = [], [], [], []
    t0 = time.perf_counter()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        first = max(0, start - margin)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(first * fps)))
        frames = int(round((end + margin - first) * fps)) + 1
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity) as holistic:
            for _ in range(frames):
                ok, bgr = cap.read()
                if not ok:
                    break
                tick = time.perf_counter()
                xy, mask, z, wh = landmarks.process_frame(holistic, bgr)
                frame_ms.append((time.perf_counter() - tick) * 1000)
                xys.append(xy); masks.append(mask); depths.append(z)
    finally:
        cap.release()
    wall = time.perf_counter() - t0
    if not xys:
        raise ValueError(f"추출된 프레임이 없음: {path}")
    return np.stack(xys), np.stack(masks), np.stack(depths), np.array(wh), fps, np.array(frame_ms), wall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--complexity", type=int, choices=(0, 1, 2), default=1)
    a = ap.parse_args()
    if a.margin < 0:
        ap.error("--margin must be >= 0")
    root = Path(a.videos).resolve()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = read_labels(a.labels)
    speeds, all_ms = [], []
    started = time.perf_counter()
    for i, r in enumerate(rows, 1):
        path = (root / r["video"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"잘못된 영상 경로: {r['video']}")
        xy, mask, z, wh, fps, ms, wall = extract_timed(
            path, r["start"], r["end"], a.margin, a.complexity)
        stem = f"yt_{i:06d}"
        np.savez_compressed(out / f"{stem}.npz", xy=xy, mask=mask, z=z, wh=wh,
                            fps=fps, word_id=r["word_id"], signer=r["signer"], gloss=r["gloss"])
        all_ms.extend(ms.tolist())
        processing = float(ms.sum() / 1000.)
        speeds.append(dict(clip=stem, video=r["video"], frames=len(ms), source_fps=fps,
                           mediapipe_seconds=processing, total_extract_seconds=wall,
                           mediapipe_fps=len(ms) / processing,
                           overall_fps=len(ms) / wall,
                           mean_frame_ms=float(ms.mean()), median_frame_ms=float(np.median(ms)),
                           p95_frame_ms=float(np.percentile(ms, 95)),
                           mode_frame_ms=json.dumps(describe(ms)["mode_interval"])))
        print(f"[{i}/{len(rows)}] {stem}: {len(ms)} frames, {len(ms)/processing:.2f} MediaPipe FPS")
    with (out / "extraction_speed.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(speeds[0])); writer.writeheader(); writer.writerows(speeds)
    summary = dict(clips=len(rows), frames=len(all_ms),
                   total_wall_seconds=time.perf_counter() - started,
                   frame_mediapipe_ms=describe(all_ms, 1.0),
                   frame_instant_fps=describe([1000/x for x in all_ms if x > 0], 1.0),
                   mediapipe_throughput_fps=len(all_ms)/sum(s["mediapipe_seconds"] for s in speeds),
                   extract_throughput_fps=len(all_ms)/sum(s["total_extract_seconds"] for s in speeds))
    (out / "extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {len(rows)} clips | MediaPipe {summary['mediapipe_throughput_fps']:.2f} FPS")


if __name__ == "__main__":
    main()
