"""유튜브 수어 영상 전처리 및 처리 속도 측정.

입력:
    videos/news_crop.mp4
    labels/labels.csv
    labels/word_map.json

출력:
    features/yt_000001.npz ...
    results/extraction_speed.csv
    results/extraction_summary.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# 1. 프로젝트 경로
# ==========================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR.parents[1]

sys.path.insert(0, str(SRC_DIR))

import landmarks


VIDEO_DIR = BASE_DIR / "videos"
LABEL_DIR = BASE_DIR / "labels"
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "results"


# ==========================================
# 2. 통계 계산
# ==========================================

def mode_bin(values, width=1.0):
    """연속값의 최빈 구간을 계산한다."""

    a = np.asarray(values, dtype=np.float64)

    if a.size == 0:
        return None

    bins, counts = np.unique(
        np.floor(a / width).astype(int),
        return_counts=True
    )

    left = float(bins[np.argmax(counts)] * width)

    return {
        "from": left,
        "to": left + width,
        "count": int(counts.max())
    }


def describe(values, mode_width=1.0):
    """평균, 중앙값, 최빈 구간 등 통계를 계산한다."""

    a = np.asarray(values, dtype=np.float64)

    if a.size == 0:
        return None

    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "mode_interval": mode_bin(a, mode_width),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std()),
        "p95": float(np.percentile(a, 95))
    }


# ==========================================
# 3. CSV 라벨 읽기
# ==========================================

def read_labels(filename):

    labels = []

    with open(
        filename,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "video_id",
            "label",
            "start_sec",
            "end_sec"
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV에는 video_id, label, start_sec, "
                "end_sec 열이 필요합니다."
            )

        for index, row in enumerate(reader, 1):

            video_id = str(row["video_id"]).strip()
            label = str(row["label"]).strip()

            start = float(row["start_sec"])
            end = float(row["end_sec"])

            if not video_id or not label:
                raise ValueError(
                    f"{index}번째 라벨의 영상명 또는 단어가 비어 있습니다."
                )

            if (
                not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError(
                    f"{index}번째 라벨의 시간 범위가 잘못되었습니다."
                )

            labels.append({
                "video_id": video_id,
                "label": label,
                "start": start,
                "end": end
            })

    if not labels:
        raise ValueError("CSV에 라벨 데이터가 없습니다.")

    return labels


# ==========================================
# 4. 단어 ID 불러오기
# ==========================================

def load_word_map(filename):

    with open(filename, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    return {
        str(label).strip(): int(word_id)
        for label, word_id in data.items()
    }


# ==========================================
# 5. 영상 파일 찾기
# ==========================================

def find_video(video_dir, video_id):

    root = Path(video_dir).resolve()

    name = Path(video_id)

    if name.name != video_id:
        raise ValueError(
            f"video_id에는 파일명만 입력해야 합니다: {video_id}"
        )

    extensions = [".mp4", ".mov", ".avi", ".mkv"]

    if name.suffix:
        candidates = [root / name]
    else:
        candidates = [
            root / f"{video_id}{ext}"
            for ext in extensions
        ]

    matches = [
        path for path in candidates
        if path.is_file()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"같은 이름의 영상이 여러 개 있습니다: {video_id}"
        )

    raise FileNotFoundError(
        f"영상을 찾을 수 없습니다: {video_id}\n"
        f"확인할 폴더: {root}"
    )


# ==========================================
# 6. 영상에서 관절 추출
# ==========================================

def extract_clip(
    video_path,
    start_sec,
    end_sec,
    margin=0.5,
    complexity=1
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    xy_list = []
    mask_list = []
    z_list = []
    frame_times = []

    wall_start = time.perf_counter()

    fps = 0.0
    wh = None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        first_sec = max(0.0, start_sec - margin)
        last_sec = end_sec + margin

        first_frame = int(round(first_sec * fps))
        last_frame = int(round(last_sec * fps))

        frame_count = last_frame - first_frame + 1

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            first_frame
        )

        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=complexity
        ) as holistic:

            for _ in range(frame_count):

                success, frame = cap.read()

                if not success:
                    break

                t0 = time.perf_counter()

                xy, mask, z, wh = landmarks.process_frame(
                    holistic,
                    frame
                )

                elapsed_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                xy_list.append(xy)
                mask_list.append(mask)
                z_list.append(z)
                frame_times.append(elapsed_ms)

    finally:
        cap.release()

    wall_time = time.perf_counter() - wall_start

    if not xy_list:
        raise ValueError(
            f"추출된 프레임이 없습니다: {video_path}"
        )

    return {
        "xy": np.stack(xy_list),
        "mask": np.stack(mask_list),
        "z": np.stack(z_list),
        "wh": np.asarray(wh, dtype=np.int32),
        "fps": fps,
        "frame_times": np.asarray(
            frame_times,
            dtype=np.float64
        ),
        "wall_time": wall_time
    }


# ==========================================
# 7. CSV 저장
# ==========================================

def save_csv(filename, rows):

    if not rows:
        return

    with open(
        filename,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 8. 메인 실행
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--videos",
        default=str(VIDEO_DIR)
    )

    parser.add_argument(
        "--labels",
        default=str(LABEL_DIR / "labels.csv")
    )

    parser.add_argument(
        "--word-map",
        default=str(LABEL_DIR / "word_map.json")
    )

    parser.add_argument(
        "--out",
        default=str(FEATURE_DIR)
    )

    parser.add_argument(
        "--results",
        default=str(RESULT_DIR)
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--complexity",
        type=int,
        choices=[0, 1, 2],
        default=1
    )

    parser.add_argument(
        "--signer",
        type=int,
        default=0
    )

    args = parser.parse_args()

    video_dir = Path(args.videos)
    feature_dir = Path(args.out)
    result_dir = Path(args.results)

    feature_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.labels)
    word_map = load_word_map(args.word_map)

    # 먼저 모든 라벨과 영상을 확인한다.
    jobs = []

    for item in labels:

        label = item["label"]

        if label not in word_map:
            raise ValueError(
                f"word_map.json에 단어가 없습니다: {label}"
            )

        jobs.append({
            **item,
            "word_id": word_map[label],
            "video_path": find_video(
                video_dir,
                item["video_id"]
            )
        })

    print("\n===== 수어 영상 전처리 시작 =====")
    print(f"전체 구간: {len(jobs)}개")

    all_frame_times = []
    speed_rows = []

    total_start = time.perf_counter()

    for index, item in enumerate(jobs, 1):

        label = item["label"]
        word_id = item["word_id"]

        print(
            f"\n[{index}/{len(jobs)}] "
            f"{label} / ID: {word_id}"
        )

        print(
            f"구간: {item['start']} ~ "
            f"{item['end']}초"
        )

        data = extract_clip(
            video_path=item["video_path"],
            start_sec=item["start"],
            end_sec=item["end"],
            margin=args.margin,
            complexity=args.complexity
        )

        filename = f"yt_{index:06d}.npz"
        save_path = feature_dir / filename

        # 기존 프로젝트의 NPZ 형식 유지
        np.savez_compressed(
            save_path,
            xy=data["xy"],
            mask=data["mask"],
            z=data["z"],
            wh=data["wh"],
            fps=data["fps"],
            word_id=np.int32(word_id),
            signer=np.int32(args.signer),
            gloss=np.asarray(label)
        )

        times = data["frame_times"]

        all_frame_times.extend(times.tolist())

        frames = len(times)

        mediapipe_seconds = float(times.sum() / 1000)
        wall_seconds = float(data["wall_time"])

        mediapipe_fps = (
            frames / mediapipe_seconds
            if mediapipe_seconds > 0
            else 0.0
        )

        total_fps = (
            frames / wall_seconds
            if wall_seconds > 0
            else 0.0
        )

        stats = describe(times)

        speed_rows.append({
            "file": filename,
            "video": item["video_id"],
            "label": label,
            "word_id": word_id,
            "start_sec": item["start"],
            "end_sec": item["end"],
            "frames": frames,
            "source_fps": data["fps"],
            "mediapipe_seconds": mediapipe_seconds,
            "total_seconds": wall_seconds,
            "mediapipe_fps": mediapipe_fps,
            "total_fps": total_fps,
            "mean_ms": stats["mean"],
            "median_ms": stats["median"],
            "mode_ms": json.dumps(
                stats["mode_interval"]
            ),
            "min_ms": stats["min"],
            "max_ms": stats["max"],
            "std_ms": stats["std"],
            "p95_ms": stats["p95"]
        })

        print(f"처리 프레임: {frames}개")
        print(f"MediaPipe FPS: {mediapipe_fps:.2f}")
        print(f"전체 처리 FPS: {total_fps:.2f}")
        print(f"평균 처리 시간: {stats['mean']:.2f}ms")
        print(f"저장 완료: {filename}")

    # ======================================
    # 9. 전체 통계
    # ======================================

    total_elapsed = time.perf_counter() - total_start

    all_times = np.asarray(
        all_frame_times,
        dtype=np.float64
    )

    total_frames = len(all_times)
    total_mediapipe_seconds = float(all_times.sum() / 1000)

    overall_fps = (
        total_frames / total_mediapipe_seconds
        if total_mediapipe_seconds > 0
        else 0.0
    )

    total_wall_fps = (
        total_frames / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    valid_times = all_times[all_times > 0]

    instant_fps = 1000.0 / valid_times

    frame_stats = describe(all_times)
    fps_stats = describe(instant_fps)

    summary = {
        "total_clips": len(jobs),
        "total_frames": total_frames,
        "total_elapsed_seconds": total_elapsed,
        "mediapipe_total_seconds": total_mediapipe_seconds,
        "mediapipe_fps": overall_fps,
        "overall_wall_fps": total_wall_fps,
        "frame_time_ms": frame_stats,
        "instant_fps": fps_stats
    }

    # ======================================
    # 10. 결과 저장
    # ======================================

    save_csv(
        result_dir / "extraction_speed.csv",
        speed_rows
    )

    with open(
        result_dir / "extraction_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ======================================
    # 11. 완료 출력
    # ======================================

    print("\n========== 전처리 완료 ==========")

    print(f"총 수어 구간: {len(jobs)}개")
    print(f"총 프레임: {total_frames}개")
    print(f"전체 실행 시간: {total_elapsed:.2f}초")
    print(f"MediaPipe FPS: {overall_fps:.2f}")
    print(f"전체 처리 FPS: {total_wall_fps:.2f}")

    if frame_stats is not None:
        print(f"평균: {frame_stats['mean']:.2f}ms")
        print(f"중앙값: {frame_stats['median']:.2f}ms")
        print(f"P95: {frame_stats['p95']:.2f}ms")
        print(f"최빈 구간: {frame_stats['mode_interval']}")

    print(f"\nNPZ 저장 위치: {feature_dir}")
    print(f"통계 저장 위치: {result_dir}")

    print("=================================")


if __name__ == "__main__":
    main()