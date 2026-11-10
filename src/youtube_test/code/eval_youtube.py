"""외부 수어 영상 NPZ 데이터의 TCN 모델 평가.

기능:
1. 기존 학습 완료 모델(.pt) 불러오기
2. NPZ 데이터 불러오기
3. 학습하지 않은 단어 자동 제외
4. 기존 학습 설정에 맞게 특징 생성
5. TCN 모델 추론
6. 정답률 및 Macro F1 계산
7. 추론 속도 측정
8. CSV 및 JSON 결과 저장
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. 프로젝트 경로
# ==========================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
BASE_DIR = CURRENT_DIR.parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

import preprocess
import models


DEFAULT_MODEL = (
    PROJECT_DIR
    / "models"
    / "final"
    / "final-tcn-c300-s0.pt"
)

DEFAULT_FEATURES = BASE_DIR / "features"
DEFAULT_RESULTS = BASE_DIR / "results"


# ==========================================
# 2. 통계 함수
# ==========================================

def mode_bin(values, width=1.0):

    a = np.asarray(values, dtype=np.float64)

    if a.size == 0:
        return None

    bins, counts = np.unique(
        np.floor(a / width).astype(np.int64),
        return_counts=True
    )

    lower = float(
        bins[np.argmax(counts)] * width
    )

    return {
        "from": lower,
        "to": lower + width,
        "count": int(counts.max())
    }


def describe(values, width=1.0):

    a = np.asarray(values, dtype=np.float64)

    if a.size == 0:
        return None

    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "mode_interval": mode_bin(a, width),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std()),
        "p95": float(np.percentile(a, 95))
    }


# ==========================================
# 3. CSV 저장
# ==========================================

def write_csv(path, records, columns):

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(records)


# ==========================================
# 4. 모델 전처리 설정
# ==========================================

def prepare_features(data, cfg):

    xy = data["xy"]
    mask = data["mask"]

    use_z = cfg.get("hand_z", False)

    if use_z and "z" not in data.files:
        return None

    z = data["z"] if use_z else None

    if "wh" in data.files:
        wh = data["wh"]
        aspect = float(wh[0]) / float(wh[1])
    else:
        aspect = 16 / 9

    features = preprocess.build_features(
        xy,
        mask,
        length=int(cfg["length"]),
        with_hand_rel=cfg.get("hand_rel", True),
        with_angles=cfg.get("angles", False),
        with_traj=cfg.get("traj", False),
        with_bones=cfg.get("bones", False),
        origin=cfg.get("origin", "shoulder"),
        z=z,
        aspect=aspect,
        with_hand_z=use_z,
        with_folds=cfg.get("folds", False)
    )

    return features


# ==========================================
# 5. 메인
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL)
    )

    parser.add_argument(
        "--features",
        default=str(DEFAULT_FEATURES)
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS)
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats는 1 이상이어야 합니다.")

    model_path = Path(args.model)
    features_dir = Path(args.features)
    results_dir = Path(args.out)

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"가중치 파일이 없습니다: {model_path}"
        )

    files = sorted(
        features_dir.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"NPZ 파일이 없습니다: {features_dir}"
        )

    # ======================================
    # 6. 기존 TCN 모델 불러오기
    # ======================================

    print("\n========== TCN 모델 평가 ==========")

    # 신뢰할 수 있는 체크포인트 파일만 로드
    ck = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    cfg = ck["config"]

    classes = [
        int(word_id)
        for word_id in ck["classes"]
    ]

    class_index = {
        word_id: index
        for index, word_id in enumerate(classes)
    }

    model = models.build(
        ck["model"],
        int(ck["n_feat"]),
        len(classes)
    )

    model.load_state_dict(
        ck["state"]
    )

    model.eval()

    print(f"모델: {ck['model']}")
    print(f"학습 클래스: {len(classes)}개")
    print(f"입력 특징: {ck['n_feat']}개")
    print(f"입력 프레임: {cfg['length']}개")
    print(f"NPZ 파일: {len(files)}개")

    # ======================================
    # 7. 결과 저장 변수
    # ======================================

    prediction_rows = []
    speed_rows = []
    skipped_rows = []

    y_true = []
    y_pred = []

    inference_times = []
    preprocessing_times = []

    total_start = time.perf_counter()

    # ======================================
    # 8. NPZ 파일별 모델 평가
    # ======================================

    for index, file in enumerate(files, 1):

        print(
            f"\n[{index}/{len(files)}] {file.name}"
        )

        with np.load(
            file,
            allow_pickle=False
        ) as data:

            word_id = int(data["word_id"])

            gloss = (
                str(data["gloss"].item())
                if "gloss" in data.files
                else ""
            )

            # 학습하지 않은 단어 제외
            if word_id not in class_index:

                reason = (
                    f"학습 클래스에 없는 단어: "
                    f"{gloss} (ID {word_id})"
                )

                skipped_rows.append({
                    "file": file.name,
                    "label": gloss,
                    "word_id": word_id,
                    "reason": reason
                })

                print(f"제외: {reason}")

                continue

            # ----------------------------------
            # 특징 전처리 시간 측정
            # ----------------------------------

            t0 = time.perf_counter()

            features = prepare_features(
                data,
                cfg
            )

            prep_ms = (
                time.perf_counter() - t0
            ) * 1000

            if features is None:

                reason = (
                    "관절 검출 실패 또는 "
                    "필요한 깊이 특징 없음"
                )

                skipped_rows.append({
                    "file": file.name,
                    "label": gloss,
                    "word_id": word_id,
                    "reason": reason
                })

                print(f"제외: {reason}")

                continue

            if features.shape[1] != ck["n_feat"]:
                raise ValueError(
                    f"특징 개수 불일치: "
                    f"{features.shape[1]} != "
                    f"{ck['n_feat']}"
                )

        preprocessing_times.append(
            prep_ms
        )

        # ----------------------------------
        # 9. 모델 입력
        # ----------------------------------

        x = torch.from_numpy(
            features[None]
        ).float()

        # 워밍업: 초기 실행 오버헤드를
        # 추론 속도 측정에서 제외
        with torch.inference_mode():
            model(x)

        clip_times = []

        # ----------------------------------
        # 10. 반복 추론
        # ----------------------------------

        with torch.inference_mode():

            for repeat in range(args.repeats):

                start = time.perf_counter()

                logits = model(x)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                clip_times.append(
                    elapsed_ms
                )

                inference_times.append(
                    elapsed_ms
                )

                speed_rows.append({
                    "file": file.name,
                    "label": gloss,
                    "repeat": repeat + 1,
                    "inference_ms": elapsed_ms,
                    "clips_per_second": (
                        1000 / elapsed_ms
                        if elapsed_ms > 0
                        else 0
                    )
                })

        # ----------------------------------
        # 11. 예측 결과
        # ----------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_word_id = classes[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index].item()
        )

        correct = (
            predicted_word_id == word_id
        )

        mean_inference_ms = float(
            np.mean(clip_times)
        )

        prediction_rows.append({
            "file": file.name,
            "true_label": gloss,
            "true_word_id": word_id,
            "predicted_word_id": predicted_word_id,
            "confidence": confidence,
            "correct": int(correct),
            "preprocessing_ms": prep_ms,
            "inference_mean_ms": mean_inference_ms
        })

        y_true.append(
            class_index[word_id]
        )

        y_pred.append(
            predicted_index
        )

        print(
            f"정답: {gloss} ({word_id})"
        )

        print(
            f"예측 ID: {predicted_word_id}"
        )

        print(
            f"정답 여부: "
            f"{'O' if correct else 'X'}"
        )

        print(
            f"추론 평균: "
            f"{mean_inference_ms:.3f} ms"
        )

    # ======================================
    # 12. 정확도 및 F1 계산
    # ======================================

    evaluated = len(y_true)
    skipped = len(skipped_rows)

    if evaluated > 0:

        accuracy = float(
            accuracy_score(
                y_true,
                y_pred
            )
        )

        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

    else:

        accuracy = None
        macro_f1 = None

    total_elapsed = (
        time.perf_counter() - total_start
    )

    inference_stats = describe(
        inference_times,
        width=1.0
    )

    preprocessing_stats = describe(
        preprocessing_times,
        width=1.0
    )

    speed_values = [
        1000 / ms
        for ms in inference_times
        if ms > 0
    ]

    speed_stats = describe(
        speed_values,
        width=10.0
    )

    # ======================================
    # 13. 전체 결과 정리
    # ======================================

    summary = {
        "model": str(model_path),
        "model_type": ck["model"],
        "model_classes": len(classes),
        "total_npz": len(files),
        "evaluated_clips": evaluated,
        "skipped_clips": skipped,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "repeats": args.repeats,
        "total_elapsed_seconds": total_elapsed,
        "inference_time_ms": inference_stats,
        "preprocessing_time_ms": preprocessing_stats,
        "clips_per_second": speed_stats,
        "skipped": skipped_rows
    }

    # ======================================
    # 14. CSV 저장
    # ======================================

    write_csv(
        results_dir / "youtube_predictions.csv",
        prediction_rows,
        [
            "file",
            "true_label",
            "true_word_id",
            "predicted_word_id",
            "confidence",
            "correct",
            "preprocessing_ms",
            "inference_mean_ms"
        ]
    )

    write_csv(
        results_dir / "youtube_speed.csv",
        speed_rows,
        [
            "file",
            "label",
            "repeat",
            "inference_ms",
            "clips_per_second"
        ]
    )

    write_csv(
        results_dir / "youtube_skipped.csv",
        skipped_rows,
        [
            "file",
            "label",
            "word_id",
            "reason"
        ]
    )

    with open(
        results_dir / "youtube_summary.json",
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
    # 15. 최종 결과 출력
    # ======================================

    print("\n========== 최종 평가 결과 ==========")

    print(f"전체 구간: {len(files)}개")
    print(f"평가 구간: {evaluated}개")
    print(f"제외 구간: {skipped}개")

    if accuracy is not None:

        print(
            f"정확도: {accuracy * 100:.2f}%"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

    else:

        print(
            "평가 가능한 구간이 없습니다."
        )

    if inference_stats is not None:

        print(
            f"평균 추론 시간: "
            f"{inference_stats['mean']:.3f} ms"
        )

        print(
            f"중앙값: "
            f"{inference_stats['median']:.3f} ms"
        )

        print(
            f"P95: "
            f"{inference_stats['p95']:.3f} ms"
        )

        print(
            f"최빈 구간: "
            f"{inference_stats['mode_interval']}"
        )

    print(
        f"전체 실행 시간: "
        f"{total_elapsed:.2f}초"
    )

    print(
        f"\n결과 저장 위치: {results_dir}"
    )

    print("====================================")


if __name__ == "__main__":
    main()