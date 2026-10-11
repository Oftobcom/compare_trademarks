#!/usr/bin/env python3
"""
compare_trademarks.py — сравнение товарного знака с коллекцией изображений.

Комбинирует три независимые метрики:
  SIFT%  — SIFT + Lowe ratio + RANSAC-гомография (геометрия формы);
  Color% — корреляция HSV-гистограмм (цветовая схема);
  pHash% — perceptual hash (общая композиция, устойчив к масштабу/яркости).

Итог = взвешенная сумма (см. --weights).

Примеры:
    python compare_trademarks.py
    python compare_trademarks.py --new logo.jpg --collections base --csv report.csv
    python compare_trademarks.py --weights 0.5 0.3 0.2 --threshold 25 --visualize
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------- загрузка ----------------------

def load_bgr(path: Path, max_dim: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


# ---------------------- pHash (DCT-based) ----------------------

def phash(bgr: np.ndarray, hash_size: int = 8, hf: int = 4) -> np.ndarray:
    """Perceptual hash: DCT -> порог по медиане (без DC-компоненты)."""
    size = hash_size * hf
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])
    return (low > med).flatten()


def phash_score(h1: np.ndarray, h2: np.ndarray) -> float:
    diff = int(np.count_nonzero(h1 != h2))
    return max(0.0, 100.0 * (1.0 - diff / h1.size))


# ---------------------- цвет ----------------------

def color_score(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """Сравнение HSV-гистограмм (H+S). Bhattacharyya: 0=идентичны, 1=разные."""
    hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    h2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    dist = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(100.0, 100.0 * (1.0 - dist)))


# ---------------------- SIFT + RANSAC ----------------------

def sift_score(
    kp1, des1, gray2,
    sift, flann,
    ratio: float, ransac_thresh: float, min_inliers: int,
):
    """
    Возвращает:
        score, good_count, inliers, n1, n2, kp2, des2, extra
    где extra = (good, mask) либо None (если homography не удалась).
    """
    kp2, des2 = sift.detectAndCompute(gray2, None)
    n1 = len(kp1) if kp1 else 0
    n2 = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0, 0, 0, n1, n2, kp2, des2, None

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0.0, len(good), 0, n1, n2, kp2, des2, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return 0.0, len(good), 0, n1, n2, kp2, des2, None

    inliers = int(mask.sum())

    # Доля инлайеров относительно меньшего набора признаков +
    # "уверенность" по абсолютному числу инлайеров.
    denom = max(1, min(n1, n2))
    raw = inliers / denom
    conf = min(1.0, inliers / max(1, min_inliers))
    score = 100.0 * raw * conf

    return score, len(good), inliers, n1, n2, kp2, des2, (good, mask)


# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--new", default="new.jpg", help="эталонное изображение")
    p.add_argument("--collections", default="collections", help="папка с коллекцией")
    p.add_argument("--csv", default=None, help="сохранить отчёт в CSV")
    p.add_argument("--threshold", type=float, default=20.0,
                   help="порог Total%% для пометки YES")
    p.add_argument("--weights", nargs=3, type=float, default=(0.60, 0.25, 0.15),
                   metavar=("SIFT", "COLOR", "PHASH"),
                   help="веса метрик SIFT / Color / pHash")
    p.add_argument("--max-dim", type=int, default=1200,
                   help="максимальная сторона изображения")
    p.add_argument("--ratio", type=float, default=0.75,
                   help="Lowe ratio для отбора совпадений")
    p.add_argument("--ransac", type=float, default=5.0,
                   help="порог репроекции RANSAC, пиксели")
    p.add_argument("--min-inliers", type=int, default=15,
                   help="число инлайеров для полной уверенности")
    p.add_argument("--sift-features", type=int, default=4000,
                   help="максимум SIFT-признаков на изображение")
    p.add_argument("--top", type=int, default=0, help="0 — показать все строки")
    p.add_argument("--visualize", action="store_true",
                   help="сохранить top1_matches.jpg для лучшего совпадения")
    return p.parse_args()


# ---------------------- main ----------------------

def main():
    args = parse_args()

    new_path = Path(args.new)
    coll_dir = Path(args.collections)
    if not new_path.exists():
        sys.exit(f"Не найден файл: {new_path}")
    if not coll_dir.is_dir():
        sys.exit(f"Не найдена папка: {coll_dir}")

    img_new = load_bgr(new_path, args.max_dim)
    if img_new is None:
        sys.exit(f"Не удалось открыть: {new_path}")

    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    hash_new = phash(img_new)

    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),   # FLANN_INDEX_KDTREE
        dict(checks=50),
    )

    kp_new, des_new = sift.detectAndCompute(gray_new, None)
    if not kp_new or len(kp_new) < 10:
        print(f"Внимание: на {new_path.name} найдено мало SIFT-признаков "
              f"({len(kp_new) if kp_new else 0}); SIFT-метрика может быть ненадёжна.")

    w_sift, w_color, w_phash = args.weights
    w_sum = max(1e-9, w_sift + w_color + w_phash)

    files = sorted(
        p for p in coll_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not files:
        sys.exit(f"В папке {coll_dir} нет изображений")

    rows = []
    best = None  # (total, path, kp2, extra, img2)

    for i, path in enumerate(files, 1):
        img2 = load_bgr(path, args.max_dim)
        if img2 is None:
            print(f"[{i}/{len(files)}] пропуск {path.name}: не читается")
            continue

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        s_score, good_n, inl, n1, n2, kp2, des2, extra = sift_score(
            kp_new, des_new, gray2,
            sift, flann,
            args.ratio, args.ransac, args.min_inliers,
        )
        c_score = color_score(img_new, img2)
        p_score = phash_score(hash_new, phash(img2))

        total = (w_sift * s_score + w_color * c_score + w_phash * p_score) / w_sum

        rows.append({
            "name": path.name,
            "total": total,
            "sift": s_score,
            "color": c_score,
            "phash": p_score,
            "inliers": inl,
            "good": good_n,
            "n_new": n1,
            "n_coll": n2,
        })

        if best is None or total > best[0]:
            best = (total, path, kp2, extra, img2)

    rows.sort(key=lambda r: r["total"], reverse=True)

    # ---------------- вывод ----------------
    header = (
        f"{'#':>3}  {'Файл':38}  {'Total%':>7}  {'SIFT%':>6}  "
        f"{'Color%':>7}  {'pHash%':>7}  {'Inl':>4}  {'Good':>5}  "
        f"{'KPn/KPc':>11}  Match"
    )
    print(header)
    print("-" * len(header))

    shown = rows if args.top <= 0 else rows[:args.top]
    for i, r in enumerate(shown, 1):
        flag = "YES" if r["total"] >= args.threshold else ""
        print(
            f"{i:>3}  {r['name']:38}  {r['total']:7.2f}  {r['sift']:6.2f}  "
            f"{r['color']:7.2f}  {r['phash']:7.2f}  {r['inliers']:>4}  "
            f"{r['good']:>5}  {r['n_new']:>5}/{r['n_coll']:<5}  {flag}"
        )

    # ---------------- CSV ----------------
    if args.csv:
        fieldnames = ["name", "total", "sift", "color", "phash",
                      "inliers", "good", "n_new", "n_coll"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in r.items()
                })
        print(f"\nCSV-отчёт сохранён: {args.csv}")

    # ---------------- визуализация top-1 ----------------
    if args.visualize and best is not None:
        total, path, kp2, extra, img2 = best
        if extra is not None and kp_new is not None and kp2 is not None:
            good, mask = extra
            inlier_matches = [m for m, ok in zip(good, mask.ravel().tolist()) if ok]
            vis = cv2.drawMatches(
                img_new, kp_new, img2, kp2, inlier_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(120, 120, 120),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            out = Path("top1_matches.jpg")
            cv2.imwrite(str(out), vis)
            print(f"Визуализация лучшего совпадения: {out}  "
                  f"({path.name}, Total={total:.2f}%)")


if __name__ == "__main__":
    main()