#!/usr/bin/env python3
"""
compare_trademarks_v04.py — сравнение товарного знака с коллекцией.

Новое в v05:
  • параллельная обработка (concurrent.futures.ProcessPoolExecutor)
  • кэш дескрипторов SIFT/ORB + pHash (по mtime файла)
  • экспорт HTML-отчёта с таблицей и (опционально) визуализацией top-1

Метрики (взвешенная сумма):
  SIFT%   — SIFT + Lowe ratio + RANSAC
  ORB%    — ORB + Lowe ratio
  Color%  — HSV-гистограммы
  pHash%  — perceptual hash 16×16
  SSIM%   — structural similarity после выравнивания

Примеры:
    python compare_trademarks_v04.py
    python compare_trademarks_v04.py --new logo.jpg --collections base \\
        --csv report.csv --html report.html --visualize --workers 8
    python compare_trademarks_v04.py --cache --cache-file .tm_cache.pkl --top 20
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_WEIGHTS = (0.40, 0.15, 0.20, 0.10, 0.15)  # SIFT ORB Color pHash SSIM


# ---------------------------------------------------------------------------
# Утилиты загрузки / предобработки
# ---------------------------------------------------------------------------

def load_bgr(path: Path, max_dim: int) -> Optional[np.ndarray]:
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


def preprocess(bgr: np.ndarray, clahe_clip: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    bgr2 = cv2.GaussianBlur(bgr2, (3, 3), 0)
    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    return bgr2, gray


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def phash(bgr: np.ndarray, hash_size: int = 16, highfreq: int = 4) -> np.ndarray:
    size = hash_size * highfreq
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])
    return (low > med).flatten()


def phash_score(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1.size != h2.size:
        return 0.0
    diff = int(np.count_nonzero(h1 != h2))
    return max(0.0, 100.0 * (1.0 - diff / h1.size))


def color_score(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0, 1], None, [36, 32], [0, 180, 0, 256])
    h2 = cv2.calcHist([hsv2], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    dist = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(100.0, 100.0 * (1.0 - dist)))


def ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return float(np.clip(ssim_map.mean() * 100.0, 0.0, 100.0))


def sift_match(
    kp1, des1, gray2: np.ndarray,
    sift, flann,
    ratio: float, ransac_thresh: float, min_inliers: int,
) -> Tuple[float, int, int, int, int, Any, Any, Optional[Tuple]]:
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
    denom = max(1, min(n1, n2))
    raw = inliers / denom
    conf = min(1.0, inliers / max(1, min_inliers))
    purity = inliers / max(1, len(good))
    score = 100.0 * raw * conf * (0.7 + 0.3 * purity)
    return score, len(good), inliers, n1, n2, kp2, des2, (good, mask, H)


def orb_score_from_des(des1, des2, ratio: float = 0.75) -> float:
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if not good:
        return 0.0
    score = (2.0 * len(good) / (len(des1) + len(des2))) * 100.0
    return min(100.0, score)


# ---------------------------------------------------------------------------
# Кэш дескрипторов
# ---------------------------------------------------------------------------

@dataclass
class CachedItem:
    mtime: float
    shape: Tuple[int, int]
    sift_des: Optional[np.ndarray] = None
    sift_kp_pts: Optional[np.ndarray] = None   # только координаты (для RANSAC)
    orb_des: Optional[np.ndarray] = None
    phash: Optional[np.ndarray] = None
    # цвет и SSIM считаем on-the-fly (нужен BGR)


def file_key(path: Path) -> str:
    return str(path.resolve())


def load_cache(cache_path: Path) -> Dict[str, CachedItem]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        # восстанавливаем dataclass
        out: Dict[str, CachedItem] = {}
        for k, v in raw.items():
            if isinstance(v, CachedItem):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = CachedItem(**v)
        return out
    except Exception as e:
        print(f"Предупреждение: кэш повреждён ({e}), начинаем с нуля", file=sys.stderr)
        return {}


def save_cache(cache_path: Path, data: Dict[str, CachedItem]) -> None:
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Предупреждение: не удалось сохранить кэш ({e})", file=sys.stderr)


def compute_descriptors(
    path: Path,
    max_dim: int,
    clahe_clip: float,
    sift_features: int,
    orb_features: int,
) -> Optional[CachedItem]:
    """Вычисляет и возвращает CachedItem для одного файла (вызывается в воркере)."""
    img_raw = load_bgr(path, max_dim)
    if img_raw is None:
        return None
    if clahe_clip > 0:
        bgr, gray = preprocess(img_raw, clahe_clip)
    else:
        bgr = img_raw
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=sift_features)
    orb = cv2.ORB_create(nfeatures=orb_features)

    kp, des = sift.detectAndCompute(gray, None)
    kp_pts = np.float32([k.pt for k in kp]) if kp else None

    _, orb_des = orb.detectAndCompute(gray, None)
    h = phash(bgr)

    return CachedItem(
        mtime=path.stat().st_mtime,
        shape=gray.shape[:2],
        sift_des=des,
        sift_kp_pts=kp_pts,
        orb_des=orb_des,
        phash=h,
    )


# ---------------------------------------------------------------------------
# Рабочая функция для одного изображения (для ProcessPool)
# ---------------------------------------------------------------------------

def process_one(
    args_tuple: Tuple,
) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает одно изображение коллекции.
    Принимает кортеж, чтобы быть picklable.
    """
    (
        path_str,
        new_des_sift,
        new_kp_pts,
        new_des_orb,
        new_phash,
        new_bgr,          # для color + ssim
        new_gray,
        max_dim,
        clahe_clip,
        ratio,
        ransac_thresh,
        min_inliers,
        sift_features,
        orb_features,
        weights,          # уже нормализованные
        cached_item,      # CachedItem или None
    ) = args_tuple

    path = Path(path_str)

    # --- дескрипторы ---
    if cached_item is not None and cached_item.mtime == path.stat().st_mtime:
        des2 = cached_item.sift_des
        kp2_pts = cached_item.sift_kp_pts
        orb_des2 = cached_item.orb_des
        h2 = cached_item.phash
        # BGR всё равно нужен для цвета
        img_raw = load_bgr(path, max_dim)
        if img_raw is None:
            return None
        if clahe_clip > 0:
            img2, gray2 = preprocess(img_raw, clahe_clip)
        else:
            img2 = img_raw
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        n2 = len(kp2_pts) if kp2_pts is not None else 0
    else:
        img_raw = load_bgr(path, max_dim)
        if img_raw is None:
            return None
        if clahe_clip > 0:
            img2, gray2 = preprocess(img_raw, clahe_clip)
        else:
            img2 = img_raw
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=sift_features)
        orb = cv2.ORB_create(nfeatures=orb_features)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        kp2_pts = np.float32([k.pt for k in kp2]) if kp2 else None
        _, orb_des2 = orb.detectAndCompute(gray2, None)
        h2 = phash(img2)
        n2 = len(kp2) if kp2 else 0

        # обновим кэш-запись (вернём вместе с результатом)
        cached_item = CachedItem(
            mtime=path.stat().st_mtime,
            shape=gray2.shape[:2],
            sift_des=des2,
            sift_kp_pts=kp2_pts,
            orb_des=orb_des2,
            phash=h2,
        )

    n1 = len(new_kp_pts) if new_kp_pts is not None else 0

    # --- SIFT + RANSAC ---
    s_score, good_n, inl = 0.0, 0, 0
    extra = None
    if (
        new_des_sift is not None
        and des2 is not None
        and len(new_des_sift) >= 2
        and len(des2) >= 2
        and new_kp_pts is not None
        and kp2_pts is not None
    ):
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )
        matches = flann.knnMatch(new_des_sift, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) >= 4:
            src = np.float32([new_kp_pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2_pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if H is not None and mask is not None:
                inl = int(mask.sum())
                good_n = len(good)
                denom = max(1, min(n1, n2))
                raw = inl / denom
                conf = min(1.0, inl / max(1, min_inliers))
                purity = inl / max(1, good_n)
                s_score = 100.0 * raw * conf * (0.7 + 0.3 * purity)
                extra = (good, mask, H)

    # --- ORB ---
    o_score = orb_score_from_des(new_des_orb, orb_des2, ratio)

    # --- Color ---
    c_score = color_score(new_bgr, img2)

    # --- pHash ---
    p_score = phash_score(new_phash, h2)

    # --- SSIM ---
    ssim_val = 0.0
    if extra is not None:
        _, _, H = extra
        h, w_img = new_gray.shape[:2]
        try:
            aligned = cv2.warpPerspective(
                gray2, np.linalg.inv(H), (w_img, h)
            )
            mask_img = cv2.warpPerspective(
                np.ones_like(gray2, dtype=np.uint8) * 255,
                np.linalg.inv(H),
                (w_img, h),
            )
            if mask_img.sum() > 1000:
                ys, xs = np.where(mask_img > 0)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                ssim_val = ssim_score(
                    new_gray[y0:y1, x0:x1],
                    aligned[y0:y1, x0:x1],
                )
        except Exception:
            ssim_val = 0.0

    total = (
        weights[0] * s_score
        + weights[1] * o_score
        + weights[2] * c_score
        + weights[3] * p_score
        + weights[4] * ssim_val
    )

    return {
        "name": path.name,
        "path": str(path),
        "total": total,
        "sift": s_score,
        "orb": o_score,
        "color": c_score,
        "phash": p_score,
        "ssim": ssim_val,
        "inliers": inl,
        "good": good_n,
        "n_new": n1,
        "n_coll": n2,
        "cached_item": cached_item,  # для обновления кэша
        "extra": extra,              # для визуализации (только top-1)
        "img2": img2 if extra is not None else None,
        "kp2_pts": kp2_pts if extra is not None else None,
    }


# ---------------------------------------------------------------------------
# HTML-отчёт
# ---------------------------------------------------------------------------

def make_html_report(
    rows: List[Dict[str, Any]],
    new_path: Path,
    threshold: float,
    top1_vis_path: Optional[Path],
    out_path: Path,
    elapsed: float,
) -> None:
    def bar(val: float, color: str = "#4CAF50") -> str:
        w = max(0, min(100, val))
        return (
            f'<div style="background:#eee;border-radius:3px;height:14px;width:100px;display:inline-block;">'
            f'<div style="background:{color};width:{w:.0f}%;height:100%;border-radius:3px;"></div></div>'
            f' <span style="font-size:12px;">{val:.1f}%</span>'
        )

    rows_html = []
    for i, r in enumerate(rows, 1):
        flag = (
            '<span style="color:#2e7d32;font-weight:bold;">YES</span>'
            if r["total"] >= threshold
            else ""
        )
        bg = "#e8f5e9" if r["total"] >= threshold else ("#fff" if i % 2 else "#fafafa")
        rows_html.append(
            f"""
            <tr style="background:{bg};">
              <td style="text-align:right;padding:6px 8px;">{i}</td>
              <td style="padding:6px 8px;font-family:monospace;">{r['name']}</td>
              <td style="padding:6px 8px;">{bar(r['total'], '#1976d2')}</td>
              <td style="padding:6px 8px;">{bar(r['sift'])}</td>
              <td style="padding:6px 8px;">{bar(r['orb'], '#7b1fa2')}</td>
              <td style="padding:6px 8px;">{bar(r['color'], '#f57c00')}</td>
              <td style="padding:6px 8px;">{bar(r['phash'], '#00838f')}</td>
              <td style="padding:6px 8px;">{bar(r['ssim'], '#c62828')}</td>
              <td style="text-align:right;padding:6px 8px;">{r['inliers']}</td>
              <td style="text-align:right;padding:6px 8px;">{r['good']}</td>
              <td style="text-align:center;padding:6px 8px;">{r['n_new']}/{r['n_coll']}</td>
              <td style="text-align:center;padding:6px 8px;">{flag}</td>
            </tr>
            """
        )

    vis_block = ""
    if top1_vis_path and top1_vis_path.exists():
        with open(top1_vis_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        vis_block = f"""
        <h2>Лучшее совпадение (SIFT inliers)</h2>
        <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;border:1px solid #ccc;border-radius:4px;"/>
        """

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>Отчёт сравнения товарных знаков</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 24px; background: #f5f5f5; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); border-radius: 6px; overflow: hidden; }}
  th {{ background: #1565c0; color: #fff; padding: 10px 8px; text-align: left;
       font-weight: 600; font-size: 13px; }}
  td {{ font-size: 13px; }}
  tr:hover td {{ background: #e3f2fd !important; }}
</style>
</head>
<body>
  <h1>Сравнение товарных знаков</h1>
  <div class="meta">
    Эталон: <b>{new_path.name}</b> &nbsp;|&nbsp;
    Обработано: <b>{len(rows)}</b> файлов &nbsp;|&nbsp;
    Время: <b>{elapsed:.1f} с</b> &nbsp;|&nbsp;
    Порог Match: <b>{threshold}%</b>
  </div>

  {vis_block}

  <h2>Результаты (сортировка по Total %)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Файл</th>
        <th>Total %</th>
        <th>SIFT %</th>
        <th>ORB %</th>
        <th>Color %</th>
        <th>pHash %</th>
        <th>SSIM %</th>
        <th>Inliers</th>
        <th>Good</th>
        <th>KP n/c</th>
        <th>Match</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML-отчёт сохранён: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--new", default="new.jpg", help="эталонное изображение")
    p.add_argument("--collections", default="collections", help="папка с коллекцией")
    p.add_argument("--csv", default=None, help="сохранить отчёт в CSV")
    p.add_argument("--html", default=None, help="сохранить HTML-отчёт")
    p.add_argument("--threshold", type=float, default=25.0,
                   help="порог Total%% для пометки YES")
    p.add_argument("--weights", nargs=5, type=float, default=DEFAULT_WEIGHTS,
                   metavar=("SIFT", "ORB", "COLOR", "PHASH", "SSIM"))
    p.add_argument("--max-dim", type=int, default=1200)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--ransac", type=float, default=5.0)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--sift-features", type=int, default=4000)
    p.add_argument("--orb-features", type=int, default=2000)
    p.add_argument("--top", type=int, default=0, help="0 = все")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--cache", action="store_true", help="использовать/обновлять кэш")
    p.add_argument("--cache-file", default=".tm_cache.pkl")
    p.add_argument("--clahe", type=float, default=2.0, help="0 = отключить")
    p.add_argument("--workers", type=int, default=0,
                   help="число процессов (0 = cpu_count)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.perf_counter()

    new_path = Path(args.new)
    coll_dir = Path(args.collections)
    if not new_path.exists():
        sys.exit(f"Не найден файл: {new_path}")
    if not coll_dir.is_dir():
        sys.exit(f"Не найдена папка: {coll_dir}")

    # --- эталон ---
    img_new_raw = load_bgr(new_path, args.max_dim)
    if img_new_raw is None:
        sys.exit(f"Не удалось открыть: {new_path}")

    if args.clahe > 0:
        img_new, gray_new = preprocess(img_new_raw, args.clahe)
    else:
        img_new = img_new_raw
        gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    hash_new = phash(img_new)

    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    orb = cv2.ORB_create(nfeatures=args.orb_features)

    kp_new, des_new = sift.detectAndCompute(gray_new, None)
    kp_new_pts = np.float32([k.pt for k in kp_new]) if kp_new else None
    _, des_orb_new = orb.detectAndCompute(gray_new, None)

    n_kp_new = len(kp_new) if kp_new else 0
    if n_kp_new < 10:
        print(f"Внимание: на {new_path.name} найдено мало SIFT-признаков "
              f"({n_kp_new}); SIFT-метрика может быть ненадёжна.")

    # --- кэш ---
    cache: Dict[str, CachedItem] = {}
    cache_path = Path(args.cache_file)
    if args.cache:
        cache = load_cache(cache_path)
        print(f"Кэш загружен: {len(cache)} записей")

    # --- веса ---
    w = np.array(args.weights, dtype=np.float64)
    w_sum = max(1e-9, w.sum())
    w /= w_sum

    files = sorted(
        p for p in coll_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not files:
        sys.exit(f"В папке {coll_dir} нет изображений")

    # --- подготовка задач ---
    tasks = []
    for path in files:
        key = file_key(path)
        cached = cache.get(key)
        # проверяем актуальность
        if cached is not None:
            try:
                if abs(cached.mtime - path.stat().st_mtime) > 1e-3:
                    cached = None
            except OSError:
                cached = None

        tasks.append((
            str(path),
            des_new,
            kp_new_pts,
            des_orb_new,
            hash_new,
            img_new,
            gray_new,
            args.max_dim,
            args.clahe,
            args.ratio,
            args.ransac,
            args.min_inliers,
            args.sift_features,
            args.orb_features,
            w.tolist(),
            cached,
        ))

    # --- параллельный запуск ---
    workers = args.workers if args.workers > 0 else None  # None → cpu_count
    rows: List[Dict[str, Any]] = []
    updated_cache: Dict[str, CachedItem] = dict(cache)

    print(f"Запуск {len(tasks)} задач (workers={workers or 'auto'})...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            path_str = futures[fut]
            print(f"\r[{done}/{len(tasks)}] {Path(path_str).name[:50]:<50}", end="", flush=True)
            try:
                res = fut.result()
            except Exception as e:
                print(f"\n  ошибка {Path(path_str).name}: {e}")
                continue
            if res is None:
                continue

            # обновляем кэш
            if args.cache and res.get("cached_item") is not None:
                updated_cache[file_key(Path(res["path"]))] = res["cached_item"]

            # убираем тяжёлые поля перед сохранением в rows
            row = {k: v for k, v in res.items()
                   if k not in ("cached_item", "extra", "img2", "kp2_pts")}
            rows.append(row)

            # сохраняем extra/img2 только для потенциального top-1 (позже отфильтруем)
            row["_extra"] = res.get("extra")
            row["_img2"] = res.get("img2")
            row["_kp2_pts"] = res.get("kp2_pts")

    print()

    rows.sort(key=lambda r: r["total"], reverse=True)

    # --- консольный вывод ---
    header = (
        f"{'#':>3}  {'Файл':36}  {'Total%':>7}  {'SIFT%':>6}  {'ORB%':>5}  "
        f"{'Color%':>7}  {'pHash%':>7}  {'SSIM%':>6}  {'Inl':>4}  {'Good':>5}  "
        f"{'KPn/KPc':>11}  Match"
    )
    print(header)
    print("-" * len(header))

    shown = rows if args.top <= 0 else rows[: args.top]
    for i, r in enumerate(shown, 1):
        flag = "YES" if r["total"] >= args.threshold else ""
        print(
            f"{i:>3}  {r['name']:36}  {r['total']:7.2f}  {r['sift']:6.2f}  "
            f"{r['orb']:5.2f}  {r['color']:7.2f}  {r['phash']:7.2f}  "
            f"{r['ssim']:6.2f}  {r['inliers']:>4}  {r['good']:>5}  "
            f"{r['n_new']:>5}/{r['n_coll']:<5}  {flag}"
        )

    # --- CSV ---
    if args.csv:
        fieldnames = [
            "name", "total", "sift", "orb", "color", "phash", "ssim",
            "inliers", "good", "n_new", "n_coll",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in r.items() if k in fieldnames
                })
        print(f"\nCSV-отчёт сохранён: {args.csv}")

    # --- визуализация top-1 ---
    top1_vis_path = None
    if args.visualize and rows:
        best = rows[0]
        extra = best.get("_extra")
        img2 = best.get("_img2")
        kp2_pts = best.get("_kp2_pts")
        if extra is not None and kp_new is not None and kp2_pts is not None and img2 is not None:
            good, mask, _ = extra
            # восстанавливаем KeyPoint-объекты только для отрисовки
            kp2_list = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in kp2_pts]
            inlier_matches = [
                m for m, ok in zip(good, mask.ravel().tolist()) if ok
            ]
            vis = cv2.drawMatches(
                img_new, kp_new, img2, kp2_list, inlier_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(120, 120, 120),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            top1_vis_path = Path("top1_matches.jpg")
            cv2.imwrite(str(top1_vis_path), vis)
            print(f"Визуализация лучшего совпадения: {top1_vis_path}  "
                  f"({best['name']}, Total={best['total']:.2f}%)")
        else:
            print("Визуализация невозможна: нет валидной гомографии у top-1.")

    # --- HTML ---
    if args.html:
        # очищаем служебные поля
        clean_rows = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in rows
        ]
        make_html_report(
            clean_rows,
            new_path,
            args.threshold,
            top1_vis_path,
            Path(args.html),
            time.perf_counter() - t0,
        )

    # --- сохранение кэша ---
    if args.cache:
        save_cache(cache_path, updated_cache)
        print(f"Кэш обновлён: {len(updated_cache)} записей → {cache_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nОбработано {len(rows)} изображений за {elapsed:.1f} с")


if __name__ == "__main__":
    # Для Windows/macOS spawn нужен guard
    main()