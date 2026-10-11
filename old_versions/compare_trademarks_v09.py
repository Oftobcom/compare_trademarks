#!/usr/bin/env python3
"""
compare_trademarks_v09.py — сравнение товарного знака с коллекцией изображений.

Ключевые отличия от v08
------------------------
  • Классические 6 метрик (SIFT, ORB, Color, pHash, SSIM, Shape) сохранены
    как в v07 — веса нормированы к 1.0, поэтому --threshold / --risk-*
    имеют ТО ЖЕ значение, что в v07, при --semantic-weight 0.
  • Семантика (CLIP/DINOv2) подключается отдельным весом --semantic-weight.
    Total% = (1 - semantic_weight) * classic_6 + semantic_weight * semantic_pct
    При --semantic-weight 0 (по умолчанию) семантика в Total% НЕ входит —
    только показывается отдельными колонками. Полная совместимость с v07.
  • Косинус CLIP/DINO переводится в проценты через сигмоиду с настраиваемым
    центром (--clip-neutral, --dino-neutral) и наклоном (--clip-scale,
    --dino-scale). Это даёт осмысленную шкалу 0..100 независимо от модели.
  • Убран мёртвый код, исправлены хрупкие проверки типов, восстановлен
    комментарий к task-кортежу.
  • HTML компактнее: 6 классических колонок + 3 семантических (скрываемые
    чекбоксом).

Метрики классические (6):
  SIFT%   — SIFT + Lowe ratio + RANSAC-гомография
  ORB%    — ORB + Lowe ratio
  Color%  — HSV-гистограмма (опционально с игнором белого)
  pHash%  — perceptual hash 16×16
  SSIM%   — structural similarity ПОСЛЕ выравнивания
  Shape%  — matchShapes + Hu-моменты, тоже ПОСЛЕ выравнивания

Метрики семантические (2):
  CLIP%   — CLIP image embedding cosine (сигмоида)
  DINO%   — DINOv2 image embedding cosine (сигмоида)
  Semantic% = 0.6*CLIP + 0.4*DINO (если оба доступны)

Зависимости
-----------
  pip install opencv-python numpy
  pip install torch transformers Pillow    # для CLIP/DINOv2
  pip install tqdm                          # опционально
  pip install open-clip-torch               # альтернатива для CLIP

  Если torch/transformers нет — скрипт работает как v07.

Примеры
-------
  # v07-совместимый запуск (семантика только в отчёте, не в Total%)
  python compare_trademarks_v09.py --new new.jpg --collections base \\
      --embedding both --cache --html report.html --visualize

  # Смешать семантику в Total% на 25%
  python compare_trademarks_v09.py --new new.jpg --collections base \\
      --embedding clip --semantic-weight 0.25 --threshold 30

  # Отключить семантику полностью
  python compare_trademarks_v09.py --embedding none   # поведение v07

  # Настроить сигмоиду для конкретной модели
  python compare_trademarks_v09.py --clip-model openai/clip-vit-large-patch14 \\
      --clip-neutral 0.45 --clip-scale 0.12 --embedding clip
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import pickle
import shutil
import sys
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ============================================================================
# Константы
# ============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Классические 6 весов (SIFT, ORB, Color, pHash, SSIM, Shape) — как в v07.
# Сумма = 1.0 → --threshold и --risk-* имеют тот же смысл, что в v07.
CLASSIC_WEIGHTS_V07: Tuple[float, ...] = (0.35, 0.10, 0.15, 0.10, 0.15, 0.15)

# Дефолтные параметры сигмоиды для семантики (см. semantic_to_percent).
# CLIP-B/32: unrelated ≈ 0.2..0.4, related ≈ 0.7..0.9 → neutral=0.50, scale=0.15
# DINOv2-base: unrelated ≈ 0.4..0.6, related ≈ 0.8..0.95 → neutral=0.65, scale=0.12
CLIP_NEUTRAL_DEFAULT = 0.50
CLIP_SCALE_DEFAULT = 0.15
DINO_NEUTRAL_DEFAULT = 0.65
DINO_SCALE_DEFAULT = 0.12

SCRIPT_BASENAME = "compare_trademarks"
SCRIPT_VERSION = "v09"
LOG_DIR_DEFAULT = "logs"
CACHE_VERSION = 2
EMB_CACHE_VERSION = 1
RISK_HIGH_DEFAULT = 60.0
RISK_MEDIUM_DEFAULT = 30.0
WHITE_THRESH_DEFAULT = 240


# ============================================================================
# Tee-логгер
# ============================================================================

class Tee:
    """Дублирует write/flush во все переданные потоки."""

    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee: нужен хотя бы один поток")
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        if "\n" in data:
            self.flush()

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        return self.streams[0].fileno()


_WORKER_LOG_FILE = None


def _init_worker_logging(log_path: str) -> None:
    global _WORKER_LOG_FILE
    if not log_path:
        return
    if isinstance(sys.stdout, Tee):
        return
    try:
        _WORKER_LOG_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, _WORKER_LOG_FILE)
        sys.stderr = Tee(sys.stderr, _WORKER_LOG_FILE)
    except Exception as e:
        print(f"[WARN] worker logging disabled: {e}", file=sys.__stdout__)


def _resolve_log_path(explicit: Optional[str],
                      flat: bool,
                      no_log: bool,
                      log_dir: str = LOG_DIR_DEFAULT) -> Optional[Path]:
    if no_log:
        return None
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if flat:
        return out_dir / f"{SCRIPT_BASENAME}.log"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{SCRIPT_BASENAME}_{ts}.log"


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(SCRIPT_BASENAME)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)  # sys.stdout уже Tee
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ============================================================================
# Загрузка и предобработка
# ============================================================================

def load_bgr_smart(path: Path, max_dim: int) -> Optional[np.ndarray]:
    """BGR-чтение с поддержкой alpha (композит на белый) и grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        white = np.full_like(b, 255, dtype=np.uint8)
        alpha = a.astype(np.float32) / 255.0
        b = (b.astype(np.float32) * alpha
             + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha
             + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha
             + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        img = cv2.merge([b, g, r])
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img,
                         (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def bgr_to_pil(bgr: np.ndarray) -> Optional["Image.Image"]:
    if not _HAS_PIL:
        return None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None


def preprocess(bgr: np.ndarray, clahe_clip: float) -> Tuple[np.ndarray, np.ndarray]:
    if clahe_clip <= 0:
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    bgr2 = cv2.GaussianBlur(bgr2, (3, 3), 0)
    return bgr2, cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)


# ============================================================================
# Классические метрики (идентичны v07)
# ============================================================================

def phash(bgr: np.ndarray, hash_size: int = 16, hf: int = 4) -> np.ndarray:
    size = hash_size * hf
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])
    return (low > med).flatten()


def phash_score(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1 is None or h2 is None or h1.size != h2.size or h1.size == 0:
        return 0.0
    diff = int(np.count_nonzero(h1 != h2))
    return max(0.0, 100.0 * (1.0 - diff / h1.size))


def color_score(bgr1: np.ndarray, bgr2: np.ndarray,
                ignore_white: bool = False,
                white_thresh: int = WHITE_THRESH_DEFAULT) -> float:
    hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)

    mask1 = mask2 = None
    if ignore_white:
        def not_white_mask(bgr):
            lo = np.array([white_thresh] * 3, dtype=np.uint8)
            hi = np.array([255, 255, 255], dtype=np.uint8)
            return cv2.bitwise_not(cv2.inRange(bgr, lo, hi))

        mask1 = not_white_mask(bgr1)
        mask2 = not_white_mask(bgr2)
        if int((mask1 > 0).sum()) < bgr1.shape[0] * bgr1.shape[1] * 0.05:
            mask1 = None
        if int((mask2 > 0).sum()) < bgr2.shape[0] * bgr2.shape[1] * 0.05:
            mask2 = None

    h1 = cv2.calcHist([hsv1], [0, 1], mask1, [36, 32], [0, 180, 0, 256])
    h2 = cv2.calcHist([hsv2], [0, 1], mask2, [36, 32], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    dist = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(100.0, 100.0 * (1.0 - dist)))


def ssim_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    if gray1 is None or gray2 is None or gray1.size == 0 or gray2.size == 0:
        return 0.0
    if gray1.shape != gray2.shape:
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        if h < 8 or w < 8:
            return 0.0
        gray1 = cv2.resize(gray1, (w, h), interpolation=cv2.INTER_AREA)
        gray2 = cv2.resize(gray2, (w, h), interpolation=cv2.INTER_AREA)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = gray1.astype(np.float64)
    b = gray2.astype(np.float64)

    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sig2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sig12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2) + 1e-12
    return float(np.clip((num / den).mean() * 100.0, 0.0, 100.0))


def shape_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """Ожидает УЖЕ ВЫРОВНЕННЫЕ изображения (после warpPerspective и маски)."""
    try:
        _, bin1 = cv2.threshold(gray1, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(gray2, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def largest_contour(bin_img):
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            area_thresh = bin_img.shape[0] * bin_img.shape[1] * 0.01
            big = [c for c in contours if cv2.contourArea(c) > area_thresh]
            if not big:
                big = contours
            return max(big, key=cv2.contourArea)

        c1 = largest_contour(bin1)
        c2 = largest_contour(bin2)
        if c1 is None or c2 is None:
            return 0.0

        dist = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
        shape_match = 100.0 * float(np.exp(-dist * 2.0))

        m1 = cv2.moments(c1)
        m2 = cv2.moments(c2)
        hu1 = cv2.HuMoments(m1).flatten()
        hu2 = cv2.HuMoments(m2).flatten()

        def log_hu(h):
            return -np.sign(h) * np.log10(np.abs(h) + 1e-12)

        hu_dist = float(np.linalg.norm(log_hu(hu1) - log_hu(hu2)))
        hu_score = 100.0 * float(np.exp(-hu_dist * 0.5))
        return float(0.6 * shape_match + 0.4 * hu_score)
    except Exception:
        return 0.0


def orb_score_from_des(des1, des2, ratio: float = 0.75) -> float:
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    try:
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
        return min(100.0, (2.0 * len(good) / (len(des1) + len(des2))) * 100.0)
    except Exception:
        return 0.0


# ============================================================================
# Кэш дескрипторов
# ============================================================================

@dataclass
class CachedItem:
    mtime: float
    size: int
    shape: Tuple[int, int]
    version: int = CACHE_VERSION
    sift_des: Optional[np.ndarray] = None
    sift_kp_pts: Optional[np.ndarray] = None
    orb_des: Optional[np.ndarray] = None
    phash: Optional[np.ndarray] = None

    def is_valid(self, path: Path) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        return (self.version == CACHE_VERSION
                and abs(self.mtime - st.st_mtime) < 1e-3
                and self.size == st.st_size)


def file_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def load_cache(cache_path: Path) -> Dict[str, CachedItem]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        print(f"[WARN] кэш повреждён ({e})", file=sys.stderr)
        return {}

    out: Dict[str, CachedItem] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, CachedItem) and v.version == CACHE_VERSION:
            out[k] = v
        elif isinstance(v, dict):
            try:
                if "size" not in v:
                    v = dict(v, size=0)
                filtered = {kk: vv for kk, vv in v.items()
                            if kk in CachedItem.__dataclass_fields__}
                item = CachedItem(**filtered)
                if item.version == CACHE_VERSION:
                    out[k] = item
            except Exception:
                continue
    return out


def save_cache(cache_path: Path, data: Dict[str, CachedItem],
               max_entries: int = 5000) -> None:
    try:
        if len(data) > max_entries:
            items = sorted(data.items(), key=lambda kv: kv[1].mtime,
                           reverse=True)
            data = dict(items[:max_entries])
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except Exception as e:
        print(f"[WARN] не удалось сохранить кэш ({e})", file=sys.stderr)


# ============================================================================
# Кэш эмбеддингов
# ============================================================================

@dataclass
class EmbCachedItem:
    mtime: float
    size: int
    model: str
    version: int = EMB_CACHE_VERSION
    embedding: Optional[np.ndarray] = None

    def is_valid(self, path: Path) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        return (self.version == EMB_CACHE_VERSION
                and abs(self.mtime - st.st_mtime) < 1e-3
                and self.size == st.st_size)


def emb_file_key(path: Path, model_name: str) -> str:
    return f"{file_key(path)}|{model_name}"


def load_emb_cache(cache_path: Path) -> Dict[str, EmbCachedItem]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        out: Dict[str, EmbCachedItem] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            if isinstance(v, EmbCachedItem):
                out[k] = v
            elif isinstance(v, dict) and v.get("embedding") is not None:
                try:
                    emb = np.asarray(v["embedding"], dtype=np.float32)
                    out[k] = EmbCachedItem(
                        mtime=float(v.get("mtime", 0.0)),
                        size=int(v.get("size", 0)),
                        model=str(v.get("model", "")),
                        version=int(v.get("version", EMB_CACHE_VERSION)),
                        embedding=emb,
                    )
                except Exception:
                    continue
        return out
    except Exception as e:
        print(f"[WARN] emb-кэш повреждён ({e})", file=sys.stderr)
        return {}


def save_emb_cache(cache_path: Path, data: Dict[str, EmbCachedItem],
                   max_entries: int = 10000) -> None:
    try:
        if len(data) > max_entries:
            items = sorted(data.items(), key=lambda kv: kv[1].mtime,
                           reverse=True)
            data = dict(items[:max_entries])
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except Exception as e:
        print(f"[WARN] не удалось сохранить emb-кэш ({e})", file=sys.stderr)


# ============================================================================
# Утилиты
# ============================================================================

def classify_risk(total: float,
                  high: float = RISK_HIGH_DEFAULT,
                  medium: float = RISK_MEDIUM_DEFAULT) -> str:
    if total >= high:
        return "HIGH"
    if total >= medium:
        return "MEDIUM"
    return "LOW"


def collect_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in root.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _img_to_base64_thumb(bgr: np.ndarray,
                         max_side: int = 200,
                         quality: int = 80) -> str:
    try:
        h, w = bgr.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return ("data:image/jpeg;base64,"
                + base64.b64encode(buf.tobytes()).decode("ascii"))
    except Exception:
        return ""


def cosine_similarity(a: Optional[np.ndarray],
                      b: Optional[np.ndarray]) -> float:
    """Чистый косинус. Возвращает 0.0, если что-то пустое/размерное."""
    if a is None or b is None:
        return 0.0
    try:
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def semantic_to_percent(cos: float, neutral: float, scale: float) -> float:
    """
    Сигмоида: cos=neutral → 50%, cos=neutral+scale → ~73%,
    cos=neutral+2*scale → ~88%. Диапазон (0..100).
    """
    try:
        s = max(float(scale), 1e-6)
        z = (float(cos) - float(neutral)) / s
        # защита от overflow в exp
        if z > 50:
            return 100.0
        if z < -50:
            return 0.0
        return 100.0 / (1.0 + float(np.exp(-z)))
    except Exception:
        return 0.0


# ============================================================================
# Эмбеддинг-менеджер (CLIP / DINOv2)
# ============================================================================

class EmbeddingManager:
    """
    Загружает CLIP и/или DINOv2 и считает эмбеддинги.
    Используется ТОЛЬКО в main-процессе (в воркеры передаются готовые скаляры).
    """

    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 dino_model_name: str = "facebook/dinov2-base",
                 device: str = "auto",
                 logger: Optional[logging.Logger] = None):
        self.clip_model_name = clip_model_name
        self.dino_model_name = dino_model_name
        self.logger = logger
        self.device_str = device
        self.device = self._resolve_device(device)
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_processor = None
        self._clip_is_hf = False
        self._dino_is_hub = False
        self._clip_available = False
        self._dino_available = False

    def _resolve_device(self, device_str: str):
        try:
            import torch
            if device_str == "cpu":
                return torch.device("cpu")
            if device_str == "cuda":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return None

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # ----- CLIP -----

    def load_clip(self) -> bool:
        if self._clip_available:
            return True

        # Попытка №1: transformers
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self._log(f"[EMB] Загрузка CLIP '{self.clip_model_name}' "
                      f"на {self.device} (transformers)...")
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_model.to(self.device)
            self.clip_model.eval()
            self._clip_is_hf = True
            self._clip_available = True
            self._log("[EMB] CLIP (transformers) загружен")
            return True
        except Exception as e_hf:
            self._log(f"[EMB] transformers CLIP не удался: {e_hf}. "
                      f"Пробую open_clip...")

        # Попытка №2: open_clip
        try:
            import torch
            import open_clip  # type: ignore
            name = self.clip_model_name
            if name.startswith("openai/"):
                name = name.split("/", 1)[1]
            model, _, preprocess = open_clip.create_model_and_transforms(
                name, pretrained="openai"
            )
            self.clip_model = model.to(self.device).eval()
            self.clip_processor = preprocess
            self._clip_is_hf = False
            self._clip_available = True
            self._log("[EMB] CLIP (open_clip) загружен")
            return True
        except Exception as e_open:
            self._log(f"[EMB] CLIP недоступен (transformers: {e_hf}; "
                      f"open_clip: {e_open})")
            return False

    def encode_clip_batch(self,
                          pil_images: List["Image.Image"]
                          ) -> List[Optional[np.ndarray]]:
        if not self._clip_available or self.clip_model is None:
            return [None] * len(pil_images)
        try:
            import torch
            if self._clip_is_hf:
                from transformers import CLIPProcessor
                if not isinstance(self.clip_processor, CLIPProcessor):
                    # на случай, если что-то подменилось
                    raise RuntimeError("clip_processor is not CLIPProcessor")
                inputs = self.clip_processor(images=pil_images,
                                             return_tensors="pt",
                                             padding=True)
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self.clip_model.get_image_features(**inputs)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                return [f.cpu().numpy().astype(np.float32) for f in feats]
            else:
                tensors = [self.clip_processor(img).unsqueeze(0)
                           for img in pil_images]
                batch = torch.cat(tensors, dim=0).to(self.device)
                with torch.no_grad():
                    feats = self.clip_model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return [f.cpu().numpy().astype(np.float32) for f in feats]
        except Exception as e:
            self._log(f"[EMB] encode_clip_batch error: {e}\n"
                      f"{traceback.format_exc()}")
            return [None] * len(pil_images)

    # ----- DINOv2 -----

    def load_dino(self) -> bool:
        if self._dino_available:
            return True

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self._log(f"[EMB] Загрузка DINOv2 '{self.dino_model_name}' "
                      f"на {self.device} (transformers)...")
            self.dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name)
            self.dino_model = AutoModel.from_pretrained(self.dino_model_name)
            self.dino_model.to(self.device)
            self.dino_model.eval()
            self._dino_is_hub = False
            self._dino_available = True
            self._log("[EMB] DINOv2 (transformers) загружен")
            return True
        except Exception as e_hf:
            self._log(f"[EMB] transformers DINOv2 не удался: {e_hf}. "
                      f"Пробую torch.hub...")

        try:
            import torch
            name = self.dino_model_name.split("/")[-1]
            if "dinov2" not in name:
                name = "dinov2_vitb14"
            self.dino_model = torch.hub.load("facebookresearch/dinov2", name)
            self.dino_model.to(self.device).eval()
            self.dino_processor = None
            self._dino_is_hub = True
            self._dino_available = True
            self._log(f"[EMB] DINOv2 (torch.hub '{name}') загружен")
            return True
        except Exception as e_hub:
            self._log(f"[EMB] DINOv2 недоступен (transformers: {e_hf}; "
                      f"hub: {e_hub})")
            return False

    def encode_dino_batch(self,
                          pil_images: List["Image.Image"]
                          ) -> List[Optional[np.ndarray]]:
        if not self._dino_available or self.dino_model is None:
            return [None] * len(pil_images)
        try:
            import torch
            if self.dino_processor is not None:
                inputs = self.dino_processor(images=pil_images,
                                             return_tensors="pt")
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
                with torch.no_grad():
                    outputs = self.dino_model(**inputs)
                    if hasattr(outputs, "last_hidden_state"):
                        # CLS-токен обычно информативнее mean-pool для DINOv2,
                        # но mean даёт более стабильный скор между доменами.
                        feats = outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        feats = outputs.pooler_output
                    else:
                        feats = outputs[0].mean(dim=1)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                return [f.cpu().numpy().astype(np.float32) for f in feats]
            else:
                # torch.hub DINOv2: свой препроцесс
                from torchvision import transforms  # type: ignore
                transform = transforms.Compose([
                    transforms.Resize(256, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
                ])
                batch = torch.stack([transform(img) for img in pil_images]).to(self.device)
                with torch.no_grad():
                    feats = self.dino_model(batch)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                return [f.cpu().numpy().astype(np.float32) for f in feats]
        except Exception as e:
            self._log(f"[EMB] encode_dino_batch error: {e}\n"
                      f"{traceback.format_exc()}")
            return [None] * len(pil_images)


# ============================================================================
# Worker: одна пара (эталон, файл коллекции)
# ============================================================================
#
# Порядок элементов кортежа task (НЕ МЕНЯТЬ!):
#   0  path_str
#   1  new_des_sift          2  new_kp_pts          3  new_des_orb
#   4  new_phash             5  new_bgr             6  new_gray
#   7  max_dim               8  clahe_clip          9  ratio
#   10 ransac_thresh         11 min_inliers         12 sift_features
#   13 orb_features          14 classic_weights_6   15 cached_item
#   16 ignore_white          17 white_thresh        18 risk_high
#   19 risk_medium           20 semantic_weight     21 semantic_score_pre
# ============================================================================

def process_one(task: Tuple) -> Optional[Dict[str, Any]]:
    (
        path_str,
        new_des_sift,
        new_kp_pts,
        new_des_orb,
        new_phash,
        new_bgr,
        new_gray,
        max_dim,
        clahe_clip,
        ratio,
        ransac_thresh,
        min_inliers,
        sift_features,
        orb_features,
        classic_weights,      # 6 значений, сумма = 1.0
        cached_item,
        ignore_white,
        white_thresh,
        risk_high,
        risk_medium,
        semantic_weight,      # ∈ [0, 1]
        semantic_score_pre,   # ∈ [0, 100] | None
    ) = task

    path = Path(path_str)
    try:
        st = path.stat()
        mtime_now = st.st_mtime
        size_now = st.st_size
    except OSError:
        return None

    # ----- дескрипторы -----
    if cached_item is not None and cached_item.is_valid(path):
        des2 = cached_item.sift_des
        kp2_pts = cached_item.sift_kp_pts
        orb_des2 = cached_item.orb_des
        h2 = cached_item.phash
        img_raw = load_bgr_smart(path, max_dim)
        if img_raw is None:
            return None
        img2, gray2 = preprocess(img_raw, clahe_clip)
        n2 = 0 if kp2_pts is None else len(kp2_pts)
        updated_cache_item = None
    else:
        img_raw = load_bgr_smart(path, max_dim)
        if img_raw is None:
            return None
        img2, gray2 = preprocess(img_raw, clahe_clip)
        sift = cv2.SIFT_create(nfeatures=sift_features)
        orb = cv2.ORB_create(nfeatures=orb_features)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        kp2_pts = np.float32([k.pt for k in kp2]) if kp2 else None
        _, orb_des2 = orb.detectAndCompute(gray2, None)
        h2 = phash(img2)
        n2 = 0 if kp2 is None else len(kp2)
        updated_cache_item = CachedItem(
            mtime=mtime_now, size=size_now, shape=gray2.shape[:2],
            version=CACHE_VERSION, sift_des=des2, sift_kp_pts=kp2_pts,
            orb_des=orb_des2, phash=h2,
        )

    n1 = 0 if new_kp_pts is None else len(new_kp_pts)

    # ----- SIFT + Lowe + RANSAC -----
    s_score, good_n, inl, extra = 0.0, 0, 0, None
    if (new_des_sift is not None and des2 is not None
            and len(new_des_sift) >= 2 and len(des2) >= 2
            and new_kp_pts is not None and kp2_pts is not None):
        try:
            flann = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
            matches = flann.knnMatch(new_des_sift, des2, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n_ = pair
                if m.distance < ratio * n_.distance:
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
        except Exception:
            s_score = 0.0

    # ----- ORB, Color, pHash -----
    o_score = orb_score_from_des(new_des_orb, orb_des2, ratio)
    c_score = color_score(new_bgr, img2,
                          ignore_white=ignore_white,
                          white_thresh=white_thresh)
    p_score = phash_score(new_phash, h2)

    # ----- SSIM + Shape (только после выравнивания) -----
    ssim_val = 0.0
    shape_val = 0.0
    if extra is not None:
        _, _, H = extra
        h_new, w_new = new_gray.shape[:2]
        try:
            inv_H = np.linalg.inv(H)
            aligned = cv2.warpPerspective(gray2, inv_H, (w_new, h_new))
            mask_img = cv2.warpPerspective(
                np.full_like(gray2, 255, dtype=np.uint8),
                inv_H, (w_new, h_new),
            )
            if int(mask_img.sum()) > 1000:
                ys, xs = np.where(mask_img > 0)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1

                new_crop = new_gray[y0:y1, x0:x1]
                aligned_crop = aligned[y0:y1, x0:x1]
                ssim_val = ssim_score(new_crop, aligned_crop)

                mask_crop = mask_img[y0:y1, x0:x1]
                valid = mask_crop > 0
                if valid.any():
                    new_for_shape = new_crop.copy()
                    aligned_for_shape = aligned_crop.copy()
                    new_for_shape[~valid] = 255
                    aligned_for_shape[~valid] = 255
                    shape_val = shape_score(new_for_shape, aligned_for_shape)
        except Exception:
            ssim_val = 0.0
            shape_val = 0.0

    # ----- классический итог 6 метрик ∈ [0, 100] -----
    classic_total = (
        classic_weights[0] * s_score
        + classic_weights[1] * o_score
        + classic_weights[2] * c_score
        + classic_weights[3] * p_score
        + classic_weights[4] * ssim_val
        + classic_weights[5] * shape_val
    )

    # ----- семантика -----
    sem = float(semantic_score_pre) if semantic_score_pre is not None else None

    # ----- итоговый Total% -----
    if sem is None or semantic_weight <= 0.0:
        total = classic_total
    else:
        w_sem = max(0.0, min(1.0, float(semantic_weight)))
        total = (1.0 - w_sem) * classic_total + w_sem * sem

    risk = classify_risk(total, risk_high, risk_medium)

    return {
        "name": path.name,
        "path": str(path),
        "total": float(total),
        "classic_total": float(classic_total),
        "sift": float(s_score),
        "orb": float(o_score),
        "color": float(c_score),
        "phash": float(p_score),
        "ssim": float(ssim_val),
        "shape": float(shape_val),
        "semantic": (None if sem is None else float(sem)),
        "inliers": int(inl),
        "good": int(good_n),
        "n_new": int(n1),
        "n_coll": int(n2),
        "risk": risk,
        "cached_item": updated_cache_item,
        # только для визуализации (не идёт в отчёты)
        "_extra": extra,
        "_img2": img2 if extra is not None else None,
        "_kp2_pts": kp2_pts if extra is not None else None,
    }


# ============================================================================
# HTML-отчёт v09
# ============================================================================

def make_html_report(rows: List[Dict[str, Any]],
                     new_path: Path,
                     new_bgr: np.ndarray,
                     threshold: float,
                     top_vis_paths: List[Tuple[Path, str, float]],
                     out_path: Path,
                     elapsed: float,
                     classic_weights: Tuple[float, ...],
                     semantic_weight: float,
                     embedding_info: str) -> None:
    new_thumb_b64 = _img_to_base64_thumb(new_bgr, 300)

    def bar(val: Optional[float], color: str = "#4CAF50") -> str:
        if val is None:
            return '<span style="color:#999;">—</span>'
        w = max(0.0, min(100.0, float(val)))
        return (
            '<div style="background:#eee;border-radius:3px;height:14px;'
            'width:70px;display:inline-block;vertical-align:middle;">'
            f'<div style="background:{color};width:{w:.0f}%;height:100%;'
            'border-radius:3px;"></div></div>'
            f' <span style="font-size:11px;">{w:.1f}%</span>'
        )

    risk_colors = {"HIGH": "#c62828", "MEDIUM": "#ef6c00", "LOW": "#2e7d32"}
    rows_html = []
    for i, r in enumerate(rows, 1):
        risk = r["risk"]
        rc = risk_colors.get(risk, "#666")
        risk_badge = (
            f'<span style="background:{rc};color:#fff;padding:2px 6px;'
            f'border-radius:10px;font-size:11px;">{risk}</span>'
        )
        flag = ('<span style="color:#2e7d32;font-weight:bold;">YES</span>'
                if r["total"] >= threshold else "")
        bg = ("#ffebee" if risk == "HIGH"
              else ("#fff3e0" if risk == "MEDIUM"
                    else ("#fff" if i % 2 else "#fafafa")))
        sem_disp = r.get("semantic")
        rows_html.append(
            f'<tr style="background:{bg};" data-total="{r["total"]:.2f}" '
            f'data-name="{r["name"]}">'
            f'<td style="text-align:right;padding:5px 6px;">{i}</td>'
            f'<td style="padding:5px 6px;font-family:monospace;max-width:200px;'
            f'overflow:hidden;text-overflow:ellipsis;" title="{r["name"]}">{r["name"]}</td>'
            f'<td style="padding:5px 6px;">{bar(r["total"], "#1976d2")}</td>'
            f'<td style="padding:5px 6px;">{bar(r["sift"])}</td>'
            f'<td style="padding:5px 6px;">{bar(r["orb"], "#7b1fa2")}</td>'
            f'<td style="padding:5px 6px;">{bar(r["color"], "#f57c00")}</td>'
            f'<td style="padding:5px 6px;">{bar(r["phash"], "#00838f")}</td>'
            f'<td style="padding:5px 6px;">{bar(r["ssim"], "#c62828")}</td>'
            f'<td style="padding:5px 6px;">{bar(r["shape"], "#33691e")}</td>'
            f'<td class="sem-col" style="padding:5px 6px;">{bar(sem_disp, "#00897b")}</td>'
            f'<td style="text-align:right;padding:5px 6px;">{r["inliers"]}</td>'
            f'<td style="text-align:center;padding:5px 6px;">{risk_badge}</td>'
            f'<td style="text-align:center;padding:5px 6px;">{flag}</td>'
            f'</tr>'
        )

    # галерея top-K
    gallery_items = []
    for vis_path, name, total in top_vis_paths:
        if not vis_path.exists():
            continue
        try:
            with open(vis_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            gallery_items.append(
                f'<div style="min-width:320px;max-width:48%;background:#fff;'
                f'border:1px solid #ddd;border-radius:8px;overflow:hidden;'
                f'box-shadow:0 1px 4px rgba(0,0,0,.1);">'
                f'<div style="padding:8px 12px;background:#fafafa;'
                f'border-bottom:1px solid #eee;font-weight:600;font-size:13px;">'
                f'{name} — {total:.1f}%</div>'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:100%;display:block;"/></div>'
            )
        except Exception:
            continue
    gallery_html = ""
    if gallery_items:
        gallery_html = (
            f'<h2>Топ-{len(gallery_items)} визуализаций (SIFT inliers)</h2>'
            f'<div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;">'
            f'{"".join(gallery_items)}</div>'
        )

    counts = {
        "HIGH": sum(1 for r in rows if r["risk"] == "HIGH"),
        "MEDIUM": sum(1 for r in rows if r["risk"] == "MEDIUM"),
        "LOW": sum(1 for r in rows if r["risk"] == "LOW"),
    }
    max_total = rows[0]["total"] if rows else 0.0
    max_name = rows[0]["name"] if rows else ""
    n_with_sem = sum(1 for r in rows if r.get("semantic") is not None)

    cw = ", ".join(f"{w:.2f}" for w in classic_weights)
    sem_visible_default = "table-row" if n_with_sem > 0 else "none"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Отчёт v09 — сравнение товарных знаков</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 24px; background: #f5f5f5; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 28px; }}
  .meta {{ color: #666; margin-bottom: 20px; line-height: 1.6; }}
  .card {{ background: #fff; border-radius: 10px; padding: 16px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); border-radius: 8px; overflow: hidden; }}
  th {{ background: #1565c0; color: #fff; padding: 9px 6px; text-align: left;
       font-weight: 600; font-size: 12px; cursor: pointer; user-select: none; }}
  th:hover {{ background: #0d47a1; }}
  td {{ font-size: 12px; }}
  tr:hover td {{ background: #e3f2fd !important; }}
  .sem-col {{ display: {sem_visible_default}; }}
  .thumb {{ border: 1px solid #ddd; border-radius: 6px; max-width: 100%; }}
  .kpi {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .kpi div {{ background: #fff; padding: 12px 16px; border-radius: 8px;
             box-shadow: 0 1px 2px rgba(0,0,0,.1); min-width: 130px; }}
  .kpi b {{ font-size: 20px; display: block; }}
</style>
</head>
<body>
  <h1>Сравнение товарных знаков
    <span style="font-weight:400;color:#666;">{SCRIPT_VERSION}</span></h1>
  <div class="meta">
    Эталон: <b>{new_path.name}</b> &nbsp;|&nbsp;
    Обработано: <b>{len(rows)}</b> &nbsp;|&nbsp;
    Время: <b>{elapsed:.1f} с</b> &nbsp;|&nbsp;
    Порог Match: <b>{threshold}%</b><br/>
    Классические веса (SIFT,ORB,Color,pHash,SSIM,Shape): {cw}<br/>
    Semantic weight: <b>{semantic_weight:.2f}</b> &nbsp;|&nbsp;
    Эмбеддинги: <code>{embedding_info}</code>
  </div>

  <div class="card" style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div style="font-weight:600;margin-bottom:8px;">Эталон</div>
      <img class="thumb" src="{new_thumb_b64}" style="max-width:280px;"/>
    </div>
    <div class="kpi">
      <div><span>Всего</span><b>{len(rows)}</b></div>
      <div><span>HIGH</span><b style="color:#c62828;">{counts['HIGH']}</b></div>
      <div><span>MEDIUM</span><b style="color:#ef6c00;">{counts['MEDIUM']}</b></div>
      <div><span>LOW</span><b style="color:#2e7d32;">{counts['LOW']}</b></div>
      <div><span>С семантикой</span><b>{n_with_sem}</b></div>
      <div><span>Макс Total%</span><b>{max_total:.1f}%</b><small>{max_name}</small></div>
    </div>
  </div>

  {gallery_html}

  <div style="margin:12px 0;">
    <input id="search" placeholder="Фильтр по имени..."
           style="padding:8px 12px;border:1px solid #ccc;border-radius:6px;width:260px;"/>
    <label style="margin-left:12px;font-size:13px;">
      <input type="checkbox" id="showSem" {'checked' if n_with_sem > 0 else ''}/>
      показать семантику
    </label>
    <span style="margin-left:12px;color:#666;font-size:13px;">
      Клик по заголовку — сортировка
    </span>
  </div>

  <table id="resultTable">
    <thead>
      <tr>
        <th data-k="i">#</th>
        <th data-k="name">Файл</th>
        <th data-k="total">Total %</th>
        <th data-k="sift">SIFT %</th>
        <th data-k="orb">ORB %</th>
        <th data-k="color">Color %</th>
        <th data-k="phash">pHash %</th>
        <th data-k="ssim">SSIM %</th>
        <th data-k="shape">Shape %</th>
        <th class="sem-col" data-k="semantic">Sem %</th>
        <th data-k="inliers">Inl</th>
        <th>Risk</th>
        <th>Match</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>

  <script>
    const table = document.getElementById('resultTable');
    const tbody = table.querySelector('tbody');
    const search = document.getElementById('search');
    const showSem = document.getElementById('showSem');
    let sortDir = 1, lastKey = 'total';

    function toggleSem() {{
      const show = showSem.checked;
      document.querySelectorAll('.sem-col').forEach(el => {{
        el.style.display = show ? '' : 'none';
      }});
    }}
    if (showSem) showSem.addEventListener('change', toggleSem);
    toggleSem();

    function parseVal(tr, key) {{
      if (key === 'name') return tr.dataset.name.toLowerCase();
      if (key === 'total') return parseFloat(tr.dataset.total);
      if (key === 'i') return parseInt(tr.cells[0].textContent);
      const ths = table.querySelectorAll('thead th');
      let idx = -1;
      ths.forEach((th, j) => {{ if (th.dataset.k === key) idx = j; }});
      if (idx < 0) return 0;
      const txt = tr.cells[idx].textContent.trim();
      if (txt === '—') return -1;
      const m = txt.match(/([0-9.]+)%/);
      return m ? parseFloat(m[1]) : (parseFloat(txt) || 0);
    }}

    table.querySelectorAll('th[data-k]').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.k;
        if (lastKey === key) sortDir *= -1;
        else {{ lastKey = key; sortDir = (key === 'name' ? 1 : -1); }}
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {{
          const va = parseVal(a, key), vb = parseVal(b, key);
          if (typeof va === 'string') return sortDir * va.localeCompare(vb);
          return sortDir * (vb - va);
        }});
        rows.forEach(r => tbody.appendChild(r));
        rows.forEach((r, i) => {{ r.cells[0].textContent = i + 1; }});
      }});
    }});

    search.addEventListener('input', () => {{
      const q = search.value.toLowerCase();
      tbody.querySelectorAll('tr').forEach(tr => {{
        tr.style.display = tr.dataset.name.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML-отчёт сохранён: {out_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--new", default="new.jpg", help="эталонное изображение")
    p.add_argument("--collections", default="collections", help="папка коллекции")
    p.add_argument("--recursive", action="store_true",
                   help="рекурсивный обход коллекции")

    p.add_argument("--csv", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--html", default=None)

    p.add_argument("--threshold", type=float, default=25.0,
                   help="порог Total%% для пометки YES")
    p.add_argument("--risk-high", type=float, default=RISK_HIGH_DEFAULT)
    p.add_argument("--risk-medium", type=float, default=RISK_MEDIUM_DEFAULT)

    # Классические веса — 6 значений (сумма нормируется)
    p.add_argument("--weights", nargs=6, type=float, default=CLASSIC_WEIGHTS_V07,
                   metavar=("SIFT", "ORB", "COLOR", "PHASH", "SSIM", "SHAPE"),
                   help="классические веса (нормализуются)")

    # Семантика — отдельный вес
    p.add_argument("--semantic-weight", type=float, default=0.0,
                   help="вес семантики в Total%% (0 = v07-совместимо, "
                        "иначе total = (1-w)*classic + w*semantic)")

    p.add_argument("--max-dim", type=int, default=1200)
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    p.add_argument("--ransac", type=float, default=5.0)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--sift-features", type=int, default=4000)
    p.add_argument("--orb-features", type=int, default=2000)

    p.add_argument("--top", type=int, default=0, help="0 = все")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--top-vis", type=int, default=3)

    p.add_argument("--cache", action="store_true")
    p.add_argument("--cache-file", default=".tm_cache_v2.pkl")

    p.add_argument("--clahe", type=float, default=2.0)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--ignore-white", action="store_true")
    p.add_argument("--white-thresh", type=int, default=WHITE_THRESH_DEFAULT)

    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--use-threads", action="store_true")

    # логирование
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--log-flat", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--log-dir", type=str, default=LOG_DIR_DEFAULT)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # эмбеддинги
    p.add_argument("--embedding", type=str, default="auto",
                   choices=["auto", "none", "clip", "dino", "both"])
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-base")
    p.add_argument("--embedding-device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--embedding-batch-size", type=int, default=16)
    p.add_argument("--embedding-cache-file", type=str, default=".emb_cache_v09.pkl")
    p.add_argument("--no-emb-cache", action="store_true")

    # сигмоида для семантики
    p.add_argument("--clip-neutral", type=float, default=CLIP_NEUTRAL_DEFAULT,
                   help="cosine → 50%% для CLIP (по умолчанию 0.50)")
    p.add_argument("--clip-scale", type=float, default=CLIP_SCALE_DEFAULT,
                   help="наклон сигмоиды CLIP (меньше = резче)")
    p.add_argument("--dino-neutral", type=float, default=DINO_NEUTRAL_DEFAULT,
                   help="cosine → 50%% для DINOv2 (по умолчанию 0.65)")
    p.add_argument("--dino-scale", type=float, default=DINO_SCALE_DEFAULT)

    return p.parse_args()


# ============================================================================
# Основная логика
# ============================================================================

def _compute_semantic_scores(
    args,
    logger: logging.Logger,
    img_new_raw: np.ndarray,
    files: List[Path],
) -> Tuple[Dict[str, Optional[float]], str, Optional[str], Optional[str]]:
    """
    Считает семантические перценты (сигмоида от косинуса) для каждого файла.
    Возвращает:
      scores  — {path_str: semantic_pct | None}
      info    — человекочитаемое описание режима
      clip_model_used, dino_model_used  — имена моделей для кэша (или None)
    """
    scores: Dict[str, Optional[float]] = {str(f): None for f in files}
    info = "disabled (embedding=none)"
    clip_used: Optional[str] = None
    dino_used: Optional[str] = None

    if args.embedding == "none":
        return scores, info, None, None

    if not _HAS_PIL:
        logger.warning("Pillow не установлен — семантика отключена "
                       "(pip install Pillow)")
        return scores, "disabled (no Pillow)", None, None

    use_clip = args.embedding in ("auto", "clip", "both")
    use_dino = args.embedding in ("auto", "dino", "both")

    mgr = EmbeddingManager(
        clip_model_name=args.clip_model,
        dino_model_name=args.dino_model,
        device=args.embedding_device,
        logger=logger,
    )

    clip_ok = mgr.load_clip() if use_clip else False
    dino_ok = mgr.load_dino() if use_dino else False

    # auto → если ничего не загрузилось, тихо откатываемся
    if args.embedding == "auto" and not clip_ok and not dino_ok:
        logger.warning("auto: ни CLIP, ни DINOv2 не загрузились — "
                       "работаем как v07")
        return scores, "auto: both failed, fallback to v07", None, None

    # Явные режимы: если запрошен только clip и он не загрузился — работаем без семантики
    if args.embedding == "clip" and not clip_ok:
        logger.warning("--embedding clip, но CLIP не загрузился — "
                       "семантика отключена")
        return scores, "clip failed", None, None
    if args.embedding == "dino" and not dino_ok:
        logger.warning("--embedding dino, но DINOv2 не загрузился — "
                       "семантика отключена")
        return scores, "dino failed", None, None

    info = (f"mode={args.embedding} clip={'ok' if clip_ok else 'off'}"
            f"({args.clip_model}) dino={'ok' if dino_ok else 'off'}"
            f"({args.dino_model}) device={mgr.device}")
    if clip_ok:
        clip_used = args.clip_model
    if dino_ok:
        dino_used = args.dino_model

    # эталон
    pil_new = bgr_to_pil(img_new_raw)
    if pil_new is None:
        logger.warning("Не удалось конвертировать эталон в PIL — "
                       "семантика отключена")
        return scores, "etalon PIL conversion failed", None, None

    clip_ref = None
    dino_ref = None
    if clip_ok:
        clip_ref = mgr.encode_clip_batch([pil_new])[0]
    if dino_ok:
        dino_ref = mgr.encode_dino_batch([pil_new])[0]

    # эмбеддинг-кэш
    emb_cache: Dict[str, EmbCachedItem] = {}
    emb_cache_path = Path(args.embedding_cache_file)
    if not args.no_emb_cache:
        emb_cache = load_emb_cache(emb_cache_path)
        logger.info(f"Кэш эмбеддингов: {len(emb_cache)} из {emb_cache_path}")

    def cos_key(fp: Path, kind: str, model: str) -> str:
        return emb_file_key(fp, f"{kind}|{model}")

    clip_raw: Dict[str, float] = {}
    dino_raw: Dict[str, float] = {}
    to_compute: List[Path] = []

    for fp in files:
        # попытка достать из кэша
        clip_emb = None
        dino_emb = None
        if clip_ok:
            key = cos_key(fp, "clip", args.clip_model)
            item = emb_cache.get(key)
            if item is not None and item.is_valid(fp) and item.embedding is not None:
                clip_emb = item.embedding
        if dino_ok:
            key = cos_key(fp, "dino", args.dino_model)
            item = emb_cache.get(key)
            if item is not None and item.is_valid(fp) and item.embedding is not None:
                dino_emb = item.embedding

        # если для нужных моделей всё нашлось — считаем сразу
        if (not clip_ok or clip_emb is not None) and (not dino_ok or dino_emb is not None):
            if clip_ok and clip_emb is not None:
                clip_raw[str(fp)] = cosine_similarity(clip_ref, clip_emb)
            if dino_ok and dino_emb is not None:
                dino_raw[str(fp)] = cosine_similarity(dino_ref, dino_emb)
        else:
            to_compute.append(fp)

    logger.info(f"Эмбеддинги: из кэша {len(files) - len(to_compute)}/{len(files)}, "
                f"к подсчёту {len(to_compute)}")

    if to_compute:
        bs = max(1, args.embedding_batch_size)
        total_batches = (len(to_compute) + bs - 1) // bs
        iterator = range(total_batches)
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc="Embeddings", unit="batch")

        for b_idx in iterator:
            batch_fps = to_compute[b_idx * bs: (b_idx + 1) * bs]
            batch_pils: List["Image.Image"] = []
            kept_fps: List[Path] = []
            for fp in batch_fps:
                bgr = load_bgr_smart(fp, args.max_dim)
                if bgr is None:
                    continue
                pil = bgr_to_pil(bgr)
                if pil is None:
                    continue
                batch_pils.append(pil)
                kept_fps.append(fp)

            if not batch_pils:
                continue

            clip_embs = mgr.encode_clip_batch(batch_pils) if clip_ok else []
            dino_embs = mgr.encode_dino_batch(batch_pils) if dino_ok else []

            for i, fp in enumerate(kept_fps):
                try:
                    st = fp.stat()
                except OSError:
                    continue

                if clip_ok and i < len(clip_embs) and clip_embs[i] is not None:
                    emb = clip_embs[i]
                    clip_raw[str(fp)] = cosine_similarity(clip_ref, emb)
                    emb_cache[cos_key(fp, "clip", args.clip_model)] = EmbCachedItem(
                        mtime=st.st_mtime, size=st.st_size,
                        model=f"clip|{args.clip_model}",
                        embedding=emb,
                    )
                if dino_ok and i < len(dino_embs) and dino_embs[i] is not None:
                    emb = dino_embs[i]
                    dino_raw[str(fp)] = cosine_similarity(dino_ref, emb)
                    emb_cache[cos_key(fp, "dino", args.dino_model)] = EmbCachedItem(
                        mtime=st.st_mtime, size=st.st_size,
                        model=f"dino|{args.dino_model}",
                        embedding=emb,
                    )

        if not args.no_emb_cache:
            save_emb_cache(emb_cache_path, emb_cache)
            logger.info(f"Кэш эмбеддингов сохранён: {len(emb_cache)} -> "
                        f"{emb_cache_path}")

    # сигмоида + смешивание CLIP и DINO
    for fp in files:
        key = str(fp)
        parts: List[Tuple[float, float]] = []  # (score, weight)
        if clip_ok and key in clip_raw:
            clip_pct = semantic_to_percent(clip_raw[key],
                                           args.clip_neutral, args.clip_scale)
            parts.append((clip_pct, 0.6))
        if dino_ok and key in dino_raw:
            dino_pct = semantic_to_percent(dino_raw[key],
                                           args.dino_neutral, args.dino_scale)
            parts.append((dino_pct, 0.4))

        if not parts:
            scores[key] = None
            continue
        wsum = sum(w for _, w in parts)
        scores[key] = sum(s * w for s, w in parts) / max(wsum, 1e-9)

    return scores, info, clip_used, dino_used


def _main_impl(args, log_path: Optional[Path]) -> None:
    t0 = time.perf_counter()
    dt_start = datetime.now()

    logger = setup_logger(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("=" * 90)
    logger.info(f"Запуск: {dt_start:%Y-%m-%d %H:%M:%S} | "
                f"compare_trademarks_{SCRIPT_VERSION}")
    logger.info(f"Лог-файл: {log_path.resolve() if log_path else 'отключён'}")
    if args.semantic_weight > 0:
        logger.warning("semantic_weight > 0: Total%% смешивает классику и "
                       "семантику. Пороги --threshold / --risk-* отличаются "
                       "от v07.")
    else:
        logger.info("semantic_weight = 0: Total%% совпадает с v07 "
                    "(семантика — только информационные колонки).")
    logger.info("=" * 90)

    new_path = Path(args.new)
    coll_dir = Path(args.collections)
    if not new_path.exists():
        logger.error(f"Не найден файл: {new_path}")
        sys.exit(1)
    if not coll_dir.is_dir():
        logger.error(f"Не найдена папка: {coll_dir}")
        sys.exit(1)

    # если --new — папка, берём первый файл
    if new_path.is_dir():
        cands = collect_files(new_path, recursive=False)
        if not cands:
            logger.error(f"В папке эталонов {new_path} нет изображений")
            sys.exit(1)
        etalon_path = cands[0]
        logger.warning(f"Эталон — папка, использую первый файл: {etalon_path.name}")
    else:
        etalon_path = new_path

    # --- классические веса ---
    w = np.asarray(args.weights, dtype=np.float64)
    if w.size != 6:
        logger.error(f"--weights ожидает 6 значений, получено {w.size}")
        sys.exit(1)
    if w.sum() <= 0:
        logger.error("Сумма классических весов должна быть > 0")
        sys.exit(1)
    w = w / w.sum()
    logger.info(f"Классические веса (SIFT,ORB,Color,pHash,SSIM,Shape): "
                f"{w[0]:.2f} {w[1]:.2f} {w[2]:.2f} {w[3]:.2f} {w[4]:.2f} {w[5]:.2f}")
    logger.info(f"Semantic weight: {args.semantic_weight:.2f}")

    # --- эталон ---
    clahe_val = 0.0 if args.no_preprocess else args.clahe
    img_new_raw = load_bgr_smart(etalon_path, args.max_dim)
    if img_new_raw is None:
        logger.error(f"Не удалось открыть эталон: {etalon_path}")
        sys.exit(1)
    img_new, gray_new = preprocess(img_new_raw, clahe_val)
    hash_new = phash(img_new)

    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    orb = cv2.ORB_create(nfeatures=args.orb_features)
    kp_new, des_new = sift.detectAndCompute(gray_new, None)
    kp_new_pts = np.float32([k.pt for k in kp_new]) if kp_new else None
    _, des_orb_new = orb.detectAndCompute(gray_new, None)

    n_kp_new = 0 if kp_new is None else len(kp_new)
    if n_kp_new < 10:
        logger.warning(f"На эталоне {etalon_path.name} мало SIFT-признаков "
                       f"({n_kp_new}); SIFT-метрика может быть ненадёжна.")

    # --- кэш SIFT ---
    cache: Dict[str, CachedItem] = {}
    cache_path = Path(args.cache_file)
    if args.cache:
        cache = load_cache(cache_path)
        logger.info(f"Кэш SIFT: {len(cache)} из {cache_path}")

    # --- файлы ---
    files = collect_files(coll_dir, args.recursive)
    if not files:
        logger.error(f"В папке {coll_dir} нет изображений")
        sys.exit(1)
    logger.info(f"Файлов коллекции: {len(files)} (recursive={args.recursive})")

    # --- семантика (предподсчёт в main) ---
    semantic_scores: Dict[str, Optional[float]] = {str(f): None for f in files}
    embedding_info = "disabled (embedding=none)"
    if args.embedding != "none" and args.semantic_weight >= 0.0:
        semantic_scores, embedding_info, _, _ = _compute_semantic_scores(
            args, logger, img_new_raw, files,
        )
    logger.info(f"Эмбеддинги: {embedding_info}")

    # --- задачи ---
    classic_w_list = tuple(w.tolist())
    tasks = []
    for path in files:
        key = file_key(path)
        cached = cache.get(key)
        if cached is not None and not cached.is_valid(path):
            cached = None
        tasks.append((
            str(path),
            des_new, kp_new_pts, des_orb_new, hash_new,
            img_new, gray_new,
            args.max_dim, clahe_val, args.ratio, args.ransac,
            args.min_inliers, args.sift_features, args.orb_features,
            classic_w_list,
            cached,
            args.ignore_white, args.white_thresh,
            args.risk_high, args.risk_medium,
            float(args.semantic_weight),
            semantic_scores.get(str(path)),
        ))

    # --- пул ---
    workers = args.workers if args.workers > 0 else None
    Executor = ThreadPoolExecutor if args.use_threads else ProcessPoolExecutor
    logger.info(f"Запуск {len(tasks)} задач (workers={workers or 'auto'}, "
                f"executor={Executor.__name__})...")

    executor_kwargs: Dict[str, Any] = {}
    log_path_str = str(log_path.resolve()) if log_path is not None else ""
    if not args.use_threads and log_path_str:
        executor_kwargs["initializer"] = _init_worker_logging
        executor_kwargs["initargs"] = (log_path_str,)

    rows: List[Dict[str, Any]] = []
    updated_cache: Dict[str, CachedItem] = dict(cache)

    with Executor(max_workers=workers, **executor_kwargs) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}
        total_n = len(tasks)
        pbar = tqdm(total=total_n, desc="Сравнение", unit="file") if _HAS_TQDM else None
        done = 0
        for fut in as_completed(futures):
            if pbar:
                pbar.update(1)
            else:
                done += 1
                name = Path(futures[fut]).name
                print(f"\r[{done}/{total_n}] {name[:60]:<60}", end="", flush=True)
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"Ошибка на {Path(futures[fut]).name}: {e}\n"
                             f"{traceback.format_exc()}")
                continue
            if res is None:
                continue
            if args.cache and res.get("cached_item") is not None:
                updated_cache[file_key(Path(res["path"]))] = res["cached_item"]
            rows.append(res)

        if pbar:
            pbar.close()
        else:
            print()

    rows.sort(key=lambda r: r["total"], reverse=True)

    # --- консольная таблица ---
    header = (
        f"{'#':>3}  {'Файл':32}  {'Total%':>7}  {'SIFT%':>6}  {'ORB%':>5}  "
        f"{'Color%':>7}  {'pHash%':>7}  {'SSIM%':>6}  {'Shape%':>7}  "
        f"{'Sem%':>6}  {'Inl':>4}  Risk   Match"
    )
    print(header)
    print("-" * len(header))
    shown = rows if args.top <= 0 else rows[: args.top]
    for i, r in enumerate(shown, 1):
        flag = "YES" if r["total"] >= args.threshold else ""
        sem = r.get("semantic")
        sem_str = "  —  " if sem is None else f"{sem:6.2f}"
        print(
            f"{i:>3}  {r['name'][:32]:32}  {r['total']:7.2f}  {r['sift']:6.2f}  "
            f"{r['orb']:5.2f}  {r['color']:7.2f}  {r['phash']:7.2f}  "
            f"{r['ssim']:6.2f}  {r['shape']:7.2f}  {sem_str}  "
            f"{r['inliers']:>4}  {r['risk']:6} {flag}"
        )

    # --- CSV ---
    if args.csv:
        fieldnames = [
            "name", "path", "total", "classic_total",
            "sift", "orb", "color", "phash", "ssim", "shape",
            "semantic", "inliers", "good", "n_new", "n_coll", "risk",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in r.items() if k in fieldnames
                })
        logger.info(f"CSV-отчёт сохранён: {args.csv}")

    # --- JSON ---
    if args.json:
        clean = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows]
        payload = {
            "etalon": str(etalon_path),
            "collections": str(coll_dir),
            "threshold": args.threshold,
            "risk_high": args.risk_high,
            "risk_medium": args.risk_medium,
            "classic_weights": w.tolist(),
            "semantic_weight": float(args.semantic_weight),
            "embedding": embedding_info,
            "elapsed": time.perf_counter() - t0,
            "results": clean,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON-отчёт сохранён: {args.json}")

    # --- визуализация top-N ---
    top_vis_paths: List[Tuple[Path, str, float]] = []
    if args.visualize and rows:
        k = max(1, args.top_vis)
        for idx in range(min(k, len(rows))):
            best = rows[idx]
            extra = best.get("_extra")
            img2 = best.get("_img2")
            kp2_pts = best.get("_kp2_pts")
            if extra is None or kp_new is None or kp2_pts is None or img2 is None:
                continue
            good, mask, _ = extra
            kp2_list = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1)
                        for pt in kp2_pts]
            inlier_matches = [m for m, ok in zip(good, mask.ravel().tolist()) if ok]
            if len(inlier_matches) > 150:
                inlier_matches = inlier_matches[:150]
            vis = cv2.drawMatches(
                img_new, kp_new, img2, kp2_list, inlier_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(120, 120, 120),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            out_name = f"top{idx + 1}_matches_{Path(best['name']).stem}.jpg"
            out_path = Path(out_name)
            cv2.imwrite(str(out_path), vis)
            logger.info(f"Визуализация top-{idx + 1}: {out_path} "
                        f"({best['name']}, Total={best['total']:.2f}%)")
            top_vis_paths.append((out_path, best["name"], best["total"]))

        if top_vis_paths:
            try:
                shutil.copy(str(top_vis_paths[0][0]), "top1_matches.jpg")
            except Exception:
                pass

    # --- HTML ---
    if args.html:
        clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows]
        make_html_report(
            clean_rows, etalon_path, img_new_raw, args.threshold,
            top_vis_paths, Path(args.html),
            time.perf_counter() - t0,
            tuple(w.tolist()), float(args.semantic_weight),
            embedding_info,
        )

    # --- сохранение кэша SIFT ---
    if args.cache:
        save_cache(cache_path, updated_cache)
        logger.info(f"Кэш SIFT: {len(updated_cache)} -> {cache_path}")

    dt_end = datetime.now()
    elapsed = time.perf_counter() - t0
    n_high = sum(1 for r in rows if r["risk"] == "HIGH")
    n_med = sum(1 for r in rows if r["risk"] == "MEDIUM")
    logger.info(f"Готово {len(rows)} за {elapsed:.1f} с. "
                f"HIGH={n_high}  MEDIUM={n_med}  LOW={len(rows) - n_high - n_med}")
    logger.info(f"Завершение: {dt_end:%Y-%m-%d %H:%M:%S} "
                f"(всего {dt_end - dt_start})")
    if log_path:
        logger.info(f"Полный лог: {log_path.resolve()}")


def main() -> None:
    args = parse_args()
    log_path = _resolve_log_path(
        explicit=args.log,
        flat=args.log_flat,
        no_log=args.no_log,
        log_dir=args.log_dir,
    )
    if log_path is None:
        _main_impl(args, log_path=None)
        return
    try:
        with open(log_path, "w", encoding="utf-8"):
            pass
    except OSError as e:
        print(f"[WARN] не удалось создать лог {log_path}: {e}. "
              f"Продолжаем без лога.", file=sys.stderr)
        _main_impl(args, log_path=None)
        return
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"[log] Script  : {Path(sys.argv[0]).resolve().name}")
            print(f"[log] Log file: {log_path.resolve()}")
            print(f"[log] Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
            _main_impl(args, log_path=log_path)
            print(f"[log] Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()