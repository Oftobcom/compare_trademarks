#!/usr/bin/env python3
"""
compare_trademarks_v06.py — сравнение товарного знака с коллекцией (финальная версия)

Что нового относительно v01-v05:
  • 6 метрик вместо 5: SIFT, ORB, Color, pHash, SSIM + SHAPE (контуры + Hu-моменты)
  • Умная загрузка: поддержка PNG с альфа-каналом (композит на белый), рекурсивный обход,
    игнорирование почти-белого фона для цвета
  • Кэш v2: ключ = путь + mtime + размер, атомарное сохранение, LRU-очистка
  • Параллельность: ProcessPoolExecutor + ThreadPool fallback + tqdm если установлен
  • Отчёты: CSV, JSON, HTML v2 (интерактивная сортировка, thumbnails base64, галерея top-K)
  • Визуализация: не только top-1, а top-N с SIFT-инлайерами + контурами
  • Логирование: logging модуль + Tee в файл, тайминги, статистика
  • Риск-классификация: HIGH/MEDIUM/LOW по Total%
  • Более устойчивые формулы: purity, confidence, exp-маппинг для Shape
  • CLI: --recursive, --ignore-white, --top-vis, --json, --risk-high/low, --clahe, --no-preprocess

Зависимости: opencv-python, numpy. Опционально: tqdm
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_WEIGHTS = (0.35, 0.10, 0.15, 0.10, 0.15, 0.15)  # SIFT, ORB, Color, pHash, SSIM, Shape
SCRIPT_BASENAME = "compare_trademarks"
LOG_DIR_DEFAULT = "logs"
CACHE_VERSION = 2
RISK_HIGH = 60.0
RISK_MEDIUM = 30.0

def setup_logging(log_path: Optional[Path], level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(SCRIPT_BASENAME)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
    def write(self, data):
        for s in self.streams:
            try: s.write(data)
            except Exception: pass
        if "\n" in data: self.flush()
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def fileno(self): return self.streams[0].fileno()

_WORKER_LOG_FILE = None
def _init_worker_logging(log_path: str) -> None:
    global _WORKER_LOG_FILE
    if not log_path: return
    if isinstance(sys.stdout, Tee): return
    try:
        _WORKER_LOG_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, _WORKER_LOG_FILE)
        sys.stderr = Tee(sys.stderr, _WORKER_LOG_FILE)
    except Exception: pass

def _resolve_log_path(explicit: Optional[str], flat: bool, no_log: bool, log_dir: str = LOG_DIR_DEFAULT) -> Optional[Path]:
    if no_log: return None
    if explicit:
        p = Path(explicit); p.parent.mkdir(parents=True, exist_ok=True); return p
    out_dir = Path(log_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if flat: return out_dir / f"{SCRIPT_BASENAME}.log"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{SCRIPT_BASENAME}_{ts}.log"

def load_bgr_smart(path: Path, max_dim: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        white = np.ones_like(b, dtype=np.uint8) * 255
        alpha = a.astype(np.float32) / 255.0
        b = (b.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        img = cv2.merge([b, g, r])
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA))
    return img

def preprocess(bgr: np.ndarray, clahe_clip: float, do_blur: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if clahe_clip <= 0:
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if do_blur: bgr2 = cv2.GaussianBlur(bgr2, (3, 3), 0)
    return bgr2, cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

def phash(bgr: np.ndarray, hash_size: int = 16, hf: int = 4) -> np.ndarray:
    size = hash_size * hf
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])
    return (low > med).flatten()

def phash_score(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1 is None or h2 is None or h1.size != h2.size or h1.size == 0: return 0.0
    diff = int(np.count_nonzero(h1 != h2))
    return max(0.0, 100.0 * (1.0 - diff / h1.size))

def color_score(bgr1: np.ndarray, bgr2: np.ndarray, ignore_white: bool = False, white_thresh: int = 240) -> float:
    hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    mask1 = mask2 = None
    if ignore_white:
        def not_white_mask(bgr):
            white_mask = cv2.inRange(bgr, np.array([white_thresh, white_thresh, white_thresh]), np.array([255,255,255]))
            return cv2.bitwise_not(white_mask)
        mask1 = not_white_mask(bgr1); mask2 = not_white_mask(bgr2)
        if int((mask1>0).sum()) < bgr1.shape[0]*bgr1.shape[1]*0.05: mask1=None
        if int((mask2>0).sum()) < bgr2.shape[0]*bgr2.shape[1]*0.05: mask2=None
    h1 = cv2.calcHist([hsv1], [0,1], mask1, [36,32], [0,180,0,256])
    h2 = cv2.calcHist([hsv2], [0,1], mask2, [36,32], [0,180,0,256])
    cv2.normalize(h1,h1); cv2.normalize(h2,h2)
    dist = float(cv2.compareHist(h1,h2,cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(100.0, 100.0*(1.0-dist)))

def ssim_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    if gray1 is None or gray2 is None or gray1.size==0 or gray2.size==0: return 0.0
    if gray1.shape != gray2.shape:
        h = min(gray1.shape[0], gray2.shape[0]); w = min(gray1.shape[1], gray2.shape[1])
        if h<8 or w<8: return 0.0
        gray1 = cv2.resize(gray1,(w,h),interpolation=cv2.INTER_AREA)
        gray2 = cv2.resize(gray2,(w,h),interpolation=cv2.INTER_AREA)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    a=gray1.astype(np.float64); b=gray2.astype(np.float64)
    mu1=cv2.GaussianBlur(a,(11,11),1.5); mu2=cv2.GaussianBlur(b,(11,11),1.5)
    mu1_sq=mu1*mu1; mu2_sq=mu2*mu2; mu1_mu2=mu1*mu2
    sigma1_sq=cv2.GaussianBlur(a*a,(11,11),1.5)-mu1_sq
    sigma2_sq=cv2.GaussianBlur(b*b,(11,11),1.5)-mu2_sq
    sigma12=cv2.GaussianBlur(a*b,(11,11),1.5)-mu1_mu2
    num=(2*mu1_mu2+C1)*(2*sigma12+C2); den=(mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2)+1e-12
    return float(np.clip((num/den).mean()*100.0,0.0,100.0))

def shape_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    try:
        _, bin1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        def largest_contour(bin_img):
            contours,_=cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            area_thresh=bin_img.shape[0]*bin_img.shape[1]*0.01
            big=[c for c in contours if cv2.contourArea(c)>area_thresh]
            if not big: big=contours
            return max(big,key=cv2.contourArea)
        c1=largest_contour(bin1); c2=largest_contour(bin2)
        if c1 is None or c2 is None: return 0.0
        dist=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)
        shape_match=100.0*float(np.exp(-dist*2.0))
        m1=cv2.moments(c1); m2=cv2.moments(c2)
        hu1=cv2.HuMoments(m1).flatten(); hu2=cv2.HuMoments(m2).flatten()
        def log_hu(h): return -np.sign(h)*np.log10(np.abs(h)+1e-12)
        lh1=log_hu(hu1); lh2=log_hu(hu2)
        hu_dist=np.linalg.norm(lh1-lh2)
        hu_score=100.0*float(np.exp(-hu_dist*0.5))
        return float(0.6*shape_match+0.4*hu_score)
    except Exception: return 0.0

def orb_score_from_des(des1, des2, ratio: float=0.75) -> float:
    if des1 is None or des2 is None or len(des1)<2 or len(des2)<2: return 0.0
    try:
        bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches=bf.knnMatch(des1,des2,k=2)
        good=[]
        for pair in matches:
            if len(pair)<2: continue
            m,n=pair
            if m.distance < ratio*n.distance: good.append(m)
        if not good: return 0.0
        return min(100.0,(2.0*len(good)/(len(des1)+len(des2)))*100.0)
    except Exception: return 0.0

@dataclass
class CachedItem:
    mtime: float
    size: int
    shape: Tuple[int,int]
    version: int = CACHE_VERSION
    sift_des: Optional[np.ndarray]=None
    sift_kp_pts: Optional[np.ndarray]=None
    orb_des: Optional[np.ndarray]=None
    phash: Optional[np.ndarray]=None
    def is_valid(self, path: Path) -> bool:
        try:
            st=path.stat()
            return self.version==CACHE_VERSION and abs(self.mtime-st.st_mtime)<1e-3 and self.size==st.st_size
        except OSError: return False

def file_key(path: Path) -> str:
    try: return str(path.resolve())
    except OSError: return str(path)

def load_cache(cache_path: Path) -> Dict[str, CachedItem]:
    if not cache_path.exists(): return {}
    try:
        import pickle
        with open(cache_path,"rb") as f: raw=pickle.load(f)
        out={}
        for k,v in raw.items():
            if isinstance(v,CachedItem) and v.version==CACHE_VERSION: out[k]=v
            elif isinstance(v,dict):
                try:
                    if "size" not in v: v["size"]=0
                    item=CachedItem(**{kk:vv for kk,vv in v.items() if kk in CachedItem.__dataclass_fields__})
                    if item.version==CACHE_VERSION: out[k]=item
                except Exception: continue
        return out
    except Exception as e:
        print(f"[WARN] кэш повреждён ({e})",file=sys.stderr); return {}

def save_cache(cache_path: Path, data: Dict[str, CachedItem], max_entries: int=5000) -> None:
    try:
        import pickle
        if len(data)>max_entries:
            sorted_items=sorted(data.items(),key=lambda x:x[1].mtime,reverse=True)
            data=dict(sorted_items[:max_entries])
        tmp=cache_path.with_suffix(cache_path.suffix+".tmp")
        with open(tmp,"wb") as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except Exception as e: print(f"[WARN] не удалось сохранить кэш ({e})",file=sys.stderr)

def classify_risk(total: float, high: float=RISK_HIGH, medium: float=RISK_MEDIUM) -> str:
    if total>=high: return "HIGH"
    if total>=medium: return "MEDIUM"
    return "LOW"

def process_one(args_tuple: Tuple) -> Optional[Dict[str, Any]]:
    (path_str,new_des_sift,new_kp_pts,new_des_orb,new_phash,new_bgr,new_gray,max_dim,clahe_clip,ratio,ransac_thresh,min_inliers,sift_features,orb_features,weights,cached_item,ignore_white,risk_high,risk_medium)=args_tuple
    path=Path(path_str)
    try: st=path.stat(); mtime_now=st.st_mtime; size_now=st.st_size
    except OSError: return None
    if cached_item is not None and cached_item.is_valid(path):
        des2=cached_item.sift_des; kp2_pts=cached_item.sift_kp_pts; orb_des2=cached_item.orb_des; h2=cached_item.phash
        img_raw=load_bgr_smart(path,max_dim)
        if img_raw is None: return None
        img2,gray2=preprocess(img_raw,clahe_clip) if clahe_clip>0 else (img_raw,cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY))
        n2=0 if kp2_pts is None else len(kp2_pts)
        updated_cache_item=None
    else:
        img_raw=load_bgr_smart(path,max_dim)
        if img_raw is None: return None
        img2,gray2=preprocess(img_raw,clahe_clip) if clahe_clip>0 else (img_raw,cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY))
        sift=cv2.SIFT_create(nfeatures=sift_features); orb=cv2.ORB_create(nfeatures=orb_features)
        kp2,des2=sift.detectAndCompute(gray2,None)
        kp2_pts=np.float32([k.pt for k in kp2]) if kp2 else None
        _,orb_des2=orb.detectAndCompute(gray2,None)
        h2=phash(img2); n2=0 if kp2 is None else len(kp2)
        updated_cache_item=CachedItem(mtime=mtime_now,size=size_now,shape=gray2.shape[:2],version=CACHE_VERSION,sift_des=des2,sift_kp_pts=kp2_pts,orb_des=orb_des2,phash=h2)
    n1=0 if new_kp_pts is None else len(new_kp_pts)
    s_score=good_n=inl=0; extra=None
    if new_des_sift is not None and des2 is not None and len(new_des_sift)>=2 and len(des2)>=2 and new_kp_pts is not None and kp2_pts is not None:
        try:
            flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=50))
            matches=flann.knnMatch(new_des_sift,des2,k=2)
            good=[]
            for pair in matches:
                if len(pair)<2: continue
                m,n_=pair
                if m.distance < ratio*n_.distance: good.append(m)
            if len(good)>=4:
                src=np.float32([new_kp_pts[m.queryIdx] for m in good]).reshape(-1,1,2)
                dst=np.float32([kp2_pts[m.trainIdx] for m in good]).reshape(-1,1,2)
                H,mask=cv2.findHomography(src,dst,cv2.RANSAC,ransac_thresh)
                if H is not None and mask is not None:
                    inl=int(mask.sum()); good_n=len(good)
                    denom=max(1,min(n1,n2)); raw=inl/denom
                    conf=min(1.0,inl/max(1,min_inliers)); purity=inl/max(1,good_n)
                    s_score=100.0*raw*conf*(0.7+0.3*purity)
                    extra=(good,mask,H)
        except Exception: s_score=0.0
    o_score=orb_score_from_des(new_des_orb,orb_des2,ratio)
    c_score=color_score(new_bgr,img2,ignore_white=ignore_white)
    p_score=phash_score(new_phash,h2)
    ssim_val=0.0
    if extra is not None:
        _,_,H=extra; h_new,w_new=new_gray.shape[:2]
        try:
            aligned=cv2.warpPerspective(gray2,np.linalg.inv(H),(w_new,h_new))
            mask_img=cv2.warpPerspective(np.ones_like(gray2,dtype=np.uint8)*255,np.linalg.inv(H),(w_new,h_new))
            if int(mask_img.sum())>1000:
                ys,xs=np.where(mask_img>0)
                y0,y1=int(ys.min()),int(ys.max())+1; x0,x1=int(xs.min()),int(xs.max())+1
                y0=max(0,y0-2); x0=max(0,x0-2)
                ssim_val=ssim_score(new_gray[y0:y1,x0:x1],aligned[y0:y1,x0:x1])
        except Exception: ssim_val=0.0
    shape_val=shape_score(new_gray,gray2)
    total=(weights[0]*s_score+weights[1]*o_score+weights[2]*c_score+weights[3]*p_score+weights[4]*ssim_val+weights[5]*shape_val)
    risk=classify_risk(total,risk_high,risk_medium)
    return {"name":path.name,"path":str(path),"total":float(total),"sift":float(s_score),"orb":float(o_score),"color":float(c_score),"phash":float(p_score),"ssim":float(ssim_val),"shape":float(shape_val),"inliers":int(inl),"good":int(good_n),"n_new":int(n1),"n_coll":int(n2),"risk":risk,"cached_item":updated_cache_item,"extra":extra,"img2":img2 if extra is not None else None,"kp2_pts":kp2_pts if extra is not None else None,"gray2":gray2}

def _img_to_base64_thumb(bgr: np.ndarray, max_side: int=200, quality: int=80) -> str:
    try:
        h,w=bgr.shape[:2]; scale=max_side/max(h,w)
        thumb=cv2.resize(bgr,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA) if scale<1.0 else bgr
        _,buf=cv2.imencode(".jpg",thumb,[cv2.IMWRITE_JPEG_QUALITY,quality])
        return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    except Exception: return ""

def make_html_report_v2(rows, new_path, new_bgr, threshold, top_vis_paths, out_path, elapsed, weights, args_dict):
    new_thumb_b64=_img_to_base64_thumb(new_bgr,300)
    def bar(val,color="#4CAF50"):
        w=max(0.0,min(100.0,float(val)))
        return f'<div style="background:#eee;border-radius:3px;height:14px;width:90px;display:inline-block;vertical-align:middle;"><div style="background:{color};width:{w:.0f}%;height:100%;border-radius:3px;"></div></div> <span style="font-size:12px;">{w:.1f}%</span>'
    rows_html=[]
    for i,r in enumerate(rows,1):
        risk_color="#c62828" if r["risk"]=="HIGH" else ("#ef6c00" if r["risk"]=="MEDIUM" else "#2e7d32")
        risk_badge=f'<span style="background:{risk_color};color:#fff;padding:2px 6px;border-radius:10px;font-size:11px;">{r["risk"]}</span>'
        flag='<span style="color:#2e7d32;font-weight:bold;">YES</span>' if r["total"]>=threshold else ""
        bg="#ffebee" if r["risk"]=="HIGH" else ("#fff3e0" if r["risk"]=="MEDIUM" else ("#fff" if i%2 else "#fafafa"))
        rows_html.append(f'<tr style="background:{bg};" data-total="{r["total"]:.2f}" data-name="{r["name"]}"><td style="text-align:right;padding:6px 8px;">{i}</td><td style="padding:6px 8px;font-family:monospace;max-width:200px;overflow:hidden;text-overflow:ellipsis;" title="{r["name"]}">{r["name"]}</td><td style="padding:6px 8px;">{bar(r["total"],"#1976d2")}</td><td style="padding:6px 8px;">{bar(r["sift"])}</td><td style="padding:6px 8px;">{bar(r["orb"],"#7b1fa2")}</td><td style="padding:6px 8px;">{bar(r["color"],"#f57c00")}</td><td style="padding:6px 8px;">{bar(r["phash"],"#00838f")}</td><td style="padding:6px 8px;">{bar(r["ssim"],"#c62828")}</td><td style="padding:6px 8px;">{bar(r["shape"],"#33691e")}</td><td style="text-align:right;padding:6px 8px;">{r["inliers"]}</td><td style="text-align:center;padding:6px 8px;">{r["n_new"]}/{r["n_coll"]}</td><td style="text-align:center;padding:6px 8px;">{risk_badge}</td><td style="text-align:center;padding:6px 8px;">{flag}</td></tr>')
    gallery_html=""
    if top_vis_paths:
        items=[]
        for vis_path,name,total in top_vis_paths:
            if not vis_path.exists(): continue
            try:
                with open(vis_path,"rb") as f: b64=base64.b64encode(f.read()).decode("ascii")
                items.append(f'<div style="min-width:320px;max-width:48%;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);"><div style="padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee;font-weight:600;font-size:13px;">{name} — {total:.1f}%</div><img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;"/></div>')
            except Exception: continue
        if items: gallery_html=f'<h2>Топ-{len(items)} визуализаций (SIFT inliers)</h2><div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;">{"".join(items)}</div>'
    html=f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>Отчёт v06</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:24px;background:#f5f5f5;color:#222;}}h1{{margin-bottom:4px;}}h2{{margin-top:28px;}}.meta{{color:#666;margin-bottom:20px;line-height:1.6;}}.card{{background:#fff;border-radius:10px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.12);margin-bottom:20px;}}table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.12);border-radius:8px;overflow:hidden;}}th{{background:#1565c0;color:#fff;padding:10px 8px;text-align:left;font-weight:600;font-size:12px;cursor:pointer;user-select:none;}}th:hover{{background:#0d47a1;}}td{{font-size:12.5px;}}tr:hover td{{background:#e3f2fd !important;}}.thumb{{border:1px solid #ddd;border-radius:6px;max-width:100%;}}.kpi{{display:flex;gap:16px;flex-wrap:wrap;}}.kpi div{{background:#fff;padding:12px 16px;border-radius:8px;box-shadow:0 1px 2px rgba(0,0,0,.1);min-width:140px;}}.kpi b{{font-size:20px;display:block;}}</style></head><body>
<h1>Сравнение товарных знаков <span style="font-weight:400;color:#666;">v06</span></h1>
<div class="meta">Эталон: <b>{new_path.name}</b> | Обработано: <b>{len(rows)}</b> | Время: <b>{elapsed:.1f} с</b> | Порог: <b>{threshold}%</b><br/>Веса: SIFT={weights[0]:.2f} ORB={weights[1]:.2f} Color={weights[2]:.2f} pHash={weights[3]:.2f} SSIM={weights[4]:.2f} Shape={weights[5]:.2f}</div>
<div class="card" style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;"><div><div style="font-weight:600;margin-bottom:8px;">Эталон</div><img class="thumb" src="{new_thumb_b64}" style="max-width:300px;"/></div><div class="kpi"><div><span>Всего</span><b>{len(rows)}</b></div><div><span>HIGH</span><b style="color:#c62828;">{sum(1 for r in rows if r['risk']=='HIGH')}</b></div><div><span>MEDIUM</span><b style="color:#ef6c00;">{sum(1 for r in rows if r['risk']=='MEDIUM')}</b></div><div><span>LOW</span><b style="color:#2e7d32;">{sum(1 for r in rows if r['risk']=='LOW')}</b></div><div><span>Макс Total%</span><b>{rows[0]['total']:.1f}%</b> <small>{rows[0]['name'] if rows else ''}</small></div></div></div>
{gallery_html}
<div style="margin:12px 0;"><input id="search" placeholder="Фильтр по имени..." style="padding:8px 12px;border:1px solid #ccc;border-radius:6px;width:280px;"/><span style="margin-left:12px;color:#666;font-size:13px;">Клик по заголовку — сортировка</span></div>
<table id="resultTable"><thead><tr><th data-k="i">#</th><th data-k="name">Файл</th><th data-k="total">Total %</th><th data-k="sift">SIFT %</th><th data-k="orb">ORB %</th><th data-k="color">Color %</th><th data-k="phash">pHash %</th><th data-k="ssim">SSIM %</th><th data-k="shape">Shape %</th><th data-k="inliers">Inliers</th><th>KP n/c</th><th>Risk</th><th>Match</th></tr></thead><tbody>{"".join(rows_html)}</tbody></table>
<script>
const table=document.getElementById('resultTable');const tbody=table.querySelector('tbody');const search=document.getElementById('search');let sortDir=1;let lastKey='total';
function parseVal(tr,key){{if(key==='name') return tr.dataset.name.toLowerCase(); if(key==='total') return parseFloat(tr.dataset.total); if(key==='i') return parseInt(tr.cells[0].textContent); const idxMap={{sift:3,orb:4,color:5,phash:6,ssim:7,shape:8,inliers:9}}; const idx=idxMap[key]; if(idx!==undefined){{const txt=tr.cells[idx].textContent; const m=txt.match(/([0-9.]+)%/); if(m) return parseFloat(m[1]); return parseFloat(tr.cells[idx].textContent)||0;}} return 0;}}
table.querySelectorAll('th[data-k]').forEach(th=>{{th.addEventListener('click',()=>{{const key=th.dataset.k; if(lastKey===key) sortDir*=-1; else {{lastKey=key; sortDir=(key==='name'?1:-1);}} const rows=Array.from(tbody.querySelectorAll('tr')); rows.sort((a,b)=>{{let va=parseVal(a,key),vb=parseVal(b,key); if(typeof va==='string') return sortDir*va.localeCompare(vb); return sortDir*(vb-va);}}); rows.forEach(r=>tbody.appendChild(r)); rows.forEach((r,i)=>{{r.cells[0].textContent=i+1;}});}});}});
search.addEventListener('input',()=>{{const q=search.value.toLowerCase(); tbody.querySelectorAll('tr').forEach(tr=>{{const name=tr.dataset.name.toLowerCase(); tr.style.display=name.includes(q)?'':'none';}});}});
</script></body></html>"""
    out_path.write_text(html,encoding="utf-8")
    print(f"HTML-отчёт v2 сохранён: {out_path}")

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new",default="new.jpg",help="эталон")
    p.add_argument("--collections",default="collections",help="папка коллекции")
    p.add_argument("--recursive",action="store_true",help="рекурсивный обход")
    p.add_argument("--csv",default=None); p.add_argument("--json",default=None); p.add_argument("--html",default=None)
    p.add_argument("--threshold",type=float,default=25.0); p.add_argument("--risk-high",type=float,default=RISK_HIGH); p.add_argument("--risk-medium",type=float,default=RISK_MEDIUM)
    p.add_argument("--weights",nargs=6,type=float,default=DEFAULT_WEIGHTS,metavar=("SIFT","ORB","COLOR","PHASH","SSIM","SHAPE"))
    p.add_argument("--max-dim",type=int,default=1200); p.add_argument("--ratio",type=float,default=0.75); p.add_argument("--ransac",type=float,default=5.0)
    p.add_argument("--min-inliers",type=int,default=12); p.add_argument("--sift-features",type=int,default=4000); p.add_argument("--orb-features",type=int,default=2000)
    p.add_argument("--top",type=int,default=0); p.add_argument("--top-vis",type=int,default=3); p.add_argument("--visualize",action="store_true")
    p.add_argument("--cache",action="store_true"); p.add_argument("--cache-file",default=".tm_cache_v2.pkl")
    p.add_argument("--clahe",type=float,default=2.0); p.add_argument("--ignore-white",action="store_true"); p.add_argument("--no-preprocess",action="store_true")
    p.add_argument("--workers",type=int,default=0); p.add_argument("--use-threads",action="store_true")
    p.add_argument("--log",type=str,default=None); p.add_argument("--log-flat",action="store_true"); p.add_argument("--no-log",action="store_true")
    p.add_argument("--log-dir",type=str,default=LOG_DIR_DEFAULT); p.add_argument("--log-level",type=str,default="INFO",choices=["DEBUG","INFO","WARN","ERROR"])
    return p.parse_args()

def collect_files(root: Path, recursive: bool):
    if recursive: return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    else: return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def _main_impl(args, log_path):
    t0=time.perf_counter(); dt_start=datetime.now()
    logger=setup_logging(log_path,level=getattr(logging,args.log_level.upper(),logging.INFO))
    logger.info("="*90); logger.info(f"Запуск: {dt_start:%Y-%m-%d %H:%M:%S} | v06")
    new_path=Path(args.new); coll_dir=Path(args.collections)
    if not new_path.exists(): logger.error(f"Не найден: {new_path}"); sys.exit(1)
    if not coll_dir.is_dir(): logger.error(f"Не найдена папка: {coll_dir}"); sys.exit(1)
    if new_path.is_dir():
        cands=collect_files(new_path,False)
        if not cands: logger.error(f"В папке эталонов {new_path} нет изображений"); sys.exit(1)
        etalon_path=cands[0]; logger.warning(f"Папка эталонов, беру первый: {etalon_path.name}")
    else: etalon_path=new_path
    clahe_val=0.0 if args.no_preprocess else args.clahe
    img_new_raw=load_bgr_smart(etalon_path,args.max_dim)
    if img_new_raw is None: logger.error(f"Не удалось открыть эталон: {etalon_path}"); sys.exit(1)
    img_new,gray_new=preprocess(img_new_raw,clahe_val) if clahe_val>0 else (img_new_raw,cv2.cvtColor(img_new_raw,cv2.COLOR_BGR2GRAY))
    hash_new=phash(img_new); sift=cv2.SIFT_create(nfeatures=args.sift_features); orb=cv2.ORB_create(nfeatures=args.orb_features)
    kp_new,des_new=sift.detectAndCompute(gray_new,None); kp_new_pts=np.float32([k.pt for k in kp_new]) if kp_new else None
    _,des_orb_new=orb.detectAndCompute(gray_new,None)
    if (0 if kp_new is None else len(kp_new))<10: logger.warning(f"Мало SIFT на {etalon_path.name}")
    cache={}; cache_path=Path(args.cache_file)
    if args.cache: cache=load_cache(cache_path); logger.info(f"Кэш: {len(cache)} из {cache_path}")
    w=np.asarray(args.weights,dtype=np.float64); w/=max(1e-9,w.sum()); logger.info(f"Веса: SIFT={w[0]:.2f} ORB={w[1]:.2f} Color={w[2]:.2f} pHash={w[3]:.2f} SSIM={w[4]:.2f} Shape={w[5]:.2f}")
    files=collect_files(coll_dir,args.recursive)
    if not files: logger.error("Нет изображений"); sys.exit(1)
    logger.info(f"Файлов: {len(files)}")
    tasks=[]
    for path in files:
        key=file_key(path); cached=cache.get(key)
        if cached is not None and not cached.is_valid(path): cached=None
        tasks.append((str(path),des_new,kp_new_pts,des_orb_new,hash_new,img_new,gray_new,args.max_dim,clahe_val,args.ratio,args.ransac,args.min_inliers,args.sift_features,args.orb_features,w.tolist(),cached,args.ignore_white,args.risk_high,args.risk_medium))
    workers=args.workers if args.workers>0 else None
    try:
        from tqdm import tqdm; has_tqdm=True
    except ImportError: has_tqdm=False
    Executor=ThreadPoolExecutor if args.use_threads else ProcessPoolExecutor
    log_path_str=str(log_path.resolve()) if log_path else ""; init=_init_worker_logging if log_path_str else None; initargs=(log_path_str,) if log_path_str else {}
    executor_kwargs={}; 
    if not args.use_threads and init: executor_kwargs["initializer"]=init; executor_kwargs["initargs"]=initargs
    rows=[]; updated_cache=dict(cache)
    with Executor(max_workers=workers,**executor_kwargs) as executor:
        futures={executor.submit(process_one,t):t[0] for t in tasks}
        if has_tqdm: pbar=tqdm(total=len(tasks),desc="Сравнение",unit="file")
        else: pbar=None; done=0
        for fut in as_completed(futures):
            if pbar: pbar.update(1)
            else: done+=1; print(f"\r[{done}/{len(tasks)}] {Path(futures[fut]).name[:60]:<60}",end="",flush=True)
            try: res=fut.result()
            except Exception as e: logger.error(f"Ошибка {Path(futures[fut]).name}: {e}\n{traceback.format_exc()}"); continue
            if res is None: continue
            if args.cache and res.get("cached_item") is not None: updated_cache[file_key(Path(res["path"]))]=res["cached_item"]
            row={k:v for k,v in res.items() if k not in ("cached_item","extra","img2","kp2_pts","gray2")}
            row["_extra"]=res.get("extra"); row["_img2"]=res.get("img2"); row["_kp2_pts"]=res.get("kp2_pts"); row["_gray2"]=res.get("gray2"); rows.append(row)
        if pbar: pbar.close()
        else: print()
    rows.sort(key=lambda r:r["total"],reverse=True)
    header=f"{'#':>3}  {'Файл':32}  {'Total%':>7}  {'SIFT%':>6}  {'ORB%':>5}  {'Color%':>7}  {'pHash%':>7}  {'SSIM%':>6}  {'Shape%':>7}  {'Inl':>4}  {'KP':>9}  Risk Match"
    print(header); print("-"*len(header))
    shown=rows if args.top<=0 else rows[:args.top]
    for i,r in enumerate(shown,1):
        flag="YES" if r["total"]>=args.threshold else ""
        print(f"{i:>3}  {r['name'][:32]:32}  {r['total']:7.2f}  {r['sift']:6.2f}  {r['orb']:5.2f}  {r['color']:7.2f}  {r['phash']:7.2f}  {r['ssim']:6.2f}  {r['shape']:7.2f}  {r['inliers']:>4}  {r['n_new']:>4}/{r['n_coll']:<4}  {r['risk']:6} {flag}")
    if args.csv:
        fieldnames=["name","path","total","sift","orb","color","phash","ssim","shape","inliers","good","n_new","n_coll","risk"]
        with open(args.csv,"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader()
            for r in rows: writer.writerow({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k in fieldnames})
        logger.info(f"CSV: {args.csv}")
    if args.json:
        clean=[{k:v for k,v in r.items() if not k.startswith("_")} for r in rows]
        with open(args.json,"w",encoding="utf-8") as f:
            json.dump({"etalon":str(etalon_path),"collections":str(coll_dir),"threshold":args.threshold,"weights":w.tolist(),"elapsed":time.perf_counter()-t0,"results":clean},f,ensure_ascii=False,indent=2)
        logger.info(f"JSON: {args.json}")
    top_vis_paths=[]
    if args.visualize and rows:
        k=max(1,args.top_vis)
        for idx in range(min(k,len(rows))):
            best=rows[idx]; extra=best.get("_extra"); img2=best.get("_img2"); kp2_pts=best.get("_kp2_pts")
            if extra is None or kp_new is None or kp2_pts is None or img2 is None: continue
            good,mask,_=extra; kp2_list=[cv2.KeyPoint(float(pt[0]),float(pt[1]),1) for pt in kp2_pts]
            inlier_matches=[m for m,ok in zip(good,mask.ravel().tolist()) if ok]
            if len(inlier_matches)>150: inlier_matches=inlier_matches[:150]
            vis=cv2.drawMatches(img_new,kp_new,img2,kp2_list,inlier_matches,None,matchColor=(0,255,0),singlePointColor=(120,120,120),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            out_name=f"top{idx+1}_matches_{Path(best['name']).stem}.jpg"; out_path=Path(out_name); cv2.imwrite(str(out_path),vis)
            logger.info(f"Визуализация top-{idx+1}: {out_path} ({best['name']}, {best['total']:.2f}%)"); top_vis_paths.append((out_path,best['name'],best['total']))
        if top_vis_paths:
            try: import shutil; shutil.copy(str(top_vis_paths[0][0]),"top1_matches.jpg")
            except Exception: pass
    if args.html:
        clean_rows=[{k:v for k,v in r.items() if not k.startswith("_")} for r in rows]
        make_html_report_v2(clean_rows,etalon_path,img_new_raw,args.threshold,top_vis_paths,Path(args.html),time.perf_counter()-t0,w.tolist(),vars(args))
    if args.cache: save_cache(cache_path,updated_cache); logger.info(f"Кэш: {len(updated_cache)} -> {cache_path}")
    dt_end=datetime.now(); elapsed=time.perf_counter()-t0
    logger.info(f"Обработано {len(rows)} за {elapsed:.1f} с | HIGH={sum(1 for r in rows if r['risk']=='HIGH')} MEDIUM={sum(1 for r in rows if r['risk']=='MEDIUM')}")
    logger.info(f"Завершение: {dt_end:%Y-%m-%d %H:%M:%S} (всего {dt_end-dt_start})")
    if log_path: logger.info(f"Лог: {log_path.resolve()}")

def main():
    args=parse_args()
    log_path=_resolve_log_path(explicit=args.log,flat=args.log_flat,no_log=args.no_log,log_dir=args.log_dir)
    if log_path is None: _main_impl(args,log_path=None); return
    try: open(log_path,"w",encoding="utf-8").close()
    except OSError as e: print(f"[WARN] {e}",file=sys.stderr); _main_impl(args,log_path=None); return
    with open(log_path,"a",encoding="utf-8",buffering=1) as log_file:
        original_stdout=sys.stdout; original_stderr=sys.stderr
        sys.stdout=Tee(original_stdout,log_file); sys.stderr=Tee(original_stderr,log_file)
        try: _main_impl(args,log_path=log_path)
        finally: sys.stdout=original_stdout; sys.stderr=original_stderr

if __name__=="__main__": main()
