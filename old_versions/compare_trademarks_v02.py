import cv2
import numpy as np
from pathlib import Path

NEW_IMAGE = "new.jpg"
COLLECTIONS_DIR = "collections"

MAX_DIM = 1200       # максимальная сторона изображения
RATIO = 0.75         # коэффициент Лоу
RANSAC_THRESH = 5.0  # порог репроекции для RANSAC, пиксели


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# SIFT доступен в opencv-python начиная с 4.4 (в основном модуле)
sift = cv2.SIFT_create(nfeatures=3000)

# FLANN-матчер для вещественных дескрипторов SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def similarity_sift(img1, img2):
    """
    Сравнение по SIFT + Lowe ratio + RANSAC-гомография.
    Возвращает: score (%) по инлайерам, число good-совпадений, число инлайеров,
                число кейпоинтов у new, у файла коллекции.
    """
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    n1 = len(kp1) if kp1 else 0
    n2 = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0, 0, 0, n1, n2

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe ratio test
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0.0, len(good), 0, n1, n2

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC-гомография: ищем геометрически согласованное подмножество
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
    inliers = int(mask.sum()) if mask is not None else 0

    # Нормируем на меньшее из множеств кейпоинтов: "какая доля структуры совпала"
    denom = max(1, min(n1, n2))
    score = min(100.0, 100.0 * inliers / denom)

    return score, len(good), inliers, n1, n2


def main():
    new_path = Path(NEW_IMAGE)
    if not new_path.exists():
        print(f"Ошибка: файл {NEW_IMAGE} не найден")
        return

    img_new = load_gray(new_path)
    if img_new is None:
        print(f"Ошибка: не удалось открыть {NEW_IMAGE}")
        return

    results = []
    for path in sorted(Path(COLLECTIONS_DIR).iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            continue

        img = load_gray(path)
        if img is None:
            print(f"Пропуск {path.name}: не удалось открыть")
            continue

        score, good, inl, n1, n2 = similarity_sift(img_new, img)
        results.append((score, good, inl, n1, n2, path.name))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"{'Файл':40} {'Совпадение %':>12} {'Good':>6} {'Inliers':>8} {'KP new':>8} {'KP coll':>8}")
    print("-" * 90)
    for score, good, inl, n1, n2, name in results:
        print(f"{name:40} {score:12.2f} {good:6d} {inl:8d} {n1:8d} {n2:8d}")


if __name__ == "__main__":
    main()