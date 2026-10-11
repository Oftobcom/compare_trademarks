import cv2
from pathlib import Path

NEW_IMAGE = "new.jpg"
COLLECTIONS_DIR = "collections"

# Настройки
MAX_DIM = 1200          # уменьшать большие изображения для скорости
RATIO = 0.75            # коэффициент Лоу для отбора хороших совпадений
ORB_FEATURES = 3000     # число признаков ORB


def load_gray(path: Path):
    """Читает изображение в оттенках серого и уменьшает, если оно слишком большое."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    return img


def similarity_orb(img1, img2):
    """
    Сравнивает два изображения через ORB-признаки.
    Возвращает: процент совпадения, число хороших совпадений,
                число дескрипторов у new.jpg, число дескрипторов у файла коллекции.
    """
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0, 0, len(des1) if des1 is not None else 0, len(des2) if des2 is not None else 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good.append(m)

    # Dice-коэффициент по количеству дескрипторов:
    # 100% — если все признаки совпали, 0% — если совпадений нет.
    score = (2.0 * len(good) / (len(des1) + len(des2))) * 100.0

    return score, len(good), len(des1), len(des2)


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

    for path in Path(COLLECTIONS_DIR).iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            continue

        img = load_gray(path)
        if img is None:
            print(f"Пропуск {path.name}: не удалось открыть")
            continue

        score, good, n1, n2 = similarity_orb(img_new, img)
        results.append((score, good, n1, n2, path.name))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"{'Файл':40} {'Совпадение %':>12} {'Good':>6} {'KP new':>8} {'KP coll':>8}")
    print("-" * 80)

    for score, good, n1, n2, name in results:
        print(f"{name:40} {score:12.2f} {good:6d} {n1:8d} {n2:8d}")


if __name__ == "__main__":
    main()