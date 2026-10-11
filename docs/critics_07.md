Проанализировал все 7 файлов из папки `/mnt/data/` и `/home/code-interpreter/`.

> **Важно:** файл `compare_trademarks_v06.py` у тебя сейчас перезаписан и содержит только 1 строку-заглушку `#!/usr/bin/env python3 # (копия ...)`. Фактически рабочими остались v01-v05 и v07. v07 — это и есть финальная, исправленная версия v06.

### Эволюция по версиям

| Версия | Метрики | Загрузка | Параллель | Кэш | Отчеты | Главная проблема |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v01** | Только ORB `2*good/(n1+n2)` | `IMREAD_GRAYSCALE`, только `.jpg` | нет | нет | консоль | Ложные срабатывания, не учитывает поворот/масштаб |
| **v02** | SIFT + FLANN + Lowe `0.75` + RANSAC `5.0px`, score=`inliers/min(n1,n2)` | то же | нет | нет | консоль `Файл / Совпадение%` | Нет цвета и композиции. Прорыв по геометрии |
| **v03** | **3 метрики**: SIFT% + Color% (HSV 50x60 Bhattacharyya) + pHash% (DCT 8x8), веса `0.6/0.25/0.15` | `IMREAD_COLOR`, `max_dim` | нет | нет | CSV + `top1_matches.jpg` + `argparse` `--threshold --weights --visualize` | Появилась архитектура. Нет ORB/SSIM, нет параллели |
| **v04** | **5 метрик**: + ORB% + SSIM% (custom Gaussian) | `load_bgr`, `max_dim` | `ProcessPoolExecutor` | `pickle` `.tm_cache.pkl` по `mtime` | HTML с `bar()` + CSV | Кэш ломается если файл перезаписан за 1 сек, `file_key` только по пути, только `iterdir()` |
| **v05** | Те же 5, веса `0.40/0.15/0.20/0.10/0.15`, CLAHE `2.0` + blur | тот же | + логирование воркеров через `Tee` + `_init_worker_logging` | атомарная запись `.tmp` + `abs(mtime diff)` | HTML + CSV + `logs/compare_trademarks_<ts>.log` | Логирование через `Tee` до `logging`, нет поддержки PNG с альфой |
| **v06 (сгенерированный)** | **6 метрик**: + Shape% (matchShapes + Hu), веса `0.35/0.10/0.15/0.15` | `load_bgr_smart` — PNG 4 канала → композит на белый, grayscale → BGR | + `tqdm` | v2 `mtime+size+version` LRU 5000 | JSON + CSV + HTML v2 интерактивный + `top-N` визуализаций | Shape считался на НЕвыровненных изображениях, `setup_logging` вызывался до установки `Tee` → логи не попадали в файл |
| **v07 (финальный, приложен)** | Те же 6 метрик, но **исправлено**: | `load_bgr_smart` + `WHITE_THRESH_DEFAULT=240` | `ProcessPoolExecutor` + fallback `ThreadPoolExecutor` + `_HAS_TQDM` | v2 + `is_valid()` проверяет `size`, LRU | CSV, JSON, HTML v2 с сортировкой, фильтром, KPI `HIGH/MEDIUM/LOW`, галерея top-K base64 | **Эталонная версия** |

### Что именно исправлено в v07 относительно v06

Это ключевой файл, смотри строки 310-670:

1.  **Shape после выравнивания** — главное исправление:
    ```python
    # v06: shape_score(new_gray, gray2) - на исходных
    # v07: 
    aligned = cv2.warpPerspective(gray2, inv(H), ...)
    mask_img = warpPerspective(ones*255, inv(H), ...)
    new_for_shape[~valid] = 255 # заливаем артефакты warp белым
    shape_val = shape_score(new_for_shape, aligned_for_shape)
    ```
    Теперь SSIM и Shape считаются на одной и той же выровненной паре с маской, а черные углы от `warpPerspective` не портят контуры.

2.  **Логгер после Tee:**
    ```python
    # v06: setup_logging() до Tee
    # v07:
    def setup_logger():
        sh = StreamHandler(sys.stdout) # sys.stdout уже Tee
    ```
    Вызывается в `_main_impl` уже после того как `sys.stdout = Tee(...)`.

3.  **Кэш:** `is_valid()` теперь проверяет и `size`, и `version==CACHE_VERSION`, `load_cache` фильтрует только `dataclass_fields`.

4.  **Color:** добавлен параметр `white_thresh` и вынесен в константу, маска отключается если белого <5% площади.

5.  **Кли:** появились `--recursive`, `--ignore-white`, `--white-thresh`, `--use-threads`.

### Архитектура v07 (рекомендую как базовую)

**Загрузка:** `IMREAD_UNCHANGED` → если 4 канала, `alpha*fg + (1-alpha)*255` → белый фон, а не черный как в v01-v05.

**Препроцесс:** `CLAHE clipLimit=2.0` по L-каналу LAB + `GaussianBlur(3,3)` — стабилизирует SIFT на плохих сканах.

**6 метрик:**

- **SIFT** — `inliers/min(n1,n2) * conf * (0.7+0.3*purity)` где `conf = min(1, inliers/min_inliers)` и `purity = inliers/good`
- **ORB** — `2*good/(n1+n2)*100`
- **Color** — HSV 36x32 гистограмма, Bhattacharyya `1-dist`
- **pHash** — DCT 16x16*4 → median без DC
- **SSIM** — custom без skimage, на выровненном кропе
- **Shape** — `matchShapes(I1)` + Hu-моменты в лог-шкале `exp(-dist*2)` и `exp(-hu_dist*0.5)` → `0.6*shape_match + 0.4*hu_score`

`Total = Σ weights[i]*metric[i]`

**Риски:** `classify_risk()` — `HIGH>=60%`, `MEDIUM>=30%` настраиваются.

**Отчеты:** HTML v2 с `data-total`, `data-name`, JS сортировкой по клику, живой фильтр, бейджи риска.

### Что делать дальше

1.  Восстанови `v06.py` из `v07.py` — сейчас он битый.
2.  Используй `v07` как финальный. Запуск:
    ```bash
    python compare_trademarks_v07.py --new new.jpg --collections collections --recursive --ignore-white --cache --csv report.csv --json report.json --html report.html --visualize --top-vis 5
    ```
3.  Если хочешь **v08**, могу добавить:
    - Папку с несколькими эталонами + матрицу `etalon x collection`
    - Эмбеддинг CLIP / DINOv2 для семантической схожести (ловит "похожий смысл", а не только форму)
    - Excel `openpyxl` с гиперссылками и миниатюрами
    - Удаление фона `rembg` перед сравнением формы

Скажи — собрать v08 с CLIP и Excel?