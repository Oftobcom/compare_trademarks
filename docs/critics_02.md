Готово. Новый файл:  
**`/home/workdir/artifacts/compare_trademarks_v04.py`**

### Ключевые улучшения относительно v03

| Аспект | v03 | v04 |
|--------|-----|-----|
| Метрики | SIFT + Color + pHash | **SIFT + ORB + Color + pHash + SSIM** |
| Предобработка | нет | CLAHE + Gaussian blur |
| pHash | 8×8 (64 бит) | **16×16 (256 бит)** |
| SIFT-score | `raw × conf` | `raw × conf × purity` (учёт доли инлайеров) |
| Геометрия после RANSAC | только счётчик | **выравнивание + SSIM** |
| Прогресс | нет | строка прогресса |
| Кэш | нет | заготовка `--cache` |
| Форматы | jpg/png/bmp/webp | + tif/tiff |
| Веса по умолчанию | 0.60 / 0.25 / 0.15 | 0.40 / 0.15 / 0.20 / 0.10 / 0.15 |

### Запуск

```bash
# базовый
python compare_trademarks_v04.py

# с визуализацией и отчётом
python compare_trademarks_v04.py --new logo.jpg --collections base \
    --csv report.csv --visualize --top 15 --threshold 30

# свои веса (SIFT ORB Color pHash SSIM)
python compare_trademarks_v04.py --weights 0.50 0.10 0.15 0.10 0.15
```

### Рекомендации по настройке

- **Логотипы с чёткой геометрией** → поднять вес SIFT и SSIM (`0.50 0.10 0.15 0.05 0.20`).
- **Цветные знаки / упаковка** → поднять Color (`0.35 0.10 0.30 0.15 0.10`).
- **Много шума / JPEG** → `--clahe 3.0`.
- **Очень большие коллекции** → можно добавить кэширование дескрипторов (структура уже есть).

Если нужно — могу сразу добавить параллельную обработку (`concurrent.futures`), кэш дескрипторов SIFT/ORB или экспорт в HTML-отчёт.