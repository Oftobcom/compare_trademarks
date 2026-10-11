Готово. Файл:

**`/home/workdir/artifacts/compare_trademarks_v05.py`**

### Что добавлено в v05

| Функция | Реализация |
|---------|------------|
| **Параллелизм** | `ProcessPoolExecutor` + `as_completed`. Число воркеров — `--workers` (по умолчанию = `cpu_count`) |
| **Кэш дескрипторов** | `CachedItem` (SIFT des + kp pts, ORB des, pHash, mtime). Проверка актуальности по `st_mtime`. `--cache` / `--cache-file` |
| **HTML-отчёт** | `--html report.html` — таблица с прогресс-барами, подсветка YES, встроенная картинка top-1 (base64) |

### Запуск

```bash
# полный набор
python compare_trademarks_v05.py \
    --new logo.jpg \
    --collections base \
    --csv report.csv \
    --html report.html \
    --visualize \
    --cache \
    --workers 8 \
    --top 30 \
    --threshold 25
```

### Кратко по архитектуре

1. Эталон обрабатывается один раз в главном процессе.
2. Для каждого файла коллекции формируется задача (кортеж) → `process_one`.
3. В воркере:
   - если кэш актуален — берём дескрипторы из него;
   - иначе считаем SIFT/ORB/pHash и возвращаем новый `CachedItem`.
4. После всех задач кэш сохраняется, результаты сортируются, пишутся CSV/HTML, рисуется `top1_matches.jpg`.

Код полностью самодостаточный, зависит только от `opencv-python` и `numpy`.