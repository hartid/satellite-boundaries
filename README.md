# Определение границ сельскохозяйственных полей по снимкам Sentinel-1 и Sentinel-2

Автоматическое выделение контуров полей на основе пиксельной классификации и
морфологической обработки спутниковых данных.

## Постановка задачи

Разработан прототип решения задачи дистанционного мониторинга
сельскохозяйственных территорий. На вход подаются совмещённые снимки Sentinel-1
(радар) и Sentinel-2 (оптика) одной территории; на выходе — векторный слой
полигонов полей с типом покрытия (`vegetation` / `bare_soil`).

## Используемые данные

| Файл | Описание | Формат |
|---|---|---|
| `dataset.csv` (≈12 ГБ) | Попиксельная разметка: класс (`vegetation`/`bare_soil`), координаты (`lon`, `lat`), спектральные каналы Sentinel-2 (B01–B12, CLD, CLP, CLM, SCL), радарные каналы Sentinel-1 (VV, VH, incidence angle, scattering area, shadow mask), метеоданные за три даты (температура, осадки, радиация, влажность, температура и влажность почвы на разных глубинах) | CSV |
| `Sentinel-2L2A.tiff` | Мультиспектральный снимок Sentinel-2, 17 каналов: B01–B12, CLD, CLP, CLM, SCL, dataMask. Разрешение ~10–20 м/пиксель | GeoTIFF (uint16) |
| `Sentinel-1.tiff` | Радарный снимок Sentinel-1 (C-диапазон), 6 каналов: VV, VH, localIncidenceAngle, scatteringArea, shadowMask, dataMask | GeoTIFF (float32) |
| `TRUE_COLOR.tif` | RGB-композит (4 канала, альфа-канал) для визуализации | GeoTIFF (uint16) |
| `field_boundaries.gpkg` | **Результат:** 1100 полигонов полей (896 vegetation, 204 bare_soil) в EPSG:4326 | GeoPackage |
| `field_mask_fixed.tif.aux.xml` | Метаданные маски полей (файл был утерян, сохранён только .aux.xml) | XML |
| `итог.qgz` | Проект QGIS | QGIS |

### Характеристики снимков

- Размер: **4721 × 3788 пикселей**
- Пространственное разрешение: ~0.00015° (≈10–15 м)
- Охват: ~38.24°–38.95° в.д., ~53.30°–53.64° с.ш.
- CRS: географическая (EPSG:4326)

### Каналы Sentinel-2 L2A

| Индекс | Канал | Длина волны (нм) |
|---|---|---|
| B01 | Coastal aerosol | 443 |
| B02 | Blue | 490 |
| B03 | Green | 560 |
| B04 | Red | 665 |
| B05 | Vegetation Red Edge 1 | 705 |
| B06 | Vegetation Red Edge 2 | 740 |
| B07 | Vegetation Red Edge 3 | 783 |
| B08 | NIR (Near Infrared) | 842 |
| B8A | Narrow NIR | 865 |
| B09 | Water vapour | 945 |
| B11 | SWIR 1 | 1610 |
| B12 | SWIR 2 | 2190 |
| CLD | Cloud mask | — |
| CLP | Cloud probability | — |
| CLM | Cloud classification | — |
| SCL | Scene classification | — |
| dataMask | Бинарная маска данных | — |

### Каналы Sentinel-1

| № | Канал | Описание |
|---|---|---|
| 1 | VV | Вертикальная поляризация излучения/приёма |
| 2 | VH | Вертикальная/горизонтальная поляризация |
| 3 | localIncidenceAngle | Локальный угол падения |
| 4 | scatteringArea | Площадь рассеяния |
| 5 | shadowMask | Маска теней/затенения |
| 6 | dataMask | Маска валидности данных |

### Классы в данных

- `vegetation` (≈90 %) — вегетирующие посевы
- `bare_soil` (≈10 %) — открытая почва (вспаханные/убранные участки)

## Метод (пайплайн)

Последовательная обработка в скрипте `extract_fields.py`:

1. **Обучение Random Forest** — на первых 300 000 строк CSV по 4 каналам
   (B02, B03, B04, B08). `n_estimators=40`, `max_depth=12`.
2. **Чтение полного снимка** — загрузка каналов 2, 3, 4, 8 из
   `Sentinel-2L2A.tiff`.
3. **Попиксельная классификация** — предсказание класса (`vegetation` /
   `bare_soil`) случайным лесом.
4. **Выделение границ** — фильтр Собеля по NIR-каналу (B08), порог —
   88-й перцентиль. Пиксели дорог/межей «стираются», разбивая смежные
   поля.
5. **Морфологическая очистка** — `remove_small_objects` (мин. 200 пикселей)
   и `remove_small_holes` (макс. 200 пикселей).
6. **Векторизация** — `rasterio.features.shapes` → GeoDataFrame →
   `field_boundaries.gpkg`.

## Установка зависимостей

```bash
pip3 install pandas numpy rasterio geopandas shapely \
            scikit-learn scikit-image pyogrio
```

На macOS могут дополнительно потребоваться GDAL и libgdal:

```bash
brew install gdal
pip3 install --no-binary rasterio rasterio
```

## Запуск

Полный конвейер (обучение + классификация + векторизация):

```bash
python3 extract_fields.py
```

Для работы требуется 4–8 ГБ свободной оперативной памяти (Random Forest
читает 300 тыс. строк, полный снимок весит ≈300 МБ в float32 на канал).

На выходе создаётся файл `field_boundaries.gpkg`.

## Просмотр результата

Откройте `field_boundaries.gpkg` в QGIS (или другом ГИС-приложении):

```bash
# QGIS (если установлен)
open field_boundaries.gpkg
```

Либо импортируйте через QGIS: **Layer → Add Layer → Add Vector Layer**.

Стиль: раскрасьте по атрибуту `land_cover`:
- `vegetation` → зелёный
- `bare_soil` → коричневый

Для подложки используйте `TRUE_COLOR.tif`.

## Результаты

- Выделено **1100 полей**: 896 с растительностью, 204 с открытой почвой.
- Визуальное качество: большинство контуров повторяют реальные границы,
  дороги и лесополосы использованы как естественные разделители.
- Ограничения: возможны случаи слияния смежных полигонов со схожим
  спектром, требуется настройка порогов морфологии.

## Файловая структура проекта

```
.
├── dataset.csv                # Попиксельная разметка (12 ГБ)
├── extract_fields.py          # Скрипт обработки
├── Sentinel-2L2A.tiff         # Мультиспектральный снимок S-2
├── Sentinel-1.tiff            # Радарный снимок S-1
├── TRUE_COLOR.tif             # Цветной композит
├── field_boundaries.gpkg      # ⭐ Результат: полигоны полей
├── field_mask_fixed.tif.aux.xml  # Метаданные маски
├── итог.qgz                   # Проект QGIS
├── Доп к снимкам.docx         # Описание каналов спутников
├── Задание на курсовую:практику (1).pdf  # Задание
├── итогPD.docx                # Отчёт по курсовой работе
└── README.md                  # Настоящий файл
```

## Источники

- [Sentinel-2 L2A Documentation](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/)
- ESA Copernicus Programme — [Sentinel-1](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1),
  [Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
- [QGIS](https://qgis.org) — свободная ГИС для просмотра результатов
- [AI4Boundaries](https://doi.org/10.21227/d0pg-qp84), [Fields of The World](https://github.com/fieldsoftheworld/ftw-baseline) — бенчмарки для задачи выделения границ полей
