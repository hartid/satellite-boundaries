import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.filters import sobel
from skimage.morphology import remove_small_holes, remove_small_objects
from sklearn.ensemble import RandomForestClassifier

from evaluate import CLASS_NAMES, CSV_PATH, build_label_raster, evaluate_mask, split_tiles
from metrics import IGNORE_INDEX

tiff_path = "Sentinel-2L2A.tiff"
output_gpkg = "field_boundaries.gpkg"
metrics_json = "metrics_extract_fields.json"

TRAIN_SAMPLES = 300_000
# Каналы растра (1-based): B02, B03, B04, B08 — те же, что S2_B02..S2_B08 в dataset.csv
BAND_INDEXES = [2, 3, 4, 8]


def main():
    print("=== СТАРТ ИНТЕЛЛЕКТУАЛЬНОГО РАЗДЕЛЕНИЯ ПОЛЕЙ (ВЕРСИЯ 4.1) ===")

    if not os.path.exists(CSV_PATH) or not os.path.exists(tiff_path):
        raise FileNotFoundError("Убедись, что файлы dataset.csv и Sentinel-2L2A.tiff находятся в папке скрипта!")

    # 1. РАЗМЕТКА И ПРОСТРАНСТВЕННОЕ РАЗБИЕНИЕ
    print("Шаг 1: Загрузка разметки и разбиение тайлов на train/test...")
    labels, tiles = build_label_raster()
    train_mask, test_mask = split_tiles(tiles)
    print(f"   Train: {train_mask.sum():,} пикселей | Test: {test_mask.sum():,} пикселей".replace(",", " "))

    # 2. ЧТЕНИЕ КАНАЛОВ
    print("\nШаг 2: Чтение спектральных каналов...")
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
        img_shape = (src.height, src.width)
        b02, b03, b04, b08 = (src.read(i).astype(np.float32) for i in BAND_INDEXES)

    # Значения каналов в растре и в dataset.csv совпадают, поэтому признаки берём прямо из растра
    X_img = np.nan_to_num(np.stack([b02.ravel(), b03.ravel(), b04.ravel(), b08.ravel()], axis=1), nan=0.0)

    # 3. ОБУЧЕНИЕ МОДЕЛИ (только на train-тайлах)
    print("\nШаг 3: Обучение классификатора на train-тайлах...")
    train_idx = np.flatnonzero(train_mask.ravel() & (labels.ravel() != IGNORE_INDEX))
    rng = np.random.default_rng(42)
    train_idx = rng.choice(train_idx, size=min(TRAIN_SAMPLES, len(train_idx)), replace=False)

    rf = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_img[train_idx], labels.ravel()[train_idx])
    print("Модель натренирована.")

    # 4. КЛАССИФИКАЦИЯ
    print("\nШаг 4: Первичная пиксельная классификация...")
    preds_id = rf.predict(X_img).astype(np.int16).reshape(img_shape)

    # 5. ВЫДЕЛЕНИЕ ГРАНИЦ И РАЗРЕЗАНИЕ БЛОКОВ (МЕТОД СОБЕЛЯ)
    print("\nШаг 5: Поиск резких переходов и разрезание смежных полей...")
    # Нормализуем инфракрасный канал для корректного поиска перепадов яркости
    b08_norm = (b08 - np.min(b08)) / (np.max(b08) - np.min(b08) + 1e-5)
    # Фильтр Собеля находит дороги, межи и лесополосы
    edges = sobel(b08_norm)

    # Выделяем топ-12% самых резких границ (линии дорог и контуры разделения)
    edge_mask = edges > np.percentile(edges, 88)

    clean_preds_id = np.full_like(preds_id, -1)
    for cls_id in CLASS_NAMES:
        mask = preds_id == cls_id

        # КРИТИЧЕСКИЙ ШАГ: Стираем пиксели дорог, превращая монолит в изолированные поля
        mask[edge_mask] = False

        # Морфологическая очистка изолированных объектов
        mask = remove_small_objects(mask, min_size=200)  # убираем точечный шум
        mask = remove_small_holes(mask, area_threshold=200)  # латаем дыры внутри полей

        clean_preds_id[mask] = cls_id

    # 6. МЕТРИКИ КАЧЕСТВА НА ОТЛОЖЕННЫХ ТАЙЛАХ
    print("\nШаг 6: Оценка качества (IoU / F1) на test-тайлах...")
    evaluate_mask(preds_id, labels, test_mask, title="RandomForest, пиксельная классификация")
    print()
    evaluate_mask(
        clean_preds_id, labels, test_mask,
        title="После разрезания границ и морфологии",
        out_path=metrics_json,
    )

    # 7. ВЕКТОРИЗАЦИЯ ОДНОРОДНЫХ ПОЛЕЙ
    print("\nШаг 7: Векторизация разделенных контуров полей...")
    records = [
        {"geometry": shape(geometry), "land_cover": CLASS_NAMES[int(value)]}
        for geometry, value in shapes(clean_preds_id, mask=(clean_preds_id != -1), transform=transform)
        if int(value) in CLASS_NAMES
    ]
    print(f"Успешно выделено уникальных изолированных полей: {len(records)}")

    # СОХРАНЕНИЕ С КОРРЕСПОНДИРУЮЩЕЙ СИСТЕМОЙ КООРДИНАТ
    if records:
        # Если в TIFF нет CRS, ставим стандартную географическую WGS84 (совпадает с lat/lon из CSV)
        gdf = gpd.GeoDataFrame(records, columns=["geometry", "land_cover"], geometry="geometry", crs=crs or "EPSG:4326")
        gdf.to_file(output_gpkg, layer="detected_fields", driver="GPKG")
        print(f"\n=== Файл {output_gpkg} успешно обновлен ===")
    else:
        print("\nОшибка сегментации. Проверь структуру входных матриц.")


if __name__ == "__main__":
    main()
