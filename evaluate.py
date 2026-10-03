"""Оценка качества сегментации по разметке из dataset.csv.

Эталон: колонка `class` в dataset.csv (попиксельно, координаты x/y — пиксели
растра Sentinel-2L2A.tiff). Классы: bare_soil, vegetation.

Примеры:
    python evaluate.py fields_segmented.geojson
    python evaluate.py fields_with_sar.geojson field_boundaries.gpkg --subset all
"""

import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from metrics import IGNORE_INDEX, format_report, save_report, segmentation_report

CSV_PATH = "dataset.csv"
S2_RASTER = "Sentinel-2L2A.tiff"
LABELS_CACHE = "labels_cache.npz"

# 0 — нет разметки
CLASS_NAMES = {1: "bare_soil", 2: "vegetation"}
CLASS_IDS = {name: cid for cid, name in CLASS_NAMES.items()}

# Классы NDVI-сегментации (satilite.py, segmentation_with_sar.py) → классы эталона
NDVI_CLASS_MAPPING = {1: "bare_soil", 2: "vegetation", 3: "vegetation"}

TEST_SIZE = 0.2
SEED = 42


def build_label_raster(csv_path=CSV_PATH, raster_path=S2_RASTER, cache_path=LABELS_CACHE):
    """Растр эталонных классов и номеров тайлов (с кэшированием в .npz).

    Чтение dataset.csv (~12 ГБ) занимает ~1–2 минуты, поэтому результат кэшируется.
    Кэш пересобирается, если CSV изменился.
    """
    stamp = np.array([os.path.getsize(csv_path), int(os.path.getmtime(csv_path))])
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if np.array_equal(cached["stamp"], stamp):
            return cached["labels"], cached["tiles"]

    with rasterio.open(raster_path) as src:
        shape = (src.height, src.width)

    labels = np.full(shape, IGNORE_INDEX, dtype=np.uint8)
    tiles = np.full(shape, -1, dtype=np.int16)

    print(f"   Сборка растра эталона из {csv_path} (один раз, затем берётся из кэша)...")
    reader = pd.read_csv(
        csv_path,
        usecols=["class", "x", "y", "tile_id"],
        dtype={"class": "category", "x": np.int32, "y": np.int32, "tile_id": np.int16},
        chunksize=2_000_000,
    )
    for chunk in reader:
        codes = chunk["class"].map(CLASS_IDS)
        if codes.isna().any():
            unknown = sorted(set(chunk.loc[codes.isna(), "class"]))
            raise ValueError(f"Неизвестные классы в {csv_path}: {unknown}")
        y, x = chunk["y"].to_numpy(), chunk["x"].to_numpy()
        labels[y, x] = codes.to_numpy(dtype=np.uint8)
        tiles[y, x] = chunk["tile_id"].to_numpy()

    np.savez_compressed(cache_path, labels=labels, tiles=tiles, stamp=stamp)
    return labels, tiles


def split_tiles(tiles, test_size=TEST_SIZE, seed=SEED):
    """Пространственное разбиение: целые тайлы уходят либо в train, либо в test.

    Соседние пиксели почти одинаковы, поэтому случайное разбиение по пикселям
    дало бы завышенные метрики. Возвращает (train_mask, test_mask).
    """
    tile_ids = np.unique(tiles[tiles >= 0])
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(tile_ids) * test_size))
    test_ids = rng.choice(tile_ids, size=n_test, replace=False)
    test_mask = np.isin(tiles, test_ids)
    train_mask = (tiles >= 0) & ~test_mask
    return train_mask, test_mask


def rasterize_prediction(vector_path, raster_path=S2_RASTER):
    """Переводит полигоны результата в растр классов эталона (-1 — не покрыто)."""
    gdf = gpd.read_file(vector_path)

    if "land_cover" in gdf.columns:          # extract_fields.py
        names = gdf["land_cover"]
    elif "class_id" in gdf.columns:          # satilite.py, segmentation_with_sar.py
        names = gdf["class_id"].map(NDVI_CLASS_MAPPING)
    else:
        raise ValueError(f"{vector_path}: нет колонки land_cover или class_id")
    values = names.map(CLASS_IDS)

    with rasterio.open(raster_path) as src:
        shape, transform = (src.height, src.width), src.transform
        crs = src.crs or "EPSG:4326"     # у растра нет CRS, координаты — градусы WGS84

    if gdf.crs is not None:
        gdf = gdf.to_crs(crs)

    pairs = [(geom, int(v)) for geom, v in zip(gdf.geometry, values) if pd.notna(v) and geom is not None]
    return rasterize(pairs, out_shape=shape, transform=transform, fill=-1, dtype=np.int16)


def evaluate_mask(pred, labels, region=None, title="Метрики качества", out_path=None):
    """Считает, печатает и (опционально) сохраняет метрики для маски предсказаний."""
    y_true = labels if region is None else np.where(region, labels, IGNORE_INDEX)
    report = segmentation_report(y_true, pred, CLASS_NAMES)
    print(format_report(report, title))
    if out_path:
        save_report(report, out_path)
        print(f"   Сохранено: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="IoU/F1 для результатов сегментации")
    parser.add_argument("predictions", nargs="+", help="GeoJSON/GPKG с результатом сегментации")
    parser.add_argument(
        "--subset",
        choices=["test", "all"],
        default="test",
        help="test — только тестовые тайлы (как у RandomForest), all — все размеченные пиксели",
    )
    args = parser.parse_args()

    labels, tiles = build_label_raster()
    region = split_tiles(tiles)[1] if args.subset == "test" else None
    subset_name = "тестовые тайлы" if args.subset == "test" else "все размеченные пиксели"

    for path in args.predictions:
        pred = rasterize_prediction(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        evaluate_mask(
            pred, labels, region,
            title=f"{path} ({subset_name})",
            out_path=f"metrics_{stem}_{args.subset}.json",
        )
        print()


if __name__ == "__main__":
    main()
