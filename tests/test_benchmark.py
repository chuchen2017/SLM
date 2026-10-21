"""The standalone benchmark reader does not use the training package."""

from pathlib import Path

import numpy as np
from shapely.geometry.base import BaseGeometry

from benchmark import benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_modes_for_one_entity():
    text = benchmark("Area_LosAngeles", index=0, entity=0, mode="TEXT", benchmark_dir=ROOT / "data" / "benchmark")
    poly = benchmark("Area_LosAngeles", 0, "POLY", entity=0, benchmark_dir=ROOT / "data" / "benchmark")
    vector = benchmark("Area_LosAngeles", 0, "rep", entity=0, benchmark_dir=ROOT / "data" / "benchmark")
    mask = benchmark("Area_LosAngeles", 0, "RASTER", entity=0, benchmark_dir=ROOT / "data" / "benchmark")
    assert text.startswith("POLYGON")
    assert isinstance(poly, BaseGeometry)
    assert poly.geom_type == "Polygon"
    assert vector.ndim == 1 and vector.shape[0] == 256
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert mask.max() == 1.0
    assert mask.min() == 0.0


def test_benchmark_returns_every_entity_in_order():
    geoms = benchmark(
        "nearest_neighbor_Point_PolyLine_LosAngeles",
        index=0,
        mode="POLY",
        benchmark_dir=ROOT / "data" / "benchmark",
    )
    masks = benchmark(
        "nearest_neighbor_Point_PolyLine_LosAngeles",
        index=0,
        mode="RASTER",
        benchmark_dir=ROOT / "data" / "benchmark",
    )
    assert len(geoms) == 5
    assert len(masks) == 5
    assert all(mask.shape == (256, 256) and mask.max() == 1.0 for mask in masks)
