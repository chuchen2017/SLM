"""Read one entity from the benchmark pickles shipped in this repository.

This file does not import the training package. It only needs the standard
library, NumPy, Shapely, and Pillow.

>>> from benchmark import benchmark
>>> benchmark("Area_LosAngeles", index=0, entity=0, mode="TEXT")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

_DIRECTORY = Path(__file__).resolve().parent / "data" / "benchmark"
_MODES = ("TEXT", "POLY", "REP", "RASTER")


def benchmark(
    name,
    index=0,
    mode="POLY",
    entity=None,
    benchmark_dir=None,
    image_size=(256, 256),
):
    """Return one view of the entities in a benchmark question.

    ``name`` is a file name, a path, or a short name such as ``Area_LosAngeles``.
    ``index`` selects the question. ``entity`` selects one entity in that
    question; omit it to get every entity, in order.

    ``mode`` is one of:

    - ``TEXT``: WKT string
    - ``POLY``: Shapely geometry
    - ``REP``: Geo2Vec vector, a 1-D NumPy array
    - ``RASTER``: float32 mask in ``[0, 1]``, shape ``image_size``. Every entity
      in the question is drawn in the same box, so the masks line up.
    """
    chosen = str(mode).upper()
    if chosen not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    record = _load_record(name, index, benchmark_dir)
    geometries = [wkt.loads(text) for text in record["geo_representations"]]
    if entity is None:
        chosen_ids = list(range(len(geometries)))
    else:
        chosen_ids = [int(entity)]
        if chosen_ids[0] < 0 or chosen_ids[0] >= len(geometries):
            raise IndexError(f"entity {entity} is outside 0..{len(geometries) - 1}.")

    if chosen == "TEXT":
        values = [record["geo_representations"][item] for item in chosen_ids]
    elif chosen == "POLY":
        values = [geometries[item] for item in chosen_ids]
    elif chosen == "REP":
        values = [np.asarray(record["geo_embeddings"][item], dtype=np.float32).reshape(-1) for item in chosen_ids]
    else:
        box = _shared_bounds(geometries)
        values = [_rasterize(geometries[item], box, image_size) for item in chosen_ids]
    if entity is None:
        return values
    return values[0]


def _load_record(name, index, benchmark_dir):
    path = _find_file(name, benchmark_dir)
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if index < 0 or index >= len(records):
        raise IndexError(f"index {index} is outside 0..{len(records) - 1} for {path.name}.")
    return records[index]


def _find_file(name, benchmark_dir):
    directory = Path(benchmark_dir) if benchmark_dir else _DIRECTORY
    raw = Path(name)
    if raw.is_file():
        return raw
    stem = raw.stem if raw.suffix == ".pkl" else str(name)
    short = stem[10:] if stem.startswith("benchmark_") else stem
    for candidate in (
        directory / stem,
        directory / f"{stem}.pkl",
        directory / f"benchmark_{short}.pkl",
    ):
        if candidate.is_file():
            return candidate
    wanted = short.lower()
    for candidate in sorted(directory.glob("*.pkl")):
        candidate_short = candidate.stem[10:] if candidate.stem.startswith("benchmark_") else candidate.stem
        if candidate_short.lower() == wanted:
            return candidate
    raise FileNotFoundError(f"No benchmark named {name!r} in {directory}.")


def _shared_bounds(geometries):
    """One box for every entity in the question, padded so edge strokes stay visible."""
    minx = min(geometry.bounds[0] for geometry in geometries)
    miny = min(geometry.bounds[1] for geometry in geometries)
    maxx = max(geometry.bounds[2] for geometry in geometries)
    maxy = max(geometry.bounds[3] for geometry in geometries)
    span_x = max(maxx - minx, 1e-9)
    span_y = max(maxy - miny, 1e-9)
    pad_x = 0.05 * span_x
    pad_y = 0.05 * span_y
    return (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)


def _transform_coords(coords, minx, miny, maxx, maxy, image_size, padding=1):
    width = maxx - minx
    height = maxy - miny
    usable_w = max(1, image_size[0] - 1 - 2 * padding)
    usable_h = max(1, image_size[1] - 1 - 2 * padding)
    scale_x = usable_w / width if width > 0 else float("inf")
    scale_y = usable_h / height if height > 0 else float("inf")
    scale = min(scale_x, scale_y)
    scaled_w = width * scale
    scaled_h = height * scale
    offset_x = padding + (usable_w - scaled_w) / 2.0
    offset_y = padding + (usable_h - scaled_h) / 2.0
    points = []
    for x_coord, y_coord in coords:
        pixel_x = (x_coord - minx) * scale + offset_x
        pixel_y = image_size[1] - 1 - ((y_coord - miny) * scale + offset_y)
        points.append((int(round(pixel_x)), int(round(pixel_y))))
    return points


def _clip_point(point, image_size):
    x_coord, y_coord = point
    return (
        max(0, min(image_size[0] - 1, x_coord)),
        max(0, min(image_size[1] - 1, y_coord)),
    )


def _mark_one_pixel(draw, point, image_size, fill=255):
    draw.point(_clip_point(point, image_size), fill=fill)


def _draw_polygon(draw, poly, minx, miny, maxx, maxy, image_size, padding=1):
    if poly.is_empty:
        return
    representative = poly.representative_point()
    pixel = _transform_coords([(representative.x, representative.y)], minx, miny, maxx, maxy, image_size, padding)[0]
    _mark_one_pixel(draw, pixel, image_size, fill=255)
    exterior = _transform_coords(list(poly.exterior.coords), minx, miny, maxx, maxy, image_size, padding)
    unique_exterior = list(dict.fromkeys(exterior))
    if len(unique_exterior) < 3:
        for point in unique_exterior:
            _mark_one_pixel(draw, point, image_size, fill=255)
        return
    draw.polygon(exterior, fill=255)
    for interior in poly.interiors:
        hole = _transform_coords(list(interior.coords), minx, miny, maxx, maxy, image_size, padding)
        if len(list(dict.fromkeys(hole))) >= 3:
            draw.polygon(hole, fill=0)


def _rasterize(geom, bounding_box, image_size=(256, 256), padding=1, line_width=2):
    """Binary mask of one geometry, normalized to ``[0, 1]``. Foreground is 1."""
    if geom is None or geom.is_empty:
        return np.zeros(image_size, dtype=np.float32)
    minx, miny, maxx, maxy = bounding_box
    image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(image)
    if maxx == minx and maxy == miny:
        _mark_one_pixel(draw, (image_size[0] // 2, image_size[1] // 2), image_size, fill=255)
        return np.array(image, dtype=np.float32) / 255.0

    def rasterize_one(geometry):
        if geometry is None or geometry.is_empty:
            return
        if isinstance(geometry, Point):
            point = _transform_coords([(geometry.x, geometry.y)], minx, miny, maxx, maxy, image_size, padding)[0]
            _mark_one_pixel(draw, point, image_size, fill=255)
        elif isinstance(geometry, LineString):
            coords = _transform_coords(list(geometry.coords), minx, miny, maxx, maxy, image_size, padding)
            unique_coords = list(dict.fromkeys(coords))
            if len(unique_coords) < 2:
                for point in unique_coords:
                    _mark_one_pixel(draw, point, image_size, fill=255)
                return
            for point in unique_coords:
                _mark_one_pixel(draw, point, image_size, fill=255)
            draw.line(coords, fill=255, width=line_width)
        elif isinstance(geometry, MultiLineString):
            for line in geometry.geoms:
                rasterize_one(line)
        elif isinstance(geometry, Polygon):
            _draw_polygon(draw, geometry, minx, miny, maxx, maxy, image_size, padding)
        elif isinstance(geometry, MultiPolygon):
            for polygon in geometry.geoms:
                rasterize_one(polygon)
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                rasterize_one(part)
        else:
            raise TypeError(f"Unsupported geometry type {geometry.geom_type}.")

    rasterize_one(geom)
    return np.array(image, dtype=np.float32) / 255.0
