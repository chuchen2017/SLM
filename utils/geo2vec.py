"""Call ``GeoNeuralRepresentation.runners.list2embedding.list2vec`` on a geometry list.

The city notebooks import that function as ``list2embedding`` and use it to fit one
vector per entity. Rows are returned in the same order as the input geometries.
Geo2Vec's embedding table is longer than the input (it reserves unused ids); this
wrapper keeps the prefix that lines up with the geometries.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from shapely.affinity import scale, translate


def default_repo() -> Path:
    configured = os.environ.get("GEO2VEC_HOME")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "GeoNeuralRepresentation"


def normalize_geometries(geometries, size: tuple[float, float] | None = None):
    """Translate a layer to the origin and scale it into the unit square.

    Distances stay proportional. ``size`` forces the scale, which is how the
    embedding notebook transfers a model between two cities.
    """
    bounds = [geometry.bounds for geometry in geometries]
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    width = max_x - min_x or 1e-9
    height = max_y - min_y or 1e-9
    if size is not None:
        width, height = size
    normalized = []
    for geometry in geometries:
        moved = translate(geometry, xoff=-min_x, yoff=-min_y)
        normalized.append(scale(moved, xfact=1 / width, yfact=1 / height, origin=(0, 0)))
    return normalized, (width, height)


def default_sampling_args(device: str = "cpu") -> Namespace:
    """Sampling settings that fit a workstation. See ``notebook_sampling_args``."""
    return Namespace(
        device=device,
        num_workers=4,
        num_process=4,
        batch_size=4096,
        samples_perUnit_location=4000,
        point_sample_location=10,
        sample_band_width_location=0.1,
        uniformed_sample_perUnit_location=30,
        epochs_location=2,
        num_layers_location=8,
        z_size_location=256,
        hidden_size_location=256,
        num_freqs_location=16,
        code_reg_weight_location=0.0,
        weight_decay_location=0.01,
        polar_fourier_location=False,
        log_sampling_location=False,
        training_ratio_location=0.95,
        samples_perUnit_shape=100,
        point_sample_shape=20,
        sample_band_width_shape=0.1,
        uniformed_sample_perUnit_shape=20,
        epochs_shape=2,
        num_layers_shape=8,
        z_size_shape=256,
        hidden_size_shape=256,
        num_freqs_shape=8,
        code_reg_weight_shape=1.0,
        weight_decay_shape=0.01,
        polar_fourier_shape=False,
        log_sampling_shape=True,
        training_ratio_shape=0.95,
        device_shape=device,
    )


def notebook_sampling_args(device: str = "cuda:0") -> Namespace:
    """The heavier settings used to build the embeddings already stored in ``data/osm``."""
    args = default_sampling_args(device)
    args.num_workers = 16
    args.num_process = 64
    args.batch_size = 1024 * 20
    args.samples_perUnit_location = 1_000_000
    args.point_sample_location = 50
    args.sample_band_width_location = 0.05
    args.uniformed_sample_perUnit_location = 50
    return args


def _load_list2vec(repo: Path):
    repo = Path(repo).resolve()
    if not (repo / "runners" / "list2embedding.py").is_file():
        raise FileNotFoundError(
            f"GeoNeuralRepresentation was not found at {repo}. "
            "Clone https://github.com/chuchen2017/GeoNeuralRepresentation and set GEO2VEC_HOME."
        )
    parent = str(repo.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from GeoNeuralRepresentation.runners.list2embedding import list2vec

    return list2vec


def list2embedding(
    geometries,
    *,
    repo: str | Path | None = None,
    pretrained_location: str | Path | None = None,
    pretrained_shape: str | Path | None = None,
    save_path: str | Path | None = None,
    save_model_path: str | Path | None = None,
    geo_dim: int = 256,
    epochs: int = 2,
    location_learning: bool = True,
    shape_learning: bool = False,
    normalize: bool = False,
    normalize_size: tuple[float, float] | None = None,
    args: Namespace | None = None,
) -> np.ndarray:
    """Embed ``geometries`` in input order.

    ``pretrained_location`` is a Geo2Vec ``*_loc.pth`` checkpoint. Pass ``None`` to
    train the location encoder from scratch. ``list2vec`` treats any value other
    than ``None`` as a file path, including ``False``, so this wrapper never
    forwards ``False``.
    """
    geometries = list(geometries)
    if normalize:
        geometries, _ = normalize_geometries(geometries, size=normalize_size)
    if pretrained_location is not None:
        pretrained_location = str(pretrained_location)
    shape_checkpoint = None if not shape_learning or pretrained_shape is None else str(pretrained_shape)
    list2vec = _load_list2vec(Path(repo) if repo is not None else default_repo())
    embeddings = list2vec(
        geometries,
        from_pretrained_model_location=pretrained_location,
        from_pretrained_model_shape=shape_checkpoint if shape_checkpoint is not None else False,
        save_model_path=None if save_model_path is None else str(save_model_path),
        Geo_dim=geo_dim,
        num_epoch=epochs,
        location_learning=location_learning,
        shape_learning=shape_learning,
        save_file_name=None if save_path is None else str(save_path),
        args=args or default_sampling_args(),
    )
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(geometries):
        if embeddings.shape[0] < len(geometries):
            raise ValueError(
                f"Geo2Vec returned {embeddings.shape[0]} rows for {len(geometries)} geometries."
            )
        embeddings = np.ascontiguousarray(embeddings[: len(geometries)])
        if save_path is not None:
            np.save(save_path, embeddings)
    return embeddings
