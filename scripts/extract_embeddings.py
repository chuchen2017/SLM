"""Extract Geo2Vec embeddings for a geometry file.

Example:
    python scripts/extract_embeddings.py \
        --geometry data/osm/LosAngeles/training_data.gpkg \
        --output data/osm/LosAngeles/training_embeddings_self.npy \
        --pretrained data/osm/LosAngeles/LosAngeles_geo2vec_model_self_loc.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.runtime import prefer_env_libstdcxx

prefer_env_libstdcxx()

from utils.geo2vec import default_sampling_args, list2embedding, notebook_sampling_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed geometries with GeoNeuralRepresentation.list2vec.")
    parser.add_argument("--geometry", required=True, help="gpkg, shp, or geojson file.")
    parser.add_argument("--output", required=True, help="Destination .npy with one row per geometry.")
    parser.add_argument("--pretrained", default=None, help="Geo2Vec location checkpoint. Omit to train from scratch.")
    parser.add_argument("--save-model", default=None, help="Where list2vec should write the fitted encoder.")
    parser.add_argument("--repo", default=None, help="GeoNeuralRepresentation checkout. Defaults to GEO2VEC_HOME.")
    parser.add_argument("--geo-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--normalize", action="store_true", help="Scale the layer into the unit square first.")
    parser.add_argument("--match-notebook", action="store_true", help="Use the heavy sampling settings from the embedding notebook.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"{output} already exists. Pass --overwrite to recompute it.")
        return
    import geopandas as gpd

    frame = gpd.read_file(args.geometry)
    device = "cuda:0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    sampling = notebook_sampling_args(device) if args.match_notebook else default_sampling_args(device)
    embeddings = list2embedding(
        frame.geometry.tolist(),
        repo=args.repo,
        pretrained_location=args.pretrained,
        save_path=output,
        save_model_path=args.save_model,
        geo_dim=args.geo_dim,
        epochs=args.epochs,
        normalize=args.normalize,
        args=sampling,
    )
    print(f"Wrote {embeddings.shape} to {output}")


if __name__ == "__main__":
    main()
