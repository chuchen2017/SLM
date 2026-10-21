## **Official codebase for *From Symbolic to Geometric: Enabling Spatial Reasoning in Large Language Models*, presented at SIGSPATIAL 2026**

SLM (Spatial Language Model) is a causal language model that reads geospatial entities as embeddings. Each entity is inserted at a `<GEO>` token. The same model answers questions about one entity (length, area) or several (distance, topology, nearest neighbor).

Training uses the city layers and questions in `data/`. Geo2Vec embeddings of those geometries are projected into the language model at each `<GEO>` token.

## Layout

```
configs/          one YAML file per city
slm/              model, data, training, and evaluation
scripts/          Geo2Vec extraction
utils/geo2vec.py  calls GeoNeuralRepresentation's list2embedding
data/osm/         city geometries and precomputed embeddings
data/instruction_dataset/
data/benchmark/   fixed evaluation prompts
```

Los Angeles and Beijing embeddings are 256-dimensional. The US embeddings are 512-dimensional. A run uses one city, because those widths do not mix in one batch.

## Setup

```bash
pip install -r requirements.txt
```

Training downloads the base model named in the config, `Qwen/Qwen3-8B`. Point `model.base_model` at a local checkpoint if you already have one.

Geo2Vec extraction needs a checkout of [GeoNeuralRepresentation](https://github.com/chuchen2017/GeoNeuralRepresentation):

```bash
export GEO2VEC_HOME=/path/to/GeoNeuralRepresentation
```

If that variable is unset, the code looks next to this repo at `../GeoNeuralRepresentation`.

## Train

From the repository root:

```bash
python -m slm.train --config configs/los_angeles.yaml
python -m slm.train --config configs/beijing.yaml
python -m slm.train --config configs/united_states.yaml
```

`runtime.cuda_visible_devices` accepts a single GPU (`"0"`) or several (`"0,1"`). Several devices relaunch one process per GPU.

The first `encoder_only_epochs` update only the geo encoder and projector. Later epochs add LoRA on every linear module, using that module's full path (for example `model.layers.0.self_attn.q_proj`) and leaving `lm_head` frozen. Checkpoints land in `train.output_dir` as `epoch_N/` and `best/`. Each checkpoint has the adapter in `llm/`, the tokenizer, and `geo_modules.pt`.

The prompt and the supervised answer are one `apply_chat_template` rendering. The role markers and the end-of-turn token are whatever that template inserts, so the default Qwen3 run and a different chat model use their own special tokens. Generation stops on the closer taken from the same template.

Set `data.include_thinking: true` to also train on `triplet_distance_QA_thinking_steps_total.json`. Those rows have no geometries, so they do not insert `<GEO>` tokens.

## Evaluate

```bash
python -m slm.evaluate --config configs/los_angeles.yaml --checkpoint outputs/slm_los_angeles/best
```

Evaluation reads `data/benchmark`. Numeric tasks report MAE on answers that parsed, plus the fraction that parsed. Topology and nearest neighbor report accuracy. The score written next to the metrics is the mean of accuracy and `1 / (1 + MAE)`.

## Embeddings

Precomputed vectors already sit beside each city split (`training_embeddings_self.npy`, `testing_embeddings_self.npy`). To recompute them:

```bash
python scripts/extract_embeddings.py \
  --geometry data/osm/LosAngeles/training_data.gpkg \
  --output data/osm/LosAngeles/training_embeddings_self.npy \
  --pretrained data/osm/LosAngeles/LosAngeles_geo2vec_model_self_loc.pth
```

`--match-notebook` uses the heavy sampling settings from the embedding notebook. Without it, sampling stays small enough to debug the call. `--normalize` scales the layer into the unit square before fitting, which is the transfer setup from that notebook.

`utils.geo2vec.list2embedding` is the function the rest of the code calls. It forwards to `GeoNeuralRepresentation.runners.list2embedding.list2vec` and drops the unused padding rows the encoder appends, so row `i` is geometry `i`.

## Data

| Path | Role |
| --- | --- |
| `data/osm/<city>/training_data.gpkg` | training geometries |
| `data/osm/<city>/testing_data.gpkg` | held-out geometries |
| `data/osm/<city>/training_embeddings_self.npy` | Geo2Vec rows for the training split |
| `data/osm/<city>/testing_embeddings_self.npy` | Geo2Vec rows for the test split |
| `data/osm/US/topo_relationships.json` | county pairs labeled touches, equals, disjoint, contains, within |
| `data/instruction_dataset/*.json` | question templates |
| `data/benchmark/*.pkl` | fixed prompts, embeddings, and answers |

Los Angeles is EPSG:32611 and Beijing is EPSG:32650, so length, area, and distance are in meters. The US layer is EPSG:4269, so those measures are in degrees, which is what the US benchmarks store.

US topology also needs `data/osm/US/all_osm_with_names.gpkg` (GEOID column). That file is large and is not part of the git history. A pair is used in training only when both counties are in the training split.

The location checkpoints used to refit embeddings are:

- `data/osm/LosAngeles/LosAngeles_geo2vec_model_self_loc.pth`
- `data/osm/Beijing/Beijing_geo2vec_model_loc_self.pth`
- `data/osm/US/US_geo2vec_model_self_loc.pth`

## What the model sees

A question template such as `How far is <location_1> from <location_0>?` becomes two entity mentions. Each mention is rewritten without brackets around the name or the phrase, for example `main street is located at <GEO>`. The embedding rows follow the `<GEO>` tokens from left to right. Repeating a location repeats its name and its embedding.

Nearest-neighbor questions come from the thinking-step file. Each training item samples a new question and a new set of entities, then the label is the sequence of distance calculations from that question's thinking trace:

```
<geo_think> Distance( <0> , <4> ) = 120 , Distance( <1> , <4> ) = 30 , Distance( <2> , <4> ) = 80 , Distance( <3> , <4> ) = 90 </geo_think>
```

Length, area, distance, and topology questions are sampled the same way: the wording is chosen at random and the geometries are chosen at random inside `__getitem__`. The training dataloader shuffles those draws. Nothing is a fixed list of queries built once before training.

Supervision is one trace:

```
<geo_think> Distance( <0> , <1> ) = 120 </geo_think>
<geo_think> LineLength( <0> ) = 2220.94 </geo_think>
<geo_think> Area( <0> ) = 4360.92 </geo_think>
<geo_think> TopoRelation( <0> , <1> ) = within </geo_think>
```

Length and area use two decimals, matching the benchmarks. Distance is an integer. The loss is only on the assistant continuation, including the end token taken from the chat template. Prompt tokens and padding are ignored.

## Benchmark only

`benchmark.py` is separate from training. Use it when you only want the questions shipped in `data/benchmark`. It needs NumPy, Shapely, and Pillow.

```python
from benchmark import benchmark

wkt = benchmark("Area_LosAngeles", index=0, entity=0, mode="TEXT")
geom = benchmark("Area_LosAngeles", index=0, entity=0, mode="POLY")
vector = benchmark("Area_LosAngeles", index=0, entity=0, mode="REP")
mask = benchmark("Area_LosAngeles", index=0, entity=0, mode="RASTER")
```

`index` selects the question. `entity` selects one geometry in that question. Leave `entity` out to get every geometry, in order.

| Mode | Returns |
| --- | --- |
| `TEXT` | WKT string |
| `POLY` | Shapely geometry |
| `REP` | Geo2Vec vector |
| `RASTER` | Float mask in `[0, 1]`, shape `(256, 256)` |

`RASTER` draws the entity into one shared box for that question, so the masks of entity 0 and entity 1 use the same coordinates. Foreground pixels are 1. Short names such as `PolyLine_Beijing`, `topo_US`, and `nearest_neighbor_Point_PolyLine_LosAngeles` match the files in `data/benchmark`.

## Citation

```bibtex
@misc{chu2026symbolicgeometricenablingspatial,
      title={From Symbolic to Geometric: Enabling Spatial Reasoning in Large Language Models},
      author={Chen Chu and Bita Azarijoo and Li Xiong and Khurram Shafique and Cyrus Shahabi},
      year={2026},
      eprint={2606.04381},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.04381},
}
```

