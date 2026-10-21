from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _as_token_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            ids.extend(_as_token_id_list(item))
        return ids
    token_id = int(value)
    return [token_id] if token_id >= 0 else []


def resolve_generation_stop_token_ids(tokenizer, llm=None) -> list[int]:
    """EOS / end-of-turn ids `generate()` should stop on.

    Training labels now use the chat-template closer (`tokenizer.eos_token`, Gemma
    `<turn|>`, Harmony `<|return|>`). Model `generation_config.eos_token_id` can
    disagree (Qwen3.5 config.json lists `<|endoftext|>` while the tokenizer EOS is
    `<|im_end|>`). Merge tokenizer EOS, the model config list, and eot when present.
    """
    ids: list[int] = []
    seen: set[int] = set()

    def add(value) -> None:
        for token_id in _as_token_id_list(value):
            if token_id not in seen:
                seen.add(token_id)
                ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(llm, "generation_config", None) if llm is not None else None
    add(getattr(generation_config, "eos_token_id", None))
    eot = getattr(tokenizer, "eot_token", None)
    if isinstance(eot, str) and hasattr(tokenizer, "convert_tokens_to_ids"):
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        unk = getattr(tokenizer, "unk_token_id", None)
        if eot_id is not None and eot_id != unk:
            add(eot_id)
    # Qwen / GPT-OSS: pad is <|endoftext|>, a documented extra stop, not a pad-only id.
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token in ("<|endoftext|>", "<|end|>"):
        add(getattr(tokenizer, "pad_token_id", None))
    return ids


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = False,
        polar_fourier: bool = False,
        l_min: float | None = None,
        l_max: float | None = None,
        more_axis: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = False #True #include_input
        self.log_sampling = False #log_sampling
        self.more_axis = True #True #more_axis
        self.polar_fourier = False #polar_fourier

        # Use the *effective* (post-override) flags, and gate more_axis on input_dims==2
        # to match what forward() actually does.
        feature_input_dims = (
            input_dims * 2 if (self.more_axis and input_dims == 2) else input_dims
        )
        self.register_buffer("freq_bands", self._create_freq_bands(l_min, l_max))

        self.output_dims = 2 * num_freqs * feature_input_dims
        if self.include_input:
            self.output_dims += feature_input_dims
        if self.polar_fourier and input_dims == 2:
            self.output_dims += input_dims + 2 * num_freqs

    def _create_freq_bands(self, l_min: float | None, l_max: float | None) -> torch.Tensor:
        if self.log_sampling:
            if l_min is not None and l_max is not None:
                return (2.0 ** torch.linspace(l_min, l_max, self.num_freqs) * torch.pi).unsqueeze(1)
            return (2.0 ** torch.linspace(-10.0, self.num_freqs - 1, self.num_freqs) * torch.pi).unsqueeze(1)
        if l_min is not None and l_max is not None:
            return torch.linspace(l_min, l_max, self.num_freqs).unsqueeze(1)
        return torch.linspace(1.0, 6.0, self.num_freqs).unsqueeze(1)
        #return torch.linspace(.01, 3.0, self.num_freqs).unsqueeze(1)

    @staticmethod
    def more_axis_fn(xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        x1 = (x + y) / (2**0.5)
        y1 = (y - x) / (2**0.5)
        return torch.stack([x1, y1, x, y], dim=-1)

    def polar_fourier_encoding(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0], xy[:, 1]
        r = torch.sqrt(x**2 + y**2)
        freqs = r.unsqueeze(1) * self.freq_bands.squeeze(1)
        return torch.cat([xy, torch.sin(freqs), torch.cos(freqs)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        if self.more_axis and self.input_dims == 2:
            x = self.more_axis_fn(x)

        x_freq = x.unsqueeze(1) * self.freq_bands
        parts = [torch.sin(x_freq), torch.cos(x_freq)]
        if self.include_input:
            parts.insert(0, x.unsqueeze(1))
        out = torch.cat(parts, dim=1).reshape(x.shape[0], -1)

        if self.polar_fourier and self.input_dims == 2:
            out = torch.cat([out, self.polar_fourier_encoding(original_x)], dim=-1)
        return out


class GeoCLIP(nn.Module):
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 4096,
        num_freqs: int = 36,
        log_sampling: bool = False,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoder(
            input_dims=2,
            num_freqs=num_freqs,
            include_input=False,
            log_sampling=log_sampling,
            polar_fourier=False,
            more_axis=True,
        )
        self.geo_proj = nn.Sequential(
            nn.Linear(self.pos_encoder.output_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def forward(self, coords: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geo_feat = F.normalize(self.location_representation(coords), dim=-1)
        text_feat = F.normalize(text_emb, dim=-1)
        return geo_feat, text_feat

    def location_representation(self, coords: torch.Tensor) -> torch.Tensor:
        return self.geo_proj(self.pos_encoder(coords))


def _load_siren_backend():
    """Load `SirenNet` from the vendored LocationEncoder repo's `locationencoder/nn/siren.py`.

    Loaded directly via `importlib` from the file path (bypassing `import locationencoder...`)
    rather than as a regular package import, for two reasons:
      1. `locationencoder/__init__.py` -> `locationencoder.py` unconditionally does
         `import locationencoder.pe`, which imports `pe/spherical_harmonics.py`, which imports
         the huge `spherical_harmonics_ylm.py` (thousands of `torch.jit.script` functions) - a
         ~30s one-time cost paid at *import* time alone, even though we don't use any of it
         (see `RealSphericalHarmonics`'s docstring for why we avoid that module altogether).
      2. That same chain also requires `lightning`/`scikit-learn`/`basemap`, which aren't needed
         for `SirenNet` itself.
    `siren.py` has no relative imports, so it can be loaded standalone without pulling in the
    rest of the `locationencoder` package.
    """
    siren_path = Path(__file__).resolve().parents[2] / "locationencoder" / "locationencoder" / "nn" / "siren.py"
    if not siren_path.is_file():
        raise ImportError(
            f"spatial_division_method='sphere' requires the vendored LocationEncoder repo's "
            f"{siren_path} (and its 'einops' dependency) to be importable."
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location("_vendored_locationencoder_siren", siren_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SirenNet


class RealSphericalHarmonics(nn.Module):
    """Vectorized real spherical harmonics Y_l^m for l in [0, legendre_polys), m in [-l, l].

    Same Racah-normalized, Condon-Shortley-phase convention as the vendored LocationEncoder
    repo's `locationencoder.pe.spherical_harmonics_closed_form.SH` (verified numerically
    equivalent). Deliberately reimplemented here rather than reused from
    `locationencoder.pe.spherical_harmonics.SphericalHarmonics` ("analytic" mode), because that
    version dispatches into `legendre_polys ** 2` individually `torch.jit.script`-compiled
    functions. Each one needs ~2 independent PyTorch JIT "profiling" warm-up calls before it
    becomes fast, and since each function is only called once per forward pass, it takes 2 full
    forward passes for *all* of them to warm up. For `legendre_polys=40` (1600 terms) that means
    the first couple of forward passes of the process each stall for 1.5-3+ seconds doing JIT
    compilation - with no GPU/CPU load to show for it - instead of actually running the model
    (this is exactly the "slow first few epochs, no GPU/CPU usage" symptom). This implementation
    computes the same associated-Legendre recurrence with plain eager tensor ops shared across
    all m, so it is fast (~10-30ms for a training-size batch) from the very first call, with no
    warm-up.
    """

    def __init__(self, legendre_polys: int = 40):
        super().__init__()
        self.legendre_polys = int(legendre_polys)
        self.embedding_dim = self.legendre_polys * self.legendre_polys

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        return self._real_sh(theta, phi)

    def _real_sh(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        num_bands = self.legendre_polys
        x = theta.cos()
        one_minus_x2 = (1.0 - x * x).clamp(min=0.0)

        # Associated Legendre polynomials P_l^m(x), via the standard stable recurrence: computed
        # once per (l, m) and shared across the batch, instead of per-sample dispatch.
        legendre: dict[tuple[int, int], torch.Tensor] = {}
        for m in range(num_bands):
            if m == 0:
                p_mm = torch.ones_like(x)
            else:
                double_fact = 1.0
                for k in range(1, 2 * m, 2):
                    double_fact *= k
                p_mm = ((-1.0) ** m) * double_fact * one_minus_x2.pow(m / 2.0)
            legendre[(m, m)] = p_mm
            if m + 1 < num_bands:
                p_mmp1 = x * (2 * m + 1) * p_mm
                legendre[(m + 1, m)] = p_mmp1
                p_prev2, p_prev1 = p_mm, p_mmp1
                for l in range(m + 2, num_bands):
                    p_l = ((2 * l - 1) * x * p_prev1 - (l + m - 1) * p_prev2) / (l - m)
                    legendre[(l, m)] = p_l
                    p_prev2, p_prev1 = p_prev1, p_l

        harmonics: list[torch.Tensor] = [None] * (num_bands * num_bands)  # type: ignore[list-item]
        for l in range(num_bands):
            base = l * l + l
            for m in range(0, l + 1):
                norm = math.sqrt((2 * l + 1) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))
                p_lm = legendre[(l, m)]
                if m == 0:
                    harmonics[base] = norm * p_lm
                else:
                    scaled = math.sqrt(2.0) * norm * p_lm
                    harmonics[base + m] = scaled * (m * phi).cos()
                    harmonics[base - m] = scaled * (m * phi).sin()
        return torch.stack(harmonics, dim=-1)


class SphereLocationEncoder(nn.Module):
    """Spherical-harmonics + SIREN location encoder, in the spirit of the vendored
    LocationEncoder repo's `LocationEncoder("sphericalharmonics", "siren", hparams)`.

    Uses `RealSphericalHarmonics` (see its docstring for why that's a from-scratch, vectorized
    reimplementation rather than reuse) for the positional encoding, and the vendored repo's own
    `SirenNet` (which has no such performance issue) for the network. Expects raw (unnormalized)
    `[latitude, longitude]` pairs in degrees - matching the repo's own `lonlat` convention (it
    takes degrees, not normalized [-1, 1] coordinates).
    """

    def __init__(
        self,
        out_dim: int,
        legendre_polys: int = 40,
        dim_hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        SirenNet = _load_siren_backend()
        self.positional_encoder = RealSphericalHarmonics(legendre_polys=legendre_polys)
        self.siren = SirenNet(
            dim_in=self.positional_encoder.embedding_dim,
            dim_hidden=dim_hidden,
            dim_out=out_dim,
            num_layers=num_layers,
        )

    def location_representation(self, coords: torch.Tensor) -> torch.Tensor:
        lat, lon = coords[:, 0], coords[:, 1]
        lonlat = torch.stack([lon, lat], dim=-1)
        # The associated-Legendre recurrence overflows/NaNs in float32 at high `legendre_polys`
        # (e.g. 40); compute it in float64, then cast back down.
        embedding = self.positional_encoder(lonlat.double()).to(dtype=coords.dtype)
        return self.siren(embedding)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.location_representation(coords)


class GeoProjector(nn.Module):
    """Notebook-style MLP projector from GeoCLIP dimension into LLM hidden size."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Geo2LLMState:
    geo_clip: dict[str, torch.Tensor]
    geo_projector: dict[str, torch.Tensor]
    geo_token_id: int


class Geo2LLMLlama(nn.Module):
    """Inject learned location embeddings at `<GEO>` token positions."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        geo_clip: GeoCLIP | SphereLocationEncoder,
        geo_token_id: int,
        geo_out_dim: int,
        geo_projector_hidden_dim: int = 2048,
        spatial_division_method: str | None = None,
        num_cells: int | None = None,
        scale_geo_clip_log: GeoCLIP | None = None,
        scale_geo_clip_linear: GeoCLIP | None = None,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.geo_clip = geo_clip
        self.geo_token_id = geo_token_id
        self.geo_out_dim = geo_out_dim
        self.spatial_division_method = spatial_division_method
        self.embed_tokens = self.llm.get_input_embeddings()
        self.llm_hidden = self._get_llm_hidden_size(self.llm)
        self.geo_projector = GeoProjector(geo_out_dim, geo_projector_hidden_dim, self.llm_hidden)
        self.embedding_layer = None
        self.scale_geo_clip_log = None
        self.scale_geo_clip_linear = None
        self.scale_geo_projector_log = None
        self.scale_geo_projector_linear = None

        if spatial_division_method == "grid":
            if num_cells is None:
                raise ValueError("num_cells is required when spatial_division_method='grid'.")
            self.num_cells = num_cells
            self.embedding_layer = nn.Embedding(num_cells, geo_out_dim)
        elif spatial_division_method in ("DIRECT", "geohash"):
            # Text-only location codes in the prompt (`(lat, lon)` or a Base32 geohash); no
            # encoder injection. `geo_feature` is empty.
            pass
        elif spatial_division_method == "sphere":
            # `geo_clip` is expected to be a `SphereLocationEncoder` in this case; it plugs into
            # the same `location_representation(coords)` / `geo_projector` path as the default
            # Fourier `GeoCLIP`, so no extra state is needed here.
            pass
        elif spatial_division_method == "HYBRID":
            # Same model-side path as the default Fourier `GeoCLIP`: the prompt spells unnormalized
            # coordinates as text before `<GEO>`, and `geo_feature` carries the normalized
            # representation this encoder consumes. Nothing extra is needed here.
            pass
        elif spatial_division_method == "SCALE":
            if scale_geo_clip_log is None or scale_geo_clip_linear is None:
                raise ValueError(
                    "scale_geo_clip_log and scale_geo_clip_linear are required "
                    "when spatial_division_method='SCALE'."
                )
            self.scale_geo_clip_log = scale_geo_clip_log
            self.scale_geo_clip_linear = scale_geo_clip_linear
            self.scale_geo_projector_log = GeoProjector(geo_out_dim, geo_projector_hidden_dim, self.llm_hidden)
            self.scale_geo_projector_linear = GeoProjector(geo_out_dim, geo_projector_hidden_dim, self.llm_hidden)
        elif spatial_division_method is not None:
            raise ValueError(f"Unsupported spatial_division_method={spatial_division_method!r}")


    @staticmethod
    def _get_llm_hidden_size(llm: nn.Module) -> int:
        base = llm.base_model.model if hasattr(llm, "base_model") and hasattr(llm.base_model, "model") else llm
        cfg = base.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        raise AttributeError("Could not find hidden_size on model config.")

    def _encode_geo(self, geo_tensor: torch.Tensor, geo_attention_mask: torch.Tensor) -> torch.Tensor:
        if self.spatial_division_method == "grid":
            cell_ids = geo_tensor.squeeze(-1).long()
            encoded = self.embedding_layer(cell_ids)
            return self.geo_projector(encoded)

        batch_size, geo_len, _ = geo_tensor.shape
        flat_geo = geo_tensor.reshape(-1, 2)
        flat_valid = geo_attention_mask.reshape(-1).bool()
        encoded = torch.zeros(
            flat_geo.shape[0],
            self.geo_out_dim,
            device=geo_tensor.device,
            dtype=self.geo_projector.net[0].weight.dtype,
        )
        if flat_valid.any():
            geo_valid = self.geo_clip.location_representation(flat_geo[flat_valid])
            encoded[flat_valid] = geo_valid.to(encoded.dtype)
        encoded = encoded.view(batch_size, geo_len, self.geo_out_dim)
        return self.geo_projector(encoded)

    def _inject_geo_embeddings(
        self,
        query_ids: torch.Tensor,
        query_embeds: torch.Tensor,
        geo_embeds: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = query_embeds.clone()
        valid_counts = geo_attention_mask.sum(dim=-1)
        for batch_idx in range(query_ids.size(0)):
            token_positions = torch.nonzero(query_ids[batch_idx] == self.geo_token_id, as_tuple=False).squeeze(-1)
            geo_count = int(valid_counts[batch_idx].item())
            if token_positions.numel() < geo_count:
                raise ValueError(
                    f"Sample {batch_idx} has {token_positions.numel()} <GEO> tokens but {geo_count} geo rows."
                )
            if geo_count:
                out[batch_idx, token_positions[:geo_count]] = geo_embeds[batch_idx, :geo_count].to(out.dtype)
        return out

    def _encode_scale_geo(
        self,
        geo_tensor: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, geo_len, _ = geo_tensor.shape
        flat_geo = geo_tensor.reshape(-1, 2)
        flat_valid = geo_attention_mask.reshape(-1).bool()
        encoded_log = torch.zeros(
            flat_geo.shape[0],
            self.geo_out_dim,
            device=geo_tensor.device,
            dtype=self.scale_geo_projector_log.net[0].weight.dtype,
        )
        encoded_linear = torch.zeros_like(encoded_log)
        if flat_valid.any():
            valid_geo = flat_geo[flat_valid]
            log_feat = self.scale_geo_clip_log.location_representation(valid_geo)
            linear_feat = self.scale_geo_clip_linear.location_representation(valid_geo)
            encoded_log[flat_valid] = log_feat.to(encoded_log.dtype)
            encoded_linear[flat_valid] = linear_feat.to(encoded_linear.dtype)
        encoded_log = encoded_log.view(batch_size, geo_len, self.geo_out_dim)
        encoded_linear = encoded_linear.view(batch_size, geo_len, self.geo_out_dim)
        return (
            self.scale_geo_projector_log(encoded_log),
            self.scale_geo_projector_linear(encoded_linear),
        )

    def _inject_scale_geo_embeddings(
        self,
        query_ids: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
        geo_embeds_log: torch.Tensor,
        geo_embeds_linear: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_embeds: list[torch.Tensor] = []
        expanded_masks: list[torch.Tensor] = []
        valid_counts = geo_attention_mask.sum(dim=-1)

        for batch_idx in range(query_ids.size(0)):
            token_positions = torch.nonzero(query_ids[batch_idx] == self.geo_token_id, as_tuple=False).squeeze(-1)
            geo_count = int(valid_counts[batch_idx].item())
            if token_positions.numel() < geo_count:
                raise ValueError(
                    f"Sample {batch_idx} has {token_positions.numel()} <GEO> tokens but {geo_count} geo rows."
                )

            parts: list[torch.Tensor] = []
            mask_parts: list[torch.Tensor] = []
            start = 0
            for geo_idx, token_pos in enumerate(token_positions[:geo_count].tolist()):
                parts.append(query_embeds[batch_idx, start:token_pos])
                mask_parts.append(query_mask[batch_idx, start:token_pos])
                parts.extend(
                    [
                        geo_embeds_log[batch_idx, geo_idx].to(query_embeds.dtype).unsqueeze(0),
                        geo_embeds_linear[batch_idx, geo_idx].to(query_embeds.dtype).unsqueeze(0),
                    ]
                )
                mask_parts.append(query_mask.new_ones(2))
                start = token_pos + 1

            parts.append(query_embeds[batch_idx, start:])
            mask_parts.append(query_mask[batch_idx, start:])
            expanded_embeds.append(torch.cat(parts, dim=0))
            expanded_masks.append(torch.cat(mask_parts, dim=0))

        max_len = max(item.size(0) for item in expanded_embeds)
        padded_embeds = query_embeds.new_zeros(query_embeds.size(0), max_len, query_embeds.size(-1))
        padded_masks = query_mask.new_zeros(query_mask.size(0), max_len)
        for batch_idx, (embeds, mask) in enumerate(zip(expanded_embeds, expanded_masks)):
            padded_embeds[batch_idx, : embeds.size(0)] = embeds
            padded_masks[batch_idx, : mask.size(0)] = mask
        return padded_embeds, padded_masks

    def _prepare_query_geo_embeddings(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        geo_tensor: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeds = self.embed_tokens(query_ids)
        if self.spatial_division_method in ("DIRECT", "geohash"):
            return query_embeds, query_mask
        if self.spatial_division_method == "SCALE":
            geo_embeds_log, geo_embeds_linear = self._encode_scale_geo(geo_tensor, geo_attention_mask)
            return self._inject_scale_geo_embeddings(
                query_ids,
                query_embeds,
                query_mask,
                geo_embeds_log,
                geo_embeds_linear,
                geo_attention_mask,
            )
        geo_embeds = self._encode_geo(geo_tensor, geo_attention_mask)
        query_embeds = self._inject_geo_embeddings(query_ids, query_embeds, geo_embeds, geo_attention_mask)
        return query_embeds, query_mask

    def forward(
        self,
        query_ids: torch.Tensor,
        geo_tensor: torch.Tensor,
        labels: torch.Tensor,
        query_mask: torch.Tensor,
        label_mask: torch.Tensor,
        geo_attention_mask: torch.Tensor | None = None,
    ):
        device = query_ids.device
        geo_tensor = geo_tensor.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        query_mask = query_mask.to(device=device, non_blocking=True)
        label_mask = label_mask.to(device=device, non_blocking=True)
        if geo_attention_mask is None:
            geo_attention_mask = torch.ones(geo_tensor.shape[:2], dtype=torch.bool, device=device)
        else:
            geo_attention_mask = geo_attention_mask.to(device=device, non_blocking=True)

        query_embeds, query_mask = self._prepare_query_geo_embeddings(
            query_ids,
            query_mask,
            geo_tensor,
            geo_attention_mask,
        )
        label_embeds = self.embed_tokens(labels)

        inputs_embeds = torch.cat([query_embeds, label_embeds], dim=1)
        attention_mask = torch.cat([query_mask.long(), label_mask.long()], dim=1)

        labels_for_loss = labels.masked_fill(~label_mask.bool(), -100)
        pad_id = self.tokenizer.pad_token_id
        stop_ids = set(resolve_generation_stop_token_ids(self.tokenizer, self.llm))
        # Wipe pad copies that are not also a stop token. If pad_id == eos_id, the attention
        # mask already drops padding; masking by id would remove the EOS we just trained on.
        if pad_id is not None and pad_id not in stop_ids:
            labels_for_loss = labels_for_loss.masked_fill(labels_for_loss == pad_id, -100)
        prefix_ignore = labels_for_loss.new_full((labels.size(0), query_embeds.size(1)), -100)
        labels_full = torch.cat([prefix_ignore, labels_for_loss], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_full)

    @torch.no_grad()
    def generate(
        self,
        query_ids: torch.Tensor,
        geo_tensor: torch.Tensor,
        query_mask: torch.Tensor,
        geo_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        device = query_ids.device
        geo_tensor = geo_tensor.to(device=device, non_blocking=True)
        query_mask = query_mask.to(device=device, non_blocking=True)
        if geo_attention_mask is None:
            geo_attention_mask = torch.ones(geo_tensor.shape[:2], dtype=torch.bool, device=device)
        else:
            geo_attention_mask = geo_attention_mask.to(device=device, non_blocking=True)

        query_embeds, query_mask = self._prepare_query_geo_embeddings(
            query_ids,
            query_mask,
            geo_tensor,
            geo_attention_mask,
        )
        eos_token_id = resolve_generation_stop_token_ids(self.tokenizer, self.llm)
        if not eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = getattr(self.llm.generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        return self.llm.generate(
            inputs_embeds=query_embeds,
            attention_mask=query_mask.long(),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
