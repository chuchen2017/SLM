"""SLM: inject one embedding per geospatial entity at each `<GEO>` token."""

from __future__ import annotations

import torch
from torch import nn

from slm.chat import resolve_generation_stop_token_ids


class Geo2VecEncoder(nn.Module):
    """Map a frozen Geo2Vec vector into the width the projector expects."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GeoProjector(nn.Module):
    """Project an entity vector into the language model's hidden size."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def llm_hidden_size(llm: nn.Module) -> int:
    base = llm.base_model.model if hasattr(llm, "base_model") and hasattr(llm.base_model, "model") else llm
    config = base.config
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return int(text_config.hidden_size)
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    raise AttributeError("Could not find hidden_size on the language-model config.")


def inject_geo_token_embeddings(
    query_ids: torch.Tensor,
    query_embeds: torch.Tensor,
    geo_embeds: torch.Tensor,
    geo_attention_mask: torch.Tensor,
    geo_token_id: int,
) -> torch.Tensor:
    """Write each entity embedding over the matching `<GEO>` token.

    Entity ``i`` of a sample replaces the ``i``-th `<GEO>` token from the left.
    Padding positions are excluded by ``geo_attention_mask``. The notebooks counted
    valid entities with ``embedding.sum(-1) != 0``, which drops a real vector that
    sums to zero and treats a non-zero pad as a real entity.
    """
    if query_ids.shape[:2] != query_embeds.shape[:2]:
        raise ValueError("query ids and query embeddings must have the same batch and length.")
    updated = query_embeds.clone()
    for batch_idx in range(query_ids.size(0)):
        positions = torch.nonzero(query_ids[batch_idx] == geo_token_id, as_tuple=False).flatten()
        valid = geo_embeds[batch_idx][geo_attention_mask[batch_idx].bool()]
        if positions.numel() != valid.size(0):
            raise ValueError(
                f"Sample {batch_idx} has {positions.numel()} <GEO> tokens but {valid.size(0)} entity embeddings."
            )
        if valid.size(0):
            updated[batch_idx, positions] = valid.to(dtype=updated.dtype)
    return updated


class SLM(nn.Module):
    """Causal LM whose `<GEO>` tokens are replaced by projected entity embeddings."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        geo_encoder: nn.Module,
        geo_token_id: int,
        geo_projector_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.geo_encoder = geo_encoder
        self.geo_token_id = int(geo_token_id)
        self.embed_tokens = self.llm.get_input_embeddings()
        self.llm_hidden = llm_hidden_size(self.llm)
        out_dim = int(getattr(geo_encoder, "out_dim"))
        self.geo_out_dim = out_dim
        self.geo_projector = GeoProjector(out_dim, geo_projector_hidden_dim, self.llm_hidden)

    def encode_entities(
        self,
        geo_tensor: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _ = geo_tensor.shape
        flat = geo_tensor.reshape(batch * count, -1)
        valid = geo_attention_mask.reshape(-1).bool()
        weight = self.geo_projector.net[0].weight
        if not valid.any():
            return flat.new_zeros(batch, count, self.llm_hidden, dtype=weight.dtype)
        filled = self.geo_projector(self.geo_encoder(flat[valid].to(dtype=weight.dtype)))
        projected = filled.new_zeros(flat.shape[0], self.llm_hidden)
        # masked_scatter walks the mask in row-major order, which matches `filled`.
        mask = valid.unsqueeze(1).expand(-1, self.llm_hidden)
        projected = projected.masked_scatter(mask, filled.reshape(-1))
        return projected.view(batch, count, self.llm_hidden)

    def prepare_inputs(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        geo_tensor: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_embeds = self.embed_tokens(query_ids)
        if geo_tensor.size(1) == 0:
            if (query_ids == self.geo_token_id).any():
                raise ValueError("The prompt contains <GEO> but this batch has no entity embeddings.")
            return query_embeds
        geo_embeds = self.encode_entities(geo_tensor, geo_attention_mask)
        return inject_geo_token_embeddings(
            query_ids,
            query_embeds,
            geo_embeds,
            geo_attention_mask,
            self.geo_token_id,
        )

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

        query_embeds = self.prepare_inputs(query_ids, query_mask, geo_tensor, geo_attention_mask)
        label_embeds = self.embed_tokens(labels)
        inputs_embeds = torch.cat((query_embeds, label_embeds), dim=1)
        attention_mask = torch.cat((query_mask.long(), label_mask.long()), dim=1)

        labels_for_loss = labels.masked_fill(~label_mask.bool(), -100)
        pad_id = self.tokenizer.pad_token_id
        stop_ids = set(resolve_generation_stop_token_ids(self.tokenizer, self.llm))
        # When pad and EOS are the same id, the attention mask already drops padding.
        # Masking that id would also drop the end-of-turn token the label is teaching.
        if pad_id is not None and pad_id not in stop_ids:
            labels_for_loss = labels_for_loss.masked_fill(labels_for_loss == pad_id, -100)
        prefix = labels_for_loss.new_full((labels.size(0), query_embeds.size(1)), -100)
        labels_full = torch.cat((prefix, labels_for_loss), dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_full)

    @torch.no_grad()
    def generate(
        self,
        query_ids: torch.Tensor,
        geo_tensor: torch.Tensor,
        query_mask: torch.Tensor,
        geo_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
    ) -> torch.Tensor:
        device = query_ids.device
        geo_tensor = geo_tensor.to(device=device, non_blocking=True)
        query_mask = query_mask.to(device=device, non_blocking=True)
        if geo_attention_mask is None:
            geo_attention_mask = torch.ones(geo_tensor.shape[:2], dtype=torch.bool, device=device)
        else:
            geo_attention_mask = geo_attention_mask.to(device=device, non_blocking=True)

        query_embeds = self.prepare_inputs(query_ids, query_mask, geo_tensor, geo_attention_mask)
        stop_ids = resolve_generation_stop_token_ids(self.tokenizer, self.llm) or self.tokenizer.eos_token_id
        pad_token_id = getattr(getattr(self.llm, "generation_config", None), "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        generate_kwargs = {
            "inputs_embeds": query_embeds,
            "attention_mask": query_mask.long(),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": stop_ids,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_k"] = top_k
            generate_kwargs["top_p"] = top_p
        previous_cache = getattr(self.llm.config, "use_cache", True)
        self.llm.config.use_cache = True
        try:
            return self.llm.generate(**generate_kwargs)
        finally:
            self.llm.config.use_cache = previous_cache
