from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


@dataclass(frozen=True)
class TransformerConfig:
    num_tokens: int
    embed_dim: int
    hidden_dim: int
    output_dim: int = 1
    num_layers: int = 4
    num_heads: int = 8
    pad_token_id: int = 1
    dropout_rate: float = 0.0

    def __post_init__(self):
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EncoderBlock(nnx.Module):
    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embed_dim,
            qkv_features=config.embed_dim,
            out_features=config.embed_dim,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )
        self.attention_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.input_linear = nnx.Linear(config.embed_dim, config.hidden_dim, rngs=rngs)
        self.output_linear = nnx.Linear(config.hidden_dim, config.embed_dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        inputs: Array,
        mask: Array,
        *,
        deterministic: bool,
        rngs: nnx.Rngs | None,
    ) -> Array:
        attended = self.attention(
            inputs,
            mask=mask,
            deterministic=deterministic,
            rngs=rngs,
        )
        attended = self.dropout(attended, deterministic=deterministic, rngs=rngs)
        inputs = self.attention_norm(inputs + attended)

        hidden = jax.nn.relu(self.input_linear(inputs))
        hidden = self.dropout(hidden, deterministic=deterministic, rngs=rngs)
        return self.output_norm(inputs + self.output_linear(hidden))


class TransformerProxy(nnx.Module):
    """Transformer encoder that predicts one or more sequence properties."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(config.num_tokens, config.embed_dim, rngs=rngs)
        self.blocks = nnx.List([EncoderBlock(config, rngs=rngs) for _ in range(config.num_layers)])
        self.hidden_layers = nnx.List(
            [
                nnx.Linear(config.embed_dim, 4 * config.hidden_dim, rngs=rngs),
                nnx.Linear(4 * config.hidden_dim, 4 * config.hidden_dim, rngs=rngs),
            ]
        )
        self.output = nnx.Linear(4 * config.hidden_dim, config.output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        tokens: Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
        return_embedding: bool = False,
    ) -> Array | tuple[Array, Array]:
        single_sequence = tokens.ndim == 1
        if single_sequence:
            tokens = tokens[None, :]

        embeddings = self.embedding(tokens) + _sinusoidal_encoding(
            tokens.shape[-1], self.config.embed_dim
        )
        embeddings = self.dropout(embeddings, deterministic=deterministic, rngs=rngs)
        mask = (tokens != self.config.pad_token_id)[:, None, None, :]

        for block in self.blocks:
            embeddings = block(
                embeddings,
                mask,
                deterministic=deterministic,
                rngs=rngs,
            )

        sequence_embedding = embeddings[:, 0]
        hidden = sequence_embedding
        for layer in self.hidden_layers:
            hidden = jax.nn.relu(layer(hidden))
        predictions = self.output(hidden)

        if single_sequence:
            predictions = predictions[0]
            sequence_embedding = sequence_embedding[0]
        if return_embedding:
            return predictions, sequence_embedding
        return predictions


class NeuralProxy(nnx.Module):
    """Convert standardized transformer logits into positive rewards."""

    def __init__(self, model: TransformerProxy, mean: float, std: float):
        if std <= 0:
            raise ValueError("std must be positive")
        self.model = model
        self.mean = mean
        self.std = std

    def __call__(self, tokens: Array) -> Array:
        logits = self.model(tokens)
        return jnp.exp((logits - self.mean) / self.std).squeeze(-1)

    def embed(self, tokens: Array) -> tuple[Array, Array]:
        logits, embeddings = self.model(tokens, return_embedding=True)
        rewards = jnp.exp((logits - self.mean) / self.std).squeeze(-1)
        return rewards, embeddings


def _sinusoidal_encoding(length: int, embed_dim: int) -> Array:
    positions = jnp.arange(length, dtype=jnp.float32)[:, None]
    frequencies = jnp.exp(
        jnp.arange(0, embed_dim, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / embed_dim)
    )
    encoding = jnp.zeros((length, embed_dim), dtype=jnp.float32)
    encoding = encoding.at[:, 0::2].set(jnp.sin(positions * frequencies))
    encoding = encoding.at[:, 1::2].set(jnp.cos(positions * frequencies))
    return encoding[None, :, :]
