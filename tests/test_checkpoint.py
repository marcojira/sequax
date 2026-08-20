import jax.numpy as jnp
import pytest
from flax import nnx

from sequax.proxies import TransformerConfig, TransformerProxy, load_proxy, save_proxy
from sequax.proxies.training import ValidationStats


def test_nnx_proxy_checkpoint_round_trip(tmp_path):
    model = TransformerProxy(
        TransformerConfig(
            num_tokens=6,
            embed_dim=8,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
        ),
        rngs=nnx.Rngs(0),
    )
    tokens = jnp.array([[0, 3, 2, 1]])
    stats = ValidationStats(jnp.float32(-1), jnp.float32(1), jnp.float32(0), jnp.float32(2))

    save_proxy(tmp_path, model, stats)
    restored = load_proxy(tmp_path)

    assert jnp.allclose(restored(tokens), jnp.exp(model(tokens).squeeze(-1) / 2))
    with pytest.raises(FileExistsError):
        save_proxy(tmp_path, model, stats)
