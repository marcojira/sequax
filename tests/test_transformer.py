import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from sequax.proxies import TransformerConfig, TransformerProxy
from sequax.proxies.training import (
    TrainingConfig,
    evaluate_proxy,
    split_dataset,
    train_proxy,
)


def small_model(seed: int = 0) -> TransformerProxy:
    config = TransformerConfig(
        num_tokens=6,
        embed_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )
    return TransformerProxy(config, rngs=nnx.Rngs(seed))


def test_transformer_forward_and_embedding_shapes():
    model = small_model()
    tokens = jnp.array([[0, 3, 2, 1], [0, 4, 5, 2]])

    predictions, embeddings = model(tokens, return_embedding=True)

    assert predictions.shape == (2, 1)
    assert embeddings.shape == (2, 8)


def test_training_updates_parameters_and_returns_validation_stats():
    model = small_model()
    before = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))
    x = jnp.array([[0, 3, 2, 1], [0, 4, 2, 1], [0, 5, 2, 1], [0, 3, 4, 2]])
    y = jnp.array([0.0, 1.0, 1.0, 0.0])

    result = train_proxy(
        model,
        x,
        y,
        TrainingConfig(
            objective="classification",
            batch_size=2,
            learning_rate=1e-3,
            max_epochs=2,
            validation_fraction=0.25,
            patience=2,
        ),
    )
    after = nnx.state(model, nnx.Param)
    changed = jax.tree.leaves(
        jax.tree.map(lambda left, right: jnp.any(left != right), before, after)
    )

    assert any(changed)
    assert len(result.train_loss) == 2


def test_dataset_split_is_reproducible_and_paired():
    x = jnp.arange(20).reshape(10, 2)
    y = jnp.arange(10)
    split = split_dataset(x, y, jax.random.key(3), validation_fraction=0.2)
    repeated = split_dataset(x, y, jax.random.key(3), validation_fraction=0.2)

    assert all(jnp.array_equal(left, right) for left, right in zip(split, repeated, strict=True))
    x_train, x_validation, y_train, y_validation = split
    assert jnp.array_equal(x_train[:, 0] // 2, y_train)
    assert jnp.array_equal(x_validation[:, 0] // 2, y_validation)


def test_training_restores_the_best_validation_parameters():
    config = TrainingConfig(
        objective="regression",
        batch_size=4,
        learning_rate=1.0,
        max_epochs=8,
        validation_fraction=0.25,
        patience=8,
        seed=0,
    )
    x = jnp.array([[0, 3, 4, 2], [0, 4, 5, 2], [0, 5, 3, 2], [0, 3, 5, 2]] * 4)
    y = jnp.array([0.0, 1.0, 2.0, 3.0] * 4)

    result = train_proxy(small_model(), x, y, config)

    key, split_key = jax.random.split(jax.random.key(config.seed))
    _, x_validation, _, y_validation = split_dataset(x, y, split_key, config.validation_fraction)
    final_loss, _ = evaluate_proxy(
        result.model, x_validation, y_validation, config.objective, config.batch_size
    )

    assert min(result.validation_loss) < max(result.validation_loss)
    assert float(final_loss) == pytest.approx(min(result.validation_loss), rel=1e-5)
