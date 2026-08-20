import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from sequax.proxies.transformer import TransformerProxy

Objective = Literal["classification", "regression"]


@dataclass(frozen=True)
class TrainingConfig:
    objective: Objective = "regression"
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_epochs: int = 250
    validation_fraction: float = 0.2
    weight_decay: float = 1e-6
    patience: int = 15
    seed: int = 0
    verbose: bool = False


class ValidationStats(NamedTuple):
    minimum: Array
    maximum: Array
    mean: Array
    std: Array


class TrainingResult(NamedTuple):
    model: TransformerProxy
    validation_stats: ValidationStats
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def split_dataset(
    x: Array, y: Array, key: Array, validation_fraction: float = 0.2
) -> tuple[Array, Array, Array, Array]:
    """Split paired arrays with a reproducible random permutation."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if x.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    validation_size = min(max(round(x.shape[0] * validation_fraction), 1), x.shape[0] - 1)
    indices = jax.random.permutation(key, x.shape[0])
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return x[train_indices], x[validation_indices], y[train_indices], y[validation_indices]


@nnx.jit(static_argnames="objective")
def _train_step(
    model: TransformerProxy,
    optimizer: nnx.Optimizer,
    x: Array,
    y: Array,
    rngs: nnx.Rngs,
    objective: Objective,
) -> tuple[Array, Array]:
    def loss_fn(model: TransformerProxy) -> tuple[Array, Array]:
        predictions = model(x, deterministic=False, rngs=rngs).squeeze(-1)
        return _loss_and_metric(predictions, y, objective)

    (loss, metric), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, gradients)
    return loss, metric


def train_proxy(
    model: TransformerProxy, x: Array, y: Array, config: TrainingConfig
) -> TrainingResult:
    """Train a transformer proxy with validation-based early stopping."""
    key = jax.random.key(config.seed)
    key, split_key = jax.random.split(key)
    x_train, x_validation, y_train, y_validation = split_dataset(
        x, y, split_key, config.validation_fraction
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        wrt=nnx.Param,
    )
    dropout_rngs = nnx.Rngs(dropout=config.seed)
    best_parameters = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(config.max_epochs):
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, x_train.shape[0])
        batch_losses = []

        for start in range(0, x_train.shape[0], config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            loss, _ = _train_step(
                model,
                optimizer,
                x_train[batch_indices],
                y_train[batch_indices],
                dropout_rngs,
                config.objective,
            )
            batch_losses.append(loss)

        train_loss = float(jnp.mean(jnp.stack(batch_losses)))
        validation_loss, _ = evaluate_proxy(
            model,
            x_validation,
            y_validation,
            config.objective,
            config.batch_size,
        )
        train_losses.append(train_loss)
        validation_losses.append(float(validation_loss))
        if config.verbose:
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"validation_loss={float(validation_loss):.6f}"
            )

        if validation_loss < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_parameters = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > config.patience:
                break

    nnx.update(model, best_parameters)
    stats = validation_stats(model, x_validation, config.batch_size)
    return TrainingResult(model, stats, tuple(train_losses), tuple(validation_losses))


def evaluate_proxy(
    model: TransformerProxy,
    x: Array,
    y: Array,
    objective: Objective,
    batch_size: int = 256,
) -> tuple[Array, Array]:
    """Return sample-weighted loss and auxiliary metric."""
    total_loss = jnp.zeros(())
    total_metric = jnp.zeros(())
    for start in range(0, x.shape[0], batch_size):
        batch_x = x[start : start + batch_size]
        batch_y = y[start : start + batch_size]
        predictions = model(batch_x).squeeze(-1)
        loss, metric = _loss_and_metric(predictions, batch_y, objective)
        total_loss += loss * batch_x.shape[0]
        total_metric += metric * batch_x.shape[0]
    return total_loss / x.shape[0], total_metric / x.shape[0]


def validation_stats(model: TransformerProxy, x: Array, batch_size: int = 256) -> ValidationStats:
    """Summarize raw validation logits used to normalize rewards."""
    predictions = []
    for start in range(0, x.shape[0], batch_size):
        predictions.append(model(x[start : start + batch_size]).squeeze(-1))
    values = jnp.concatenate(predictions)
    return ValidationStats(values.min(), values.max(), values.mean(), values.std())


def _loss_and_metric(
    predictions: Array, targets: Array, objective: Objective
) -> tuple[Array, Array]:
    if objective == "classification":
        loss = optax.sigmoid_binary_cross_entropy(predictions, targets).mean()
        accuracy = ((predictions > 0) == targets.astype(jnp.bool_)).mean()
        return loss, accuracy
    if objective == "regression":
        errors = predictions - targets
        return jnp.mean(errors**2), jnp.mean(jnp.abs(errors))
    raise ValueError(f"Unknown objective: {objective}")
