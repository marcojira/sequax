import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

from sequax.proxies import (
    AMP_MODEL_CONFIG,
    AMP_TRAINING_CONFIG,
    GFP_MODEL_CONFIG,
    GFP_TRAINING_CONFIG,
    UTR_MODEL_CONFIG,
    UTR_TRAINING_CONFIG,
    create_model,
    save_proxy,
)
from sequax.proxies.training import TrainingConfig, train_proxy
from sequax.proxies.transformer import TransformerConfig
from sequax.tasks import (
    AMPSequence,
    GFPSequence,
    UTRSequence,
    load_amp_data,
    load_gfp_data,
    load_utr_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sequax biological reward proxies")
    parser.add_argument("tasks", nargs="+", choices=("all", "amp", "gfp", "utr"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ("amp", "gfp", "utr") if "all" in args.tasks else tuple(dict.fromkeys(args.tasks))
    for task in tasks:
        train_task(
            task,
            args.data_dir,
            args.output_dir,
            seed=args.seed,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )


def train_task(
    task: str,
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    max_epochs: int | None,
    batch_size: int | None,
) -> None:
    model_config, training_config, x, y = load_task(task, data_dir)
    overrides = {"seed": seed, "verbose": True}
    if max_epochs is not None:
        overrides["max_epochs"] = max_epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    training_config = replace(training_config, **overrides)

    print(f"Training {task.upper()} on {x.shape[0]} sequences")
    model = create_model(model_config, seed)
    result = train_proxy(model, jnp.asarray(x), jnp.asarray(y), training_config)
    checkpoint_path = output_dir / task
    save_proxy(checkpoint_path, result.model, result.validation_stats)
    print(f"Saved {task.upper()} proxy to {checkpoint_path}")


def load_task(
    task: str, data_dir: Path
) -> tuple[TransformerConfig, TrainingConfig, jnp.ndarray, jnp.ndarray]:
    task_dir = data_dir / task
    if task == "amp":
        x, y = load_amp_data(
            task_dir / "neg_amp.pkl",
            task_dir / "pos_amp.pkl",
        )
        model_config = replace(AMP_MODEL_CONFIG, num_tokens=len(AMPSequence.alphabet))
        return model_config, AMP_TRAINING_CONFIG, x, y
    if task == "gfp":
        x, y = load_gfp_data(task_dir / "gfp_x.npy", task_dir / "gfp_y.npy")
        model_config = replace(GFP_MODEL_CONFIG, num_tokens=len(GFPSequence.alphabet))
        return model_config, GFP_TRAINING_CONFIG, x, y
    if task == "utr":
        x, y = load_utr_data(task_dir / "utr_x.npy", task_dir / "utr_y.npy")
        model_config = replace(UTR_MODEL_CONFIG, num_tokens=len(UTRSequence.alphabet))
        return model_config, UTR_TRAINING_CONFIG, x, y
    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
