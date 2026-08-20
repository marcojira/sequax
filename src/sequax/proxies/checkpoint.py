import json
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from sequax.proxies.training import ValidationStats
from sequax.proxies.transformer import NeuralProxy, TransformerConfig, TransformerProxy


def save_proxy(
    directory: str | Path,
    model: TransformerProxy,
    validation_stats: ValidationStats,
) -> None:
    """Save an NNX proxy and its reward-normalization statistics."""
    directory = Path(directory)
    params_path = directory / "params"
    if params_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model.config.to_dict(),
        "validation": {
            "minimum": float(validation_stats.minimum),
            "maximum": float(validation_stats.maximum),
            "mean": float(validation_stats.mean),
            "std": float(validation_stats.std),
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(params_path.absolute(), state)
    checkpointer.wait_until_finished()


def load_proxy(directory: str | Path) -> NeuralProxy:
    """Load an NNX transformer as a normalized reward proxy."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    model = TransformerProxy(
        TransformerConfig(**metadata["model"]),
        rngs=nnx.Rngs(0),
    )
    graphdef, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(
        (directory / "params").absolute(),
        state,
    )
    model = nnx.merge(graphdef, restored)
    validation = metadata["validation"]
    return NeuralProxy(model, validation["mean"], validation["std"])
