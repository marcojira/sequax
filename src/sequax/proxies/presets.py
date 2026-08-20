from flax import nnx

from sequax.proxies.training import TrainingConfig
from sequax.proxies.transformer import TransformerConfig, TransformerProxy

AMP_MODEL_CONFIG = TransformerConfig(
    num_tokens=23,
    embed_dim=64,
    hidden_dim=64,
    num_layers=4,
    num_heads=8,
)
GFP_MODEL_CONFIG = TransformerConfig(
    num_tokens=23,
    embed_dim=128,
    hidden_dim=128,
    num_layers=3,
    num_heads=8,
)
UTR_MODEL_CONFIG = TransformerConfig(
    num_tokens=7,
    embed_dim=64,
    hidden_dim=64,
    num_layers=4,
    num_heads=8,
)

AMP_TRAINING_CONFIG = TrainingConfig(
    objective="classification",
    batch_size=256,
    learning_rate=1e-4,
    patience=15,
)
GFP_TRAINING_CONFIG = TrainingConfig(
    objective="regression",
    batch_size=128,
    learning_rate=1e-5,
    patience=5,
)
UTR_TRAINING_CONFIG = TrainingConfig(
    objective="regression",
    batch_size=128,
    learning_rate=1e-4,
    patience=15,
)


def create_model(config: TransformerConfig, seed: int = 0) -> TransformerProxy:
    """Initialize a transformer proxy from a task preset."""
    return TransformerProxy(config, rngs=nnx.Rngs(seed))
