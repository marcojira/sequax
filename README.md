# Sequax

Sequax provides JAX environments for autoregressive sequence generation. An episode ends when the
agent emits EOS. The environment returns zero reward on every earlier step and evaluates the
completed sequence once, on the terminal step.

The package is unbatched. Callers can use `jax.vmap`, and RL libraries such as Modrax can own
batching, auto-reset, and episode metrics.

## Environment API

```python
import jax
import jax.numpy as jnp

from sequax import SequenceEnv

alphabet = ("BOS", "PAD", "EOS", "A", "B")
env = SequenceEnv(
    alphabet,
    reward_fn=lambda tokens: jnp.sum(tokens == 3),
    min_length=2,
    max_length=8,
)

state = env.reset(jax.random.key(0))
mask = env.action_mask(state)
output, state = env.step(state, jnp.int32(3), jax.random.key(1))
```

`min_length` and `max_length` count generated content tokens. They exclude BOS and EOS. Illegal
actions are replaced with the first legal action, matching the behavior of the original environments.

The included task definitions are `AMPSequence`, `GFPSequence`, `UTRSequence`, and `BitSequence`.
Biological tasks accept any callable proxy. `BitSequence` includes its synthetic reward.

## Neural proxies

The neural proxy is a Flax NNX transformer. Training supports binary classification and regression,
AdamW, validation splitting, early stopping, and validation-logit statistics.

```python
import jax.numpy as jnp

from sequax import AMPSequence
from sequax.proxies import (
    AMP_MODEL_CONFIG,
    AMP_TRAINING_CONFIG,
    create_model,
    load_proxy,
    save_proxy,
)
from sequax.proxies.training import train_proxy
from sequax.tasks import AMPSequence, load_amp_data

x, y = load_amp_data("neg_amp.pkl", "pos_amp.pkl")
model = create_model(AMP_MODEL_CONFIG)
result = train_proxy(model, jnp.asarray(x), jnp.asarray(y), AMP_TRAINING_CONFIG)
save_proxy("checkpoints/amp", result.model, result.validation_stats)

env = AMPSequence(load_proxy("checkpoints/amp"))
```

Equivalent model, training, and data-loading functions are provided by the GFP and UTR task modules.

To train all three proxies from the original dataset layout:

```text
data/
├── amp/neg_amp.pkl
├── amp/pos_amp.pkl
├── gfp/gfp_x.npy
├── gfp/gfp_y.npy
├── utr/utr_x.npy
└── utr/utr_y.npy
```

```bash
uv run python scripts/train_proxies.py all \
    --data-dir data \
    --output-dir checkpoints
```

Pass any subset of `amp gfp utr`. `--seed`, `--max-epochs`, and `--batch-size` are available for
overrides and quick runs.

Checkpoints use the same Orbax `StandardCheckpointer` and NNX split/merge pattern as Modrax. The
model configuration and reward-normalization statistics live in a small JSON sidecar.

## Modrax adapter

Sequax does not import Modrax. A Modrax wrapper can translate the small state boundary directly:

```python
from modrax.env import DiscreteActionSpec, Env, State, StepOutput


class SequaxEnv(Env):
    def __init__(self, config, sequence_env):
        self.sequence_env = sequence_env
        self.obs_shape = sequence_env.obs_shape
        self.action_spec = DiscreteActionSpec(sequence_env.num_actions)
        super().__init__(config)

    def _inner_reset_fn(self, key):
        state = self.sequence_env.reset(key)
        return State(state, state.tokens, self.sequence_env.action_mask(state))

    def _inner_step_fn(self, state, action, key):
        output, env_state = self.sequence_env.step(state.env_state, action, key)
        output = StepOutput(output.reward, output.done, output.truncation, output.info)
        next_state = State(
            env_state,
            env_state.tokens,
            self.sequence_env.action_mask(env_state),
        )
        return output, next_state
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

This package ports the environment, proxy, and proxy-training ideas from
[`marcojira/tgm`](https://github.com/marcojira/tgm/tree/main/src/medium_rl/envs/proxies).
