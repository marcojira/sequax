# Sequax

Sequax provides JAX environments for autoregressive sequence generation. An episode ends when the
agent emits EOS. The environment returns zero reward on every earlier step and evaluates the
completed sequence once, on the terminal step.

The package is unbatched. Callers can use `jax.vmap`, and RL libraries such as Modrax can own
batching, auto-reset, and episode metrics.

The code was ported over from https://github.com/marcojira/tgm/tree/main to support Flax NNX and to isolate the environment components.

## Supported tasks

- `AMPSequence`: variable-length antimicrobial peptide generation.
- `GFPSequence`: fixed-length green fluorescent protein generation.
- `UTRSequence`: fixed-length DNA 5′ UTR generation.
- `BitSequence`: synthetic bit-string generation with a built-in multimodal reward.

Biological tasks load their bundled trained proxy by default. Pass a different proxy to override it,
or create the self-contained synthetic task. See [Training proxies](#training-proxies) to train a
proxy.

```python
from sequax import AMPSequence, BitSequence

amp_env = AMPSequence()
bit_env = BitSequence()
```

To use a different proxy, pass it to the task: `AMPSequence(proxy=my_proxy)`.

## Writing a custom task

Define the task's alphabet and length limits in a `SequenceEnv` subclass. The alphabet must contain
`BOS`, `PAD`, and `EOS`. Pass a JAX-compatible reward function that accepts the completed, padded
token array and returns a scalar.

```python
import jax
import jax.numpy as jnp

from sequax import SequenceEnv


class MySequence(SequenceEnv):
    name = "MySequence"
    alphabet = ("BOS", "PAD", "EOS", "A", "B")

    def __init__(self, reward_fn):
        super().__init__(self.alphabet, reward_fn, min_length=2, max_length=8)


env = MySequence(lambda tokens: jnp.sum(tokens == 3))
state = env.reset(jax.random.key(0))
mask = env.action_mask(state)
output, state = env.step(state, jnp.int32(3), jax.random.key(1))
```

The environment uses the length limits to build its action masks and calls the reward only when EOS
ends the sequence. Lengths count content tokens, excluding BOS and EOS.

## Training proxies

Install the dependencies with `uv sync`, then arrange the training data as follows:

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
uv run python train_proxies.py all \
    --data-dir data \
    --output-dir checkpoints
```

This trains all three proxies from scratch and saves them under `checkpoints/amp`, `checkpoints/gfp`,
and `checkpoints/utr`. Replace `all` with any subset of `amp`, `gfp`, and `utr` to train selected
tasks. Use `--max-epochs` or `--batch-size` to override the defaults.

## Acknowledgments
The biological environments are jax implementations with moderate modifications of the environments of [Biological Sequence Design with GFlowNets
](https://github.com/MJ10/BioSeq-GFN-AL) as well as the benchmarks of [Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization](https://github.com/brandontrabucco/design-bench). The training process for the proxy reward functions comes from the former and the data used from the latter. The BitSequence environment comes from [Trajectory balance: Improved credit assignment in GFlowNets
](https://arxiv.org/abs/2201.13259). The design of the `SequenceEnv` environment is inspired by the [PGX library](https://github.com/sotetsuk/pgx).
