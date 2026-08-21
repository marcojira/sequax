import jax
import jax.numpy as jnp
import pytest

from sequax import BitSequence
from sequax.tasks import amp, gfp, utr
from sequax.tasks.bit_sequence import _levenshtein


def _zero_proxy(tokens):
    """Return a constant reward for constructor tests."""
    return jnp.float32(0)


@pytest.mark.parametrize(
    ("task_module", "task_class", "checkpoint_name"),
    [
        (amp, amp.AMPSequence, "amp"),
        (gfp, gfp.GFPSequence, "gfp"),
        (utr, utr.UTRSequence, "utr"),
    ],
)
def test_biological_task_loads_bundled_proxy_by_default(
    monkeypatch, task_module, task_class, checkpoint_name
):
    loaded_paths = []
    monkeypatch.setattr(
        task_module, "load_proxy", lambda path: loaded_paths.append(path) or _zero_proxy
    )

    env = task_class()
    task_class()

    assert env.reward_fn is _zero_proxy
    assert len(loaded_paths) == 2
    assert all(
        path.parts[-3:] == ("sequax", "checkpoints", checkpoint_name) for path in loaded_paths
    )


@pytest.mark.parametrize("task_class", [amp.AMPSequence, gfp.GFPSequence, utr.UTRSequence])
def test_biological_task_accepts_proxy_override(task_class):
    env = task_class(proxy=_zero_proxy)

    assert env.reward_fn is _zero_proxy


def test_bit_sequence_is_jittable_and_rewards_only_at_terminal():
    env = BitSequence(bit_length=16, substring_length=4, num_modes=3)
    state = env.reset()
    step = jax.jit(env.step)

    for _ in range(4):
        output, state = step(state, jnp.int32(3))
        assert env.terminal_reward(state) == 0

    output, state = step(state, jnp.int32(env.eos_token_id))

    assert output.done
    assert jax.jit(env.terminal_reward)(state) > 0
    assert env.proxy.embed(state.tokens).shape == (16,)


def test_bit_sequence_uses_levenshtein_distance():
    left = jnp.array([0, 1, 0])
    right = jnp.array([1, 0, 1])

    assert _levenshtein(left, right) == 2
