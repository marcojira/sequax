import jax
import jax.numpy as jnp

from sequax import BitSequence
from sequax.tasks.bit_sequence import _levenshtein


def test_bit_sequence_is_jittable_and_rewards_only_at_terminal():
    env = BitSequence(bit_length=16, substring_length=4, num_modes=3)
    state = env.reset()
    step = jax.jit(env.step)

    for _ in range(4):
        output, state = step(state, jnp.int32(3))
        assert output.reward == 0

    output, state = step(state, jnp.int32(env.eos_token_id))

    assert output.done
    assert output.reward > 0
    assert env.proxy.embed(state.tokens).shape == (16,)


def test_bit_sequence_uses_levenshtein_distance():
    left = jnp.array([0, 1, 0])
    right = jnp.array([1, 0, 1])

    assert _levenshtein(left, right) == 2
