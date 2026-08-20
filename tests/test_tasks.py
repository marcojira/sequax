import jax
import jax.numpy as jnp

from sequax import AMPSequence, BitSequence, GFPSequence, UTRSequence
from sequax.tasks.bit_sequence import _levenshtein


def test_biological_tasks_use_expected_content_lengths():
    def reward(tokens):
        return jnp.float32(tokens.sum())

    assert AMPSequence(reward).obs_shape == (62,)
    assert GFPSequence(reward).obs_shape == (239,)
    assert UTRSequence(reward).obs_shape == (52,)


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
