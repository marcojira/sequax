import jax
import jax.numpy as jnp

from sequax import SequenceEnv

ALPHABET = ("BOS", "PAD", "EOS", "A", "B")


def test_terminal_reward_is_zero_until_eos():
    env = SequenceEnv(ALPHABET, lambda tokens: tokens.sum(), min_length=1, max_length=3)
    state = env.reset()

    first_output, state = env.step(state, jnp.int32(3))
    assert first_output.reward == 0
    assert not first_output.done
    assert env.terminal_reward(state) == 0

    final_output, state = env.step(state, jnp.int32(2))
    assert final_output.reward == 0
    assert final_output.done
    assert env.terminal_reward(state) == state.tokens.sum()
    assert state.length == 1


def test_action_masks_enforce_length_bounds():
    env = SequenceEnv(ALPHABET, lambda tokens: jnp.float32(1), min_length=1, max_length=2)
    state = env.reset()

    assert not env.action_mask(state)[env.eos_token_id]
    _, state = env.step(state, jnp.int32(3))
    assert env.action_mask(state)[env.eos_token_id]
    _, state = env.step(state, jnp.int32(4))
    assert jnp.array_equal(
        env.action_mask(state),
        jnp.array([False, False, True, False, False]),
    )


def test_reset_and_step_can_be_jitted_and_vmapped():
    env = SequenceEnv(ALPHABET, lambda tokens: tokens.sum(), min_length=0, max_length=2)
    keys = jax.random.split(jax.random.key(0), 4)
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.full(4, env.eos_token_id)

    outputs, states = jax.jit(jax.vmap(env.step))(states, actions, keys)

    assert outputs.reward.shape == (4,)
    assert jnp.all(outputs.done)
    assert states.tokens.shape == (4, 4)
    assert jax.jit(jax.vmap(env.terminal_reward))(states).shape == (4,)
