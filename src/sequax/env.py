from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class SequenceState(NamedTuple):
    tokens: Array
    length: Array
    done: Array


class StepOutput(NamedTuple):
    reward: Array
    done: Array
    truncation: Array
    info: Any = None


RewardFn = Callable[[Array], Array]


class SequenceEnv:
    """Build a token sequence and score it when EOS is emitted."""

    def __init__(
        self,
        alphabet: Sequence[str],
        reward_fn: RewardFn,
        min_length: int,
        max_length: int,
        *,
        bos_token: str = "BOS",
        pad_token: str = "PAD",
        eos_token: str = "EOS",
    ):
        if not 0 <= min_length <= max_length:
            raise ValueError("Expected 0 <= min_length <= max_length")
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet tokens must be unique")

        token_to_id = {token: index for index, token in enumerate(alphabet)}
        missing = {bos_token, pad_token, eos_token} - token_to_id.keys()
        if missing:
            raise ValueError(f"Alphabet is missing special tokens: {sorted(missing)}")

        self.alphabet = tuple(alphabet)
        self.token_to_id = token_to_id
        self.reward_fn = reward_fn
        self.min_length = min_length
        self.max_length = max_length
        self.bos_token_id = token_to_id[bos_token]
        self.pad_token_id = token_to_id[pad_token]
        self.eos_token_id = token_to_id[eos_token]
        self.num_actions = len(alphabet)
        self.obs_shape = (max_length + 2,)

        content_mask = jnp.ones(self.num_actions, dtype=jnp.bool_)
        content_mask = content_mask.at[self.bos_token_id].set(False)
        content_mask = content_mask.at[self.pad_token_id].set(False)
        self._content_mask = content_mask.at[self.eos_token_id].set(False)
        self._eos_mask = (
            jnp.zeros(self.num_actions, dtype=jnp.bool_).at[self.eos_token_id].set(True)
        )

    def reset(self, key: Array | None = None) -> SequenceState:
        """Return an empty sequence state; the key is accepted for wrapper symmetry."""
        del key
        tokens = jnp.full(self.obs_shape, self.pad_token_id, dtype=jnp.int32)
        tokens = tokens.at[0].set(self.bos_token_id)
        return SequenceState(tokens, jnp.int32(0), jnp.bool_(False))

    def action_mask(self, state: SequenceState) -> Array:
        """Return legal next tokens for a single state."""
        can_stop = state.length >= self.min_length
        at_limit = state.length >= self.max_length
        mask = self._content_mask.at[self.eos_token_id].set(can_stop)
        mask = jax.lax.select(at_limit, self._eos_mask, mask)
        return jax.lax.select(state.done, jnp.zeros_like(mask), mask)

    def step(
        self, state: SequenceState, action: Array, key: Array | None = None
    ) -> tuple[StepOutput, SequenceState]:
        """Append one legal token and return a terminal reward on EOS."""
        del key
        mask = self.action_mask(state)
        action = jnp.asarray(action, dtype=jnp.int32)
        action = jax.lax.select(mask[action], action, jnp.argmax(mask).astype(jnp.int32))

        ends_episode = jnp.logical_and(~state.done, action == self.eos_token_id)
        writes_token = ~state.done
        token_index = jnp.minimum(state.length + 1, self.max_length + 1)
        tokens = jax.lax.cond(
            writes_token,
            lambda value: value.at[token_index].set(action),
            lambda value: value,
            state.tokens,
        )
        length = state.length + jnp.asarray(writes_token & ~ends_episode, dtype=state.length.dtype)
        done = state.done | ends_episode
        next_state = SequenceState(tokens, length, done)
        reward = jax.lax.cond(
            ends_episode,
            lambda value: jnp.asarray(self.reward_fn(value), dtype=jnp.float32).squeeze(),
            lambda value: jnp.zeros((), dtype=jnp.float32),
            tokens,
        )
        output = StepOutput(reward, done, jnp.bool_(False))
        return output, next_state

    def decode(self, tokens: Array) -> str:
        """Decode one token array, excluding special tokens."""
        special_ids = {self.bos_token_id, self.pad_token_id, self.eos_token_id}
        return "".join(
            self.alphabet[int(token)] for token in tokens if int(token) not in special_ids
        )
