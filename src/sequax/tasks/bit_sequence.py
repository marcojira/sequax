import itertools
import random

import jax
import jax.numpy as jnp
from jax import Array

from sequax.env import SequenceEnv


class BitSequenceProxy:
    def __init__(self, bit_length: int, substring_length: int, num_modes: int, seed: int = 0):
        if bit_length % substring_length:
            raise ValueError("bit_length must be divisible by substring_length")
        if bit_length % 8:
            raise ValueError("bit_length must be divisible by 8")

        self.bit_length = bit_length
        self.substring_length = substring_length
        self.num_modes = num_modes
        self.embed_dim = bit_length
        rng = random.Random(seed)
        blocks = ("00000000", "11111111", "11110000", "00001111", "00111100")
        mode_strings = ["".join(rng.choices(blocks, k=bit_length // 8)) for _ in range(num_modes)]
        self.modes = jnp.asarray([[int(bit) for bit in mode] for mode in mode_strings])

    def __call__(self, tokens: Array) -> Array:
        distances = self.distances(tokens)
        return jnp.exp(1 - distances.min(axis=-1) / self.bit_length)

    def embed(self, tokens: Array) -> Array:
        token_values = tokens[..., 1:-1] - 3
        shifts = jnp.arange(self.substring_length - 1, -1, -1)
        bits = (token_values[..., :, None] >> shifts) & 1
        return bits.reshape(*tokens.shape[:-1], self.bit_length)

    def distances(self, tokens: Array) -> Array:
        bits = self.embed(tokens)
        flat_bits = bits.reshape(-1, self.bit_length)
        distances = jax.vmap(
            lambda sequence: jax.vmap(lambda mode: _levenshtein(sequence, mode))(self.modes)
        )(flat_bits)
        return distances.reshape(*bits.shape[:-1], self.num_modes)


class BitSequence(SequenceEnv):
    name = "BitSequence"

    def __init__(
        self,
        bit_length: int = 120,
        substring_length: int = 8,
        num_modes: int = 60,
        seed: int = 0,
    ):
        proxy = BitSequenceProxy(bit_length, substring_length, num_modes, seed)
        alphabet = ("BOS", "PAD", "EOS") + tuple(
            "".join(bits) for bits in itertools.product("01", repeat=substring_length)
        )
        sequence_length = bit_length // substring_length
        super().__init__(alphabet, proxy, sequence_length, sequence_length)
        self.proxy = proxy


def _levenshtein(left: Array, right: Array) -> Array:
    initial_row = jnp.arange(right.shape[0] + 1, dtype=jnp.int32)

    def scan_left(previous_row: Array, indexed_token: tuple[Array, Array]):
        index, left_token = indexed_token

        def scan_right(carry: tuple[Array, Array], values: tuple[Array, Array]):
            left_cost, diagonal_cost = carry
            right_token, upper_cost = values
            value = jnp.minimum(
                jnp.minimum(upper_cost + 1, left_cost + 1),
                diagonal_cost + (left_token != right_token),
            )
            return (value, upper_cost), value

        first_cost = index + 1
        _, remaining_row = jax.lax.scan(
            scan_right,
            (first_cost, previous_row[0]),
            (right, previous_row[1:]),
        )
        current_row = jnp.concatenate((first_cost[None], remaining_row))
        return current_row, None

    final_row, _ = jax.lax.scan(scan_left, initial_row, (jnp.arange(left.shape[0]), left))
    return final_row[-1]
