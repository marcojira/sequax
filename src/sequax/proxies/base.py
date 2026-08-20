from collections.abc import Callable
from typing import Protocol

from jax import Array


class Proxy(Protocol):
    def __call__(self, tokens: Array) -> Array: ...


class CallableProxy:
    def __init__(self, reward_fn: Callable[[Array], Array]):
        self.reward_fn = reward_fn

    def __call__(self, tokens: Array) -> Array:
        return self.reward_fn(tokens)
