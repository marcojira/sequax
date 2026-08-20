from pathlib import Path

import numpy as np

from sequax.env import SequenceEnv
from sequax.proxies.base import Proxy

GFP_ALPHABET = (
    "BOS",
    "PAD",
    "EOS",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)


class GFPSequence(SequenceEnv):
    name = "GFP"
    alphabet = GFP_ALPHABET

    def __init__(self, proxy: Proxy, sequence_length: int = 237):
        super().__init__(self.alphabet, proxy, sequence_length, sequence_length)


def load_gfp_data(x_path: str | Path, y_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load GFP sequences and add the environment's special tokens."""
    x = np.load(x_path).astype(np.int32) + GFPSequence.alphabet.index("A")
    y = np.load(y_path).astype(np.float32)
    bos = np.full((x.shape[0], 1), GFPSequence.alphabet.index("BOS"), dtype=np.int32)
    eos = np.full((x.shape[0], 1), GFPSequence.alphabet.index("EOS"), dtype=np.int32)
    return np.concatenate([bos, x, eos], axis=1), y
