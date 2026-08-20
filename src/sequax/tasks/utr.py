from pathlib import Path

import numpy as np

from sequax.env import SequenceEnv
from sequax.proxies.base import Proxy

UTR_ALPHABET = ("BOS", "PAD", "EOS", "A", "C", "G", "T")


class UTRSequence(SequenceEnv):
    name = "UTR"
    alphabet = UTR_ALPHABET

    def __init__(self, proxy: Proxy, sequence_length: int = 50):
        super().__init__(self.alphabet, proxy, sequence_length, sequence_length)


def load_utr_data(x_path: str | Path, y_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UTR sequences and add the environment's special tokens."""
    x = np.load(x_path).astype(np.int32) + UTRSequence.alphabet.index("A")
    y = np.load(y_path).astype(np.float32).reshape(-1)
    bos = np.full((x.shape[0], 1), UTRSequence.alphabet.index("BOS"), dtype=np.int32)
    eos = np.full((x.shape[0], 1), UTRSequence.alphabet.index("EOS"), dtype=np.int32)
    return np.concatenate([bos, x, eos], axis=1), y
