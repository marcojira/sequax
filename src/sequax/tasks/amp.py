import pickle
from pathlib import Path

import numpy as np

from sequax.env import SequenceEnv
from sequax.proxies.base import Proxy

AMP_ALPHABET = (
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


class AMPSequence(SequenceEnv):
    name = "AMP"
    alphabet = AMP_ALPHABET

    def __init__(self, proxy: Proxy, min_length: int = 12, max_length: int = 60):
        super().__init__(self.alphabet, proxy, min_length, max_length)


def load_amp_data(
    negative_path: str | Path,
    positive_path: str | Path,
    max_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and tokenize the positive and negative AMP datasets."""
    with Path(negative_path).open("rb") as file:
        negative = pickle.load(file)
    with Path(positive_path).open("rb") as file:
        positive = pickle.load(file)

    x = np.concatenate(
        [
            _tokenize_sequences(negative, max_length),
            _tokenize_sequences(positive, max_length),
        ]
    )
    y = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])
    return x, y.astype(np.float32)


def _tokenize_sequences(sequences: list[str], max_length: int) -> np.ndarray:
    token_to_id = {token: index for index, token in enumerate(AMPSequence.alphabet)}
    rows = []
    for sequence in sequences:
        if len(sequence) > max_length:
            raise ValueError(f"Sequence length {len(sequence)} exceeds {max_length}")
        tokens = [token_to_id["BOS"]]
        tokens.extend(token_to_id[token] for token in sequence)
        tokens.append(token_to_id["EOS"])
        tokens.extend([token_to_id["PAD"]] * (max_length - len(sequence)))
        rows.append(tokens)
    return np.asarray(rows, dtype=np.int32)
