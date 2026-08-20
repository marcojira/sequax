import pickle

import numpy as np
import pytest

from sequax.tasks import load_amp_data, load_gfp_data, load_utr_data


def test_amp_loader_tokenizes_strings(tmp_path):
    negative_path = tmp_path / "negative.pkl"
    positive_path = tmp_path / "positive.pkl"
    with negative_path.open("wb") as file:
        pickle.dump(["AC"], file)
    with positive_path.open("wb") as file:
        pickle.dump(["D"], file)

    x, y = load_amp_data(negative_path, positive_path, max_length=3)

    assert np.array_equal(x, np.array([[0, 3, 4, 2, 1], [0, 5, 2, 1, 1]]))
    assert np.array_equal(y, np.array([0.0, 1.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("loader", "raw_x", "expected_x"),
    [
        (load_gfp_data, [[0, 2]], [[0, 3, 5, 2]]),
        (load_utr_data, [[0, 2, 3]], [[0, 3, 5, 6, 2]]),
    ],
)
def test_array_loaders_add_special_tokens(tmp_path, loader, raw_x, expected_x):
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.array(raw_x, dtype=np.int32))
    np.save(y_path, np.array([[2.5]], dtype=np.float32))

    x, y = loader(x_path, y_path)

    assert np.array_equal(x, np.array(expected_x, dtype=np.int32))
    assert np.array_equal(y, np.array([2.5], dtype=np.float32))
