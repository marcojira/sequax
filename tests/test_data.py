import pickle

import numpy as np

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


def test_gfp_loader_adds_special_tokens(tmp_path):
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.array([[0, 2]], dtype=np.int32))
    np.save(y_path, np.array([2.5], dtype=np.float32))

    x, y = load_gfp_data(x_path, y_path)

    assert np.array_equal(x, np.array([[0, 3, 5, 2]], dtype=np.int32))
    assert np.array_equal(y, np.array([2.5], dtype=np.float32))


def test_utr_loader_uses_environment_alphabet(tmp_path):
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.array([[0, 2, 3]], dtype=np.int32))
    np.save(y_path, np.array([1.5], dtype=np.float32))

    x, y = load_utr_data(x_path, y_path)

    assert np.array_equal(x, np.array([[0, 3, 5, 6, 2]], dtype=np.int32))
    assert np.array_equal(y, np.array([1.5], dtype=np.float32))
