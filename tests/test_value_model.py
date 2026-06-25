import numpy as np

from arcagi3.value_model import ValueModel


def test_value_model_ranks_positive_above_negative():
    rows = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 0.0),
        (np.array([1.0, 0.0, 1.0]), 1.0),
        (np.array([0.0, 1.0, 1.0]), 0.0),
    ]

    model = ValueModel(input_dim=3)
    model.fit(rows, epochs=50)

    assert model.score(np.array([1.0, 0.0, 0.0])) > model.score(np.array([0.0, 1.0, 0.0]))
