import random

import numpy as np

from cordconv2d.utils.main import RunningStd

def test_plain_std_equal_running(running_std):
    data = np.random.randint(low=1, high=1000, size=(random.randint(10, 100), 2))
    rs: RunningStd = running_std(n_features=2)
    rs.update(data)
    print(rs.std(), data.std(axis=0))
    assert (rs.std() == data.std(axis=0)).all()

def test_plain_std_equal_incrementaly_running(running_std):
    data = np.random.randint(low=1, high=1000, size=(random.randint(10, 100), 2))
    rs: RunningStd = running_std(n_features=2)
    b1, b2 = np.array_split(data, 2)
    rs.update(b1)
    rs.update(b2)
    np.testing.assert_allclose(rs.std(), data.std(axis=0))