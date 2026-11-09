import pytest

from cordconv2d.utils.main import RunningStd

@pytest.fixture
def running_std():
    def _make(n_features = None):
        return RunningStd(n_features)
    return _make