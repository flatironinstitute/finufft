import pytest

import numpy as np
from ctypes.util import find_library

@pytest.mark.skip(reason="Patching seems to fail in CI")
def test_fallback(mocker):
    def fake_load_library(lib_name, path):
        if lib_name in ["libfinufft", "finufft"]:
            raise OSError()
        else:
            return np.ctypeslib.load_library(lib_name, path)

    mocker.patch("numpy.ctypeslib.load_library", fake_load_library)

    if find_library("finufft") is None:
        with pytest.raises(ImportError, match="suitable finufft"):
            import finufft
    else:
        import finufft
