import pytest

import numpy as np
from ctypes.util import find_library


# Check to make sure the fallback mechanism works if there is no bundled
# dynamic library.
@pytest.mark.skip(reason="Patching seems to fail in CI")
def test_fallback(mocker):
    def fake_load_library(lib_name, path):
        if lib_name in ["libcufinufft", "cufinufft"]:
            raise OSError()
        else:
            return np.ctypeslib.load_library(lib_name, path)

    # Block out the bundled library.
    mocker.patch("numpy.ctypeslib.load_library", fake_load_library)

    # Make sure an error is raised if no system library is found.
    if find_library("cufinufft") is None:
        with pytest.raises(ImportError, match="suitable cufinufft"):
            import cufinufft
    else:
        import cufinufft
