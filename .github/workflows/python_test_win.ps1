python -m pip install --pre finufft -f .\wheelhouse\
if (-not $?) {throw "Failed to pip install finufft"}
python python/finufft/test/run_accuracy_tests.py
if (-not $?) {throw "Tests failed"}
python python/finufft/examples/simple1d1.py
if (-not $?) {throw "Simple1d1 test failed"}
python -m pip install pytest
python -m pytest python/finufft/test
if (-not $?) {throw "Pytest suite failed"}
