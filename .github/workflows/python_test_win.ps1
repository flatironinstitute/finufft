python -m pip install finufft -f .\wheelhouse\
if (-not $?) {throw "Failed to pip install finufft"}
python python/finufft/test/run_accuracy_tests.py
if (-not $?) {throw "Tests failed"}
python python/finufft/examples/simple1d1.py
if (-not $?) {throw "Simple1d1 test failed"}
