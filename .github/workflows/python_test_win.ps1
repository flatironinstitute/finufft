python -m pip install --pre finufft -f .\wheelhouse\
if (-not $?) {throw "Failed to pip install finufft"}
python python/test/run_accuracy_tests.py
if (-not $?) {throw "Tests failed"}
python python/examples/simple1d1.py
if (-not $?) {throw "Simple1d1 test failed"}
