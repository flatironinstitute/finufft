## Basic tests of FINUFFT CPU Python wrappers

To install the python wrappers for FINUFFT
see ../../../docs/install.rst

Then, from the present directory, you may run human-readable tests as follows:

```sh
python3 run_accuracy_tests.py
python3 run_speed_tests.py
```

You may run the pass-fail tests used in CI via:

```sh
python3 -m pytest .
```

The codes `accuracy_speed_tests.py` and `../examples/*` illustrate how to call
FINUFFT from Python.
