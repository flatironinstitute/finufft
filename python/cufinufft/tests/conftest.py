def pytest_addoption(parser):
    parser.addoption("--framework", action="append", default=[], help="List of frameworks")

def pytest_generate_tests(metafunc):
    if "framework" in metafunc.fixturenames:
        metafunc.parametrize("framework", metafunc.config.getoption("framework"))
