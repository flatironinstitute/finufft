import pytest

import utils


def pytest_addoption(parser):
    parser.addoption("--framework", action="append", default=[], help="List of frameworks")

def pytest_generate_tests(metafunc):
    if "framework" in metafunc.fixturenames:
        metafunc.parametrize("framework", metafunc.config.getoption("framework"))

@pytest.fixture
def to_gpu(framework):
    to_gpu, _ = utils.transfer_funcs(framework)

    return to_gpu


@pytest.fixture
def to_cpu(framework):
    _, to_cpu = utils.transfer_funcs(framework)

    return to_cpu
