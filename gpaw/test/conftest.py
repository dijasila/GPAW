import os

import pytest


@pytest.fixture
def in_tmp_dir(tmpdir):
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        yield
    finally:
        os.chdir(cwd)
