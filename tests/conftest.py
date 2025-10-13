import pytest, shutil

pytest_plugins = [
    "tests.fixtures.functions",
    "tests.fixtures.paths",
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.types",
]

@pytest.fixture(autouse=True)
def cleanup_tmp_dir():
    shutil.rmtree("log_dir", ignore_errors=True)


