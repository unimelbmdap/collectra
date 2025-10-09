import pytest, traceback
from collectra.utils import error_msg
from rich import print


@pytest.fixture
def debug(request):
    def debugger(e: Exception):
        print(error_msg(f"Test failed. Retaining log_dir for debugging."))
        traceback.print_exc()
        if request.config.getoption("--pdb"):
            breakpoint()  # Debug here
        raise e

    return debugger
