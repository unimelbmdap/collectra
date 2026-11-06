import builtins
import sys

import pytest

from collectra.tasks.image_segmentation import sam3 as sam3_module


@pytest.fixture
def import_fails(monkeypatch):
    """Make ``import sam3`` fail as if ``missing`` could not be imported."""

    def fail(missing=None, error=None):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sam3":
                if error is not None:
                    raise error
                raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "sam3", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

    return fail


def test_a_missing_sam3_says_how_to_install_it(import_fails, monkeypatch):
    import_fails("sam3")
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ImportError) as raised:
        sam3_module._sam3()
    message = str(raised.value)
    assert "SAM 3 is not installed" in message
    assert "uv sync --extra sam3" in message
    assert "git+https://github.com/facebookresearch/sam3.git" in message
    assert "only works on Linux" not in message


def test_a_missing_sam3_on_a_mac_says_it_needs_linux(import_fails, monkeypatch):
    import_fails("sam3")
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ImportError, match="only works on Linux with an NVIDIA GPU"):
        sam3_module._sam3()


def test_a_missing_dependency_of_sam3_is_named(import_fails, monkeypatch):
    import_fails("einops")
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ImportError) as raised:
        sam3_module._sam3()
    message = str(raised.value)
    assert "SAM 3 is installed but needs the Python package 'einops'" in message
    assert "uv sync --extra sam3" in message
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_triton_on_a_mac_explains_that_it_only_exists_for_linux(
    import_fails, monkeypatch
):
    import_fails("triton")
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ImportError, match="triton.*only.*Linux"):
        sam3_module._sam3()


def test_other_errors_from_importing_sam3_are_not_hidden(import_fails):
    import_fails(error=RuntimeError("broken install"))
    with pytest.raises(RuntimeError, match="broken install"):
        sam3_module._sam3()


def test_an_installed_sam3_is_returned(monkeypatch):
    package = object()
    monkeypatch.setitem(sys.modules, "sam3", package)
    assert sam3_module._sam3() is package
