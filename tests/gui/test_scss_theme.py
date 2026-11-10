import pytest
import sass

from collectra.gui import scss_theme
from collectra.gui.scss_theme import ScssThemeError, compile_theme


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(scss_theme, "user_cache_dir", lambda *args: str(tmp_path / "cache"))


def write(tmp_path, text, name="theme.scss"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_variables_reach_bootstrap_and_the_app_properties(tmp_path):
    css = compile_theme(write(tmp_path, "$primary: #123456;\n$body-bg: #000;\n"))

    assert ".btn-primary" in css
    assert "#123456" in css
    assert "--app-primary: #123456" in css
    assert "--app-primary-rgb: 18, 52, 86" in css
    assert "--app-bg: #000" in css


def test_result_is_cached_until_the_source_changes(tmp_path, monkeypatch):
    path = write(tmp_path, "$primary: #123456;\n")
    first = compile_theme(path)
    monkeypatch.setattr(sass, "compile", lambda **kw: pytest.fail("recompiled"))
    assert compile_theme(path) == first

    monkeypatch.undo()
    monkeypatch.setattr(scss_theme, "user_cache_dir", lambda *args: str(tmp_path / "cache"))
    path.write_text("$primary: #654321;\n")
    assert "--app-primary: #654321" in compile_theme(path)


def test_a_syntax_error_is_reported(tmp_path):
    with pytest.raises(ScssThemeError, match="theme.scss"):
        compile_theme(write(tmp_path, "$primary: ;\n"))


def test_backend_compiles_a_scss_theme(backend_with_extension, tmp_path):
    collectra = tmp_path / "proj.collectra"
    collectra.mkdir()
    (collectra / "mine.scss").write_text("$primary: #123456;\n")
    backend = backend_with_extension
    backend._parent_folder = str(tmp_path)
    backend._pipeline.pipeline_metadata["theme"] = "mine.scss"

    result = backend.get_theme_css()

    assert result["success"] is True
    assert result["path"].endswith("mine.scss")
    assert "--app-primary: #123456" in result["css"]
    assert "warning" not in result


def test_backend_falls_back_to_the_default_theme_when_scss_is_broken(
    backend_with_extension, tmp_path
):
    collectra = tmp_path / "proj.collectra"
    collectra.mkdir()
    (collectra / "mine.scss").write_text("$primary: ;\n")
    backend = backend_with_extension
    backend._parent_folder = str(tmp_path)
    backend._pipeline.pipeline_metadata["theme"] = "mine.scss"

    result = backend.get_theme_css()

    assert result["success"] is True
    assert "mine.scss" in result["warning"]
    assert "--dag-bg" in result["css"]  # the bundled default
