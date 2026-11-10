"""Compile a project's SCSS theme (a few Bootstrap variables) into the GUI's CSS.

A pipeline's ``theme: FILENAME.scss`` holds Bootstrap 5 variable overrides
(``$primary``, ``$body-bg``, ...). They are compiled with the bundled
Bootstrap sources by libsass, then the GUI's own ``--app-*``
custom properties are derived from the same variables, so the theme file does
not have to know about them. The result is cached by content, so only the
first load after an edit pays for the compile.
"""

import hashlib
import tempfile
from pathlib import Path

from platformdirs import user_cache_dir

BOOTSTRAP_VERSION = "5.3.8"
BOOTSTRAP_SCSS = Path(__file__).parent / "vendor" / "bootstrap" / "scss"

# Bump when ``_APP_VARIABLES`` or the way the sheet is assembled changes, so a
# cached sheet from before the change is not used.
_CACHE_VERSION = "1"

# Derives the GUI's --app-* properties from the Bootstrap variables the theme
# set (or Bootstrap's defaults for those it did not).
_APP_VARIABLES = """
:root {
    --app-bg: #{$body-bg};
    --app-fg: #{$body-color};
    --app-fg-bright: #{mix($body-color, if(lightness($body-bg) < 50, #fff, #000), 75%)};
    --app-fg-muted: #{mix($body-color, $body-bg, 65%)};
    --app-surface: #{mix($body-bg, $body-color, 92%)};
    --app-surface-alt: #{mix($body-bg, $body-color, 96%)};
    --app-border: #{mix($body-bg, $body-color, 84%)};
    --app-primary: #{$primary};
    --app-primary-rgb: #{red($primary)}, #{green($primary)}, #{blue($primary)};
    --app-primary-dark: #{shade-color($primary, 15%)};
    --app-primary-fg: #{color-contrast($primary)};
    --app-secondary: #{$secondary};
    --app-secondary-rgb: #{red($secondary)}, #{green($secondary)}, #{blue($secondary)};
    --app-secondary-fg: #{color-contrast($secondary)};
    --app-danger: #{$danger};
    --app-divider: #{rgba($primary, 0.084)};
    --app-font-family: #{inspect($font-family-sans-serif)};
}
"""


class ScssThemeError(Exception):
    """The theme could not be compiled; the message is for the user."""


def _cache_key(source: str, directory: Path) -> str:
    digest = hashlib.sha256()
    for part in (BOOTSTRAP_VERSION, _CACHE_VERSION, _APP_VARIABLES, source):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    # Partials the theme may @import from beside it
    try:
        for sibling in sorted(directory.glob("*.scss")):
            digest.update(sibling.read_bytes())
    except OSError:
        pass
    return digest.hexdigest()


def compile_theme(path: Path) -> str:
    """Return the CSS for the SCSS theme at ``path`` (cached by content)."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    cache_file = Path(user_cache_dir("collectra")) / "themes" / f"{_cache_key(source, path.parent)}.css"
    try:
        return cache_file.read_text(encoding="utf-8")
    except OSError:
        pass

    # The theme goes in between Bootstrap's functions and the rest of
    # Bootstrap, the order its docs prescribe for overriding variables.
    sheet = f'@import "functions";\n{source}\n@import "bootstrap";\n{_APP_VARIABLES}'
    try:
        import sass

        css = sass.compile(
            string=sheet,
            include_paths=[str(path.parent), str(BOOTSTRAP_SCSS)],
            output_style="compressed",
        )
    except ImportError as e:
        raise ScssThemeError("libsass is not installed (pip install libsass)") from e
    except sass.CompileError as e:
        raise ScssThemeError(f"{path.name}: {str(e).strip()}") from e
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=cache_file.parent, delete=False
        ) as handle:
            handle.write(css)
        Path(handle.name).replace(cache_file)
    except OSError:
        pass  # an unwritable cache only costs a recompile next time
    return css
