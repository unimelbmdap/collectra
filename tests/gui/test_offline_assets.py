"""Check that the GUI's executable assets and CSS dependencies are local."""

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2] / "collectra/gui"


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "script" and attrs.get("src"):
            self.references.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.references.append(attrs["href"])


def test_gui_asset_dependency_closure_is_local():
    assets = Assets()
    assets.feed((ROOT / "index.html").read_text())
    pending = [ROOT / reference for reference in assets.references]
    assert all(
        not urlsplit(reference).scheme and not reference.startswith("//")
        for reference in assets.references
    )
    visited = set()
    while pending:
        asset = pending.pop().resolve()
        if asset in visited:
            continue
        visited.add(asset)
        assert asset.is_relative_to(ROOT)
        assert asset.is_file(), f"Missing bundled asset: {asset}"
        if asset.suffix == ".css":
            css = asset.read_text()
            for reference in re.findall(r'url\(\s*["\']?([^\)"\']+)', css):
                if reference.startswith("data:"):
                    continue
                assert not urlsplit(reference).scheme and not reference.startswith(
                    "//"
                ), reference
                pending.append(asset.parent / unquote(urlsplit(reference).path))
            assert not re.search(r'@import\s+["\']https?://', css)


def test_editor_and_image_controls_do_not_download_assets():
    html = (ROOT / "index.html").read_text()
    assert "autoDownloadFontAwesome: false" in html
    assert "spellChecker: false" in html
    assert 'prefixUrl: "openseadragon/images/"' in html
    assert 'prefixUrl: "https://' not in html
    for name in ["zoomin", "zoomout", "home", "fullpage"]:
        for state in ["rest", "grouphover", "hover", "pressed"]:
            assert (ROOT / f"openseadragon/images/{name}_{state}.png").is_file()
