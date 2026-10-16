"""Smoke-test a built wheel without exposing the checkout or editable .pth files.

Run with the project's dependencies installed:
    python tests/packaging/check_wheel.py dist/collectra-*.whl
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main():
    wheel = Path(sys.argv[1]).resolve()
    dependency_paths = [path for path in sys.path if Path(path).name in {"site-packages", "dist-packages"}]
    with tempfile.TemporaryDirectory(prefix="collectra-wheel-") as directory:
        root = Path(directory)
        packages = root / "installed"
        with ZipFile(wheel) as archive:
            archive.extractall(packages)
        pipeline = root / "pipeline.yaml"
        pipeline.write_text(
            "collectra_pipeline_metadata: {name: packaged, ext: collectra, version: '1.0'}\n"
        )
        # -I -S excludes the working directory, PYTHONPATH and editable .pth
        # processing. Dependencies remain available, but our code must come
        # from the wheel. This catches missing top-level packages like utils.
        code = f"""
import sys
from pathlib import Path
sys.path[:0] = {json.dumps([str(packages), *dependency_paths])}
import collectra.main
import collectra.gui.backend
import utils.get_types
import utils.main
for module in (collectra.main, collectra.gui.backend, utils.get_types, utils.main):
    assert Path(module.__file__).resolve().is_relative_to(Path({str(packages)!r}).resolve())
collectra.main.main(['--pipeline', {str(pipeline)!r}, '--help'])
"""
        subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=root, check=True)


if __name__ == "__main__":
    main()
