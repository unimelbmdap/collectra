"""Core workflows remain usable when YOLO dependencies are absent."""

import subprocess
import sys
import textwrap


def test_core_and_help_without_yolo_dependencies():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path

        class WithoutYOLO(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'ultralytics', 'drawyolo'}:
                    raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])

        sys.meta_path.insert(0, WithoutYOLO())
        import cappa
        from PIL import Image as PILImage
        from collectra import (
            Image, LineDetectorRLSA, ObjectDetectionYOLO, ImageClassifierYOLO,
            ImageClassifierTorchvision, ObjectDetectionRFDETR,
        )
        from collectra.cli import invoke

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.png'
            PILImage.new('RGB', (80, 80), 'white').save(path)
            assert LineDetectorRLSA('lines').run(Image(name='page', data=path))
            for cls in [ObjectDetectionYOLO, ImageClassifierYOLO]:
                task = cls('task', model='unused.pt')
                try:
                    invoke(task, ['train', '--help'], name='task')
                except cappa.HelpExit:
                    pass
                else:
                    raise AssertionError('Expected help exit')
                for operation in [task._load, task._init_model, task._train]:
                    try:
                        operation()
                    except ImportError as error:
                        assert 'collectra[yolo]' in str(error)
                    else:
                        raise AssertionError('Expected optional dependency error')
            try:
                ObjectDetectionYOLO('detector')._preview_assets(Path(folder), [])
            except ImportError as error:
                assert 'collectra[yolo]' in str(error)
            else:
                raise AssertionError('Expected drawyolo dependency error')
        assert 'ultralytics' not in sys.modules
        assert 'drawyolo' not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_locked_dependency_graph_excludes_optional_backends():
    """Catch indirect reintroduction, including through drawyolo or extras."""
    import tomllib
    from collections import defaultdict
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = defaultdict(list)
    for package in lock["package"]:
        packages[package["name"]].append(package)

    def reachable(extras=()):
        pending = [("collectra", tuple(extras))]
        seen = set()
        while pending:
            name, extras = pending.pop()
            if (name, extras) in seen:
                continue
            seen.add((name, extras))
            assert name in packages, f"Missing lockfile entry: {name}"
            for package in packages[name]:
                dependencies = list(package.get("dependencies", []))
                for extra in extras:
                    dependencies.extend(
                        package.get("optional-dependencies", {}).get(extra, [])
                    )
                # Include every platform/version branch conservatively.
                for dependency in dependencies:
                    pending.append(
                        (dependency["name"], tuple(dependency.get("extra", [])))
                    )
        return {name for name, _ in seen}

    excluded = {
        "ultralytics", "ultralytics-thop", "ultralytics-platform", "drawyolo",
        "surya-ocr", "pyqt5", "pyqt6", "pyqt6-webengine", "pyqtwebengine",
    }
    assert not (reachable() & excluded)
    assert {"ultralytics", "drawyolo"} <= reachable(["yolo"])
