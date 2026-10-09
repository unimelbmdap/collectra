"""Create macOS application bundles for pipeline GUI launchers."""

import hashlib
import plistlib
import shlex
import shutil
import sys
import tempfile
from pathlib import Path


def install_app(
    pipeline_path: Path,
    *,
    name: str = "Collectra",
    app_dir: Path | None = None,
    icon: Path | None = None,
    inputs: tuple[Path, ...] = (),
    force: bool = False,
) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("install-app is only supported on macOS")
    if not name.strip() or name in {".", ".."} or any(c in name for c in "/:\\"):
        raise ValueError(
            "App name must be a nonempty filename without slashes or colons"
        )
    pipeline_path = Path(pipeline_path).expanduser().resolve()
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    icon = (
        Path(icon).expanduser().resolve()
        if icon
        else Path(__file__).resolve().parents[1] / "gui/assets/CollectraLogo.icns"
    )
    if icon.suffix.lower() not in {".icns", ".png"} or not icon.is_file():
        raise ValueError("App icon must be an existing .icns or PNG file")
    destination = (app_dir or Path.home() / "Applications").expanduser().resolve()
    bundle = destination / f"{name}.app"
    if (bundle.exists() or bundle.is_symlink()) and not force:
        raise FileExistsError(
            f"Application already exists: {bundle}; use --force to replace it"
        )
    arguments = [
        sys.executable,
        "-m",
        "collectra.main",
        "--pipeline",
        str(pipeline_path),
        "gui",
    ]
    arguments.extend(str(Path(value).expanduser().resolve()) for value in inputs)
    identifier = hashlib.sha256(str(pipeline_path).encode()).hexdigest()[:16]
    destination.mkdir(parents=True, exist_ok=True)
    # Build first so failed icon conversion leaves the installed app intact.
    staging = Path(tempfile.mkdtemp(prefix=".collectra-app-", dir=destination))
    new_bundle = staging / bundle.name
    backup = staging / "previous"
    try:
        contents = new_bundle / "Contents"
        executable_dir = contents / "MacOS"
        resources = contents / "Resources"
        executable_dir.mkdir(parents=True)
        resources.mkdir()
        launcher = executable_dir / "collectra"
        launcher.write_text(
            "#!/bin/sh\n"
            f"cd {shlex.quote(str(pipeline_path.parent))} || exit 1\n"
            f"exec {shlex.join(arguments)}\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        if icon.suffix.lower() == ".icns":
            shutil.copyfile(icon, resources / "AppIcon.icns")
        else:
            from PIL import Image

            with Image.open(icon) as image:
                image.convert("RGBA").save(resources / "AppIcon.icns", format="ICNS")
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(
                {
                    "CFBundleName": name,
                    "CFBundleDisplayName": name,
                    "CFBundleIdentifier": f"org.collectra.pipeline.{identifier}",
                    "CFBundleExecutable": "collectra",
                    "CFBundleIconFile": "AppIcon.icns",
                    "CFBundlePackageType": "APPL",
                    "CFBundleVersion": "1",
                    "CFBundleShortVersionString": "1.0",
                    "NSHighResolutionCapable": True,
                },
                stream,
            )
        if bundle.exists() or bundle.is_symlink():
            if not force:
                raise FileExistsError(
                    f"Application already exists: {bundle}; use --force to replace it"
                )
            bundle.rename(backup)
        try:
            new_bundle.rename(bundle)
        except Exception:
            if backup.exists() or backup.is_symlink():
                backup.rename(bundle)
            raise
    finally:
        shutil.rmtree(staging)
    return bundle
