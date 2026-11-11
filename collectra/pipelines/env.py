"""Environment variables a user stores for a pipeline, outside the pipeline.

Each pipeline has a file in the user's config folder named after the pipeline,
for example ``~/.config/collectra/env/hespi.yaml``::

    variables:
      HF_HOME: /data/scratch/hf
    private:
    - ANTHROPIC_API_KEY
    override:
    - HF_HOME

Private values are kept in the operating system's keyring (the macOS Keychain,
Windows Credential Manager or the Linux Secret Service) and only their names are
written to the file. Reading a pipeline with ``Pipeline.from_file`` puts the
variables into the environment. A variable already set in the shell is left as
it is, unless the stored variable is listed under ``override``.
"""

from __future__ import annotations

import getpass
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cappa
import yaml

from ..cli import Positional, command

__all__ = ["EnvError", "EnvEntry", "PipelineEnv"]

NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def env_directory() -> Path:
    """The folder that holds each pipeline's stored variables."""
    from platformdirs import user_config_dir

    return Path(user_config_dir("collectra", appauthor=False)) / "env"


def keyring_backend():
    """The operating system's keyring, as chosen by the ``keyring`` package."""
    import keyring

    return keyring.get_keyring()


class EnvError(ValueError):
    """A stored variable cannot be read, written or found."""


@dataclass(frozen=True)
class EnvEntry:
    """One stored variable.

    ``value`` is None for a private variable the keyring cannot give back.
    ``overridden`` is True when the shell's value is used instead of this one.
    ``override`` is True when this value is to replace the shell's, and
    ``replaces_shell`` when it has replaced a value the shell set.
    """

    name: str
    value: str | None
    private: bool
    overridden: bool
    override: bool = False
    replaces_shell: bool = False

    def display(self, reveal: bool = False) -> str:
        """The value to show: private values are masked unless ``reveal``."""
        if self.value is None:
            return ""
        if not self.private or reveal:
            return self.value
        if len(self.value) >= 12:
            return f"{self.value[:4]}…{self.value[-4:]}"
        return "••••"


@dataclass
class _Stored:
    """The contents of a pipeline's file."""

    variables: dict[str, str]
    private: list[str]
    override: list[str]

    def __contains__(self, name: str) -> bool:
        return name in self.variables or name in self.private

    def names(self) -> list[str]:
        return sorted([*self.variables, *self.private])

    def discard(self, name: str) -> None:
        self.variables.pop(name, None)
        for names in (self.private, self.override):
            if name in names:
                names.remove(name)


def _console():
    from rich.console import Console

    return Console(highlight=False)


def _say(*parts) -> None:
    """Print a message built from text and (text, style) pairs, never as markup."""
    from rich.text import Text

    _console().print(Text.assemble(*parts), soft_wrap=True)


def _count(number: int) -> str:
    return f"{number} variable" if number == 1 else f"{number} variables"


class PipelineEnv:
    """The variables stored for the pipeline called ``name``."""

    def __init__(self, name: str):
        self.name = name
        # The variables this session put into the environment. Any other
        # variable already in the environment came from the shell.
        self._applied: set[str] = set()
        # The shell's values of the variables an override replaced.
        self._shell: dict[str, str] = {}

    @property
    def path(self) -> Path:
        stem = re.sub(r"[/\\\x00-\x1f]", "_", self.name).strip()
        if stem in {"", ".", ".."}:
            stem = "pipeline"
        return env_directory() / f"{stem}.yaml"

    @property
    def service(self) -> str:
        """The keyring service under which private values are kept."""
        return f"collectra:{self.name}"

    @staticmethod
    def keyring_available() -> bool:
        """Whether this machine has a keyring that can hold private values."""
        from keyring.backends import fail

        return not isinstance(keyring_backend(), fail.Keyring)

    # Reading and writing the file

    def _read(self) -> _Stored:
        if not self.path.exists():
            return _Stored({}, [], [])
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise TypeError("expected a mapping")
            return _Stored(
                {
                    str(key): str(value)
                    for key, value in (data.get("variables") or {}).items()
                },
                [str(name) for name in data.get("private") or []],
                [str(name) for name in data.get("override") or []],
            )
        except (yaml.YAMLError, TypeError, AttributeError) as error:
            raise EnvError(f"Cannot read {self.path}: {error}") from error

    def _write(self, stored: _Stored) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        data = {"variables": stored.variables, "private": stored.private}
        if stored.override:
            data["override"] = stored.override
        partial = directory / f".{self.path.name}.partial"
        try:
            descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)

    # The keyring

    def _secret(self, name: str) -> str | None:
        from keyring.errors import KeyringError

        try:
            return keyring_backend().get_password(self.service, name)
        except KeyringError:
            return None

    def _forget_secret(self, name: str) -> None:
        from keyring.errors import KeyringError

        try:
            keyring_backend().delete_password(self.service, name)
        except KeyringError:
            pass  # not there, or no keyring to hold it

    def _value(self, stored: _Stored, name: str) -> str | None:
        return (
            stored.variables[name] if name in stored.variables else self._secret(name)
        )

    # The environment of this session

    def _overridden(self, name: str) -> bool:
        return name in os.environ and name not in self._applied

    def _apply(self, name: str, value: str | None, override: bool = False) -> None:
        if value is None:
            return
        if self._overridden(name):
            if not override:
                return
            self._shell[name] = os.environ[name]
        os.environ[name] = value
        self._applied.add(name)

    def _unapply(self, name: str) -> None:
        if name not in self._applied:
            return
        if name in self._shell:
            os.environ[name] = self._shell.pop(name)
        else:
            os.environ.pop(name, None)
        self._applied.discard(name)

    def _say_overridden(self, name: str) -> None:
        _say(
            ("! ", "bold yellow"),
            (name, "bold"),
            f" is set in the shell, so the stored value for {self.name} is not used.",
        )

    def _say_replaces(self, name: str) -> None:
        _say(
            ("» ", "bold blue"),
            (name, "bold"),
            f" for {self.name} replaces the value set in the shell.",
        )

    def _report(self, name: str) -> None:
        """Say when the shell's value and the stored one are not the same one."""
        if self._overridden(name):
            self._say_overridden(name)
        elif name in self._shell:
            self._say_replaces(name)

    # The interface used by the command line and the GUI

    def entries(self) -> list[EnvEntry]:
        """Every stored variable, sorted by name."""
        stored = self._read()
        return [
            EnvEntry(
                name,
                self._value(stored, name),
                name in stored.private,
                self._overridden(name),
                name in stored.override,
                name in self._shell,
            )
            for name in stored.names()
        ]

    def load(self) -> list[str]:
        """Put the stored variables into the environment.

        A variable the shell already set is left alone unless it is stored with
        ``override``. Returns the names of the variables set, and says which
        differ from the shell's. Problems are warnings, so a pipeline still
        loads when its stored variables cannot be read.
        """
        try:
            entries = self.entries()
        except EnvError as error:
            warnings.warn(str(error), stacklevel=2)
            return []
        loaded = []
        for entry in entries:
            if entry.value is None:
                warnings.warn(
                    f"{entry.name} is stored as private for {self.name} but is not "
                    "in the keyring on this machine, so it is not set.",
                    stacklevel=2,
                )
                continue
            self._apply(entry.name, entry.value, entry.override)
            self._report(entry.name)
            if entry.name in self._applied:
                loaded.append(entry.name)
        return loaded

    def store(
        self, name: str, value: str, *, private: bool = False, override: bool = False
    ) -> None:
        """Store ``value`` for ``name``.

        ``private`` keeps it in the keyring; ``override`` makes it replace a
        value set in the shell.
        """
        if not NAME_PATTERN.fullmatch(name):
            raise EnvError(f"{name!r} is not a valid environment variable name")
        stored = self._read()
        stored.discard(name)
        if private:
            from keyring.errors import KeyringError

            try:
                keyring_backend().set_password(self.service, name, value)
            except KeyringError as error:
                raise EnvError(
                    "No secure keyring is available on this machine, so private "
                    f"values cannot be stored ({error}). Store it without "
                    "--private, which keeps it in a file only you can read, or "
                    "set it in the environment instead."
                ) from error
            stored.private.append(name)
        else:
            self._forget_secret(name)
            stored.variables[name] = value
        if override:
            stored.override.append(name)
        self._write(stored)
        self._unapply(name)
        self._apply(name, value, override)

    def set_override(self, name: str, override: bool) -> None:
        """Choose whether the stored ``name`` replaces a value set in the shell."""
        stored = self._read()
        if name not in stored:
            raise EnvError(f"{name} is not stored for {self.name}")
        if name in stored.override:
            stored.override.remove(name)
        if override:
            stored.override.append(name)
        self._write(stored)
        self._unapply(name)
        self._apply(name, self._value(stored, name), override)

    def remove(self, name: str) -> None:
        """Delete the variable ``name``."""
        stored = self._read()
        if name not in stored:
            raise EnvError(f"{name} is not stored for {self.name}")
        if name in stored.private:
            self._forget_secret(name)
        stored.discard(name)
        self._write(stored)
        self._unapply(name)

    def remove_all(self) -> list[str]:
        """Delete every variable, returning their names."""
        stored = self._read()
        for name in stored.private:
            self._forget_secret(name)
        for name in stored.names():
            self._unapply(name)
        self.path.unlink(missing_ok=True)
        return stored.names()

    # Commands

    @staticmethod
    def _fail(error: EnvError):
        return cappa.Exit(str(error), code=1)

    @command(name="set")
    def cli_set(
        self,
        name: str,
        value: Annotated[str | None, Positional] = None,
        private: bool = False,
        override: bool = False,
    ):
        """Store a variable. Without a value, it is asked for without being shown.

        --private keeps it in the system keyring. --override makes it replace a
        value set in the shell.
        """
        if value is None:
            value = getpass.getpass(f"Value for {name}: ")
        try:
            self.store(name, value, private=private, override=override)
        except EnvError as error:
            raise self._fail(error) from error
        where = "the keyring" if private else str(self.path)
        _say(
            ("✓ ", "bold green"),
            "Stored ",
            (name, "bold"),
            f" for {self.name} in ",
            (where, "dim"),
        )
        self._report(name)

    @command(name="override")
    def cli_override(self, name: str, off: bool = False):
        """Make a stored variable replace a value set in the shell (--off: stop)."""
        try:
            self.set_override(name, not off)
        except EnvError as error:
            raise self._fail(error) from error
        if self._overridden(name) or name in self._shell:
            self._report(name)
        else:
            change = "no longer overrides" if off else "now overrides"
            _say(
                ("✓ ", "bold green"),
                (name, "bold"),
                f" for {self.name} {change} the shell.",
            )

    @command(name="show")
    def cli_show(self, reveal: bool = False):
        """Show the stored variables, masking private ones unless --reveal is given."""
        from rich.table import Table
        from rich.text import Text

        try:
            entries = self.entries()
        except EnvError as error:
            raise self._fail(error) from error
        if not entries:
            _say(f"No environment variables are stored for {self.name}.")
            return
        table = Table(
            title=Text(f"Environment variables for {self.name}", style="bold"),
            title_justify="left",
        )
        table.add_column("Name", style="bold cyan", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_column("Storage", style="magenta")
        table.add_column("Status")
        for entry in entries:
            if entry.value is None:
                status = Text("not in the keyring on this machine", style="red")
            elif entry.overridden:
                status = Text("overridden by the shell", style="yellow")
            elif entry.replaces_shell:
                status = Text("in use, overrides the shell", style="bold green")
            else:
                status = Text("in use", style="green")
            if entry.override and not entry.replaces_shell:
                status.append(" (override on)", style="dim")
            table.add_row(
                Text(entry.name),
                Text(entry.display(reveal)),
                "keyring" if entry.private else "file",
                status,
            )
        console = _console()
        console.print(table)
        console.print(
            Text.assemble("Stored in ", (str(self.path), "dim")), soft_wrap=True
        )

    @command(name="delete")
    def cli_delete(self, name: str):
        """Delete a stored variable."""
        try:
            self.remove(name)
        except EnvError as error:
            raise self._fail(error) from error
        _say(("✓ ", "bold green"), "Deleted ", (name, "bold"), f" for {self.name}")

    @command(name="clear")
    def cli_clear(self, yes: bool = False):
        """Delete all the stored variables. --yes skips asking."""
        from rich.prompt import Confirm

        try:
            count = len(self.entries())
        except EnvError as error:
            raise self._fail(error) from error
        if not count:
            _say(f"No environment variables are stored for {self.name}.")
            return
        question = f"Delete {_count(count)} stored for {self.name}?"
        if not yes and not Confirm.ask(question, default=False, console=_console()):
            _say("Nothing deleted.")
            return
        removed = self.remove_all()
        _say(("✓ ", "bold green"), f"Deleted {_count(len(removed))} for {self.name}")
