"""Environment variables stored per pipeline, outside the pipeline itself.

Each pipeline has a file in the user's config folder, named after the pipeline.
Private values are kept in the operating system's keyring and only their names
are written to the file. Loading a pipeline puts the variables into the
environment, unless the environment already has them.
"""

import os
import stat

import cappa
import pytest
import yaml
from keyring.backends import fail

from collectra.cli import invoke
from collectra.pipelines import env as env_module
from collectra.pipelines.base import Pipeline
from collectra.pipelines.env import EnvError, PipelineEnv

PLAIN = "COLLECTRA_TEST_PLAIN"
SECRET = "COLLECTRA_TEST_SECRET"
OTHER = "COLLECTRA_TEST_OTHER"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (PLAIN, SECRET, OTHER):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def keyring(pipeline_env_keyring):
    return pipeline_env_keyring


@pytest.fixture
def no_keyring(monkeypatch):
    monkeypatch.setattr(env_module, "keyring_backend", fail.Keyring)


def stored(store: PipelineEnv) -> dict:
    return yaml.safe_load(store.path.read_text())


def write_pipeline(folder, name="hespi"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pipeline.yaml").write_text(
        f"{name}:\n  type: collectra.Pipeline\n  ext: hp\n  version: '1'\n"
    )
    return folder


# Where the variables are kept


def test_each_pipeline_has_a_file_named_after_it(pipeline_env_directory):
    assert PipelineEnv("hespi").path == pipeline_env_directory / "hespi.yaml"
    assert PipelineEnv("a/b\\c").path == pipeline_env_directory / "a_b_c.yaml"
    assert PipelineEnv("..").path == pipeline_env_directory / "pipeline.yaml"


def test_the_default_directory_is_the_user_config_folder(monkeypatch):
    monkeypatch.undo()  # the autouse fixture points the directory at a tmp folder
    from platformdirs import user_config_dir

    expected = os.path.join(user_config_dir("collectra", appauthor=False), "env")
    assert str(env_module.env_directory()) == expected


def test_the_default_keyring_is_the_one_keyring_chooses(monkeypatch):
    import keyring as keyring_package

    monkeypatch.undo()
    chosen = object()
    monkeypatch.setattr(keyring_package, "get_keyring", lambda: chosen)
    assert env_module.keyring_backend() is chosen


def test_nothing_stored_means_no_variables():
    store = PipelineEnv("hespi")
    assert store.entries() == []
    assert store.load() == []


def test_a_plain_value_is_written_only_the_owner_can_read(pipeline_env_directory):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "/data/hf")
    assert stored(store) == {"variables": {PLAIN: "/data/hf"}, "private": []}
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(pipeline_env_directory.stat().st_mode) == 0o700
    assert not list(pipeline_env_directory.glob(".*partial"))


def test_a_private_value_goes_to_the_keyring_not_the_file(keyring):
    store = PipelineEnv("hespi")
    store.store(SECRET, "sk-abcdef")
    store.store(SECRET, "sk-secret-value", private=True)
    assert "sk-" not in store.path.read_text()
    assert stored(store) == {"variables": {}, "private": [SECRET]}
    assert keyring.get_password("collectra:hespi", SECRET) == "sk-secret-value"


def test_making_a_private_value_plain_removes_it_from_the_keyring(keyring):
    store = PipelineEnv("hespi")
    store.store(SECRET, "first", private=True)
    store.store(SECRET, "second")
    assert stored(store) == {"variables": {SECRET: "second"}, "private": []}
    assert keyring.get_password("collectra:hespi", SECRET) is None


def test_private_values_need_a_keyring(no_keyring):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "kept")
    with pytest.raises(EnvError, match="No secure keyring.*without --private"):
        store.store(SECRET, "sk-secret", private=True)
    assert stored(store) == {"variables": {PLAIN: "kept"}, "private": []}


@pytest.mark.parametrize("name", ["", "1ABC", "MY-KEY", "A B", "A=B"])
def test_names_must_be_environment_variable_names(name):
    with pytest.raises(EnvError, match="not a valid environment variable name"):
        PipelineEnv("hespi").store(name, "value")


def test_an_unreadable_file_is_reported_with_its_path():
    store = PipelineEnv("hespi")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("variables: [unclosed\n")
    with pytest.raises(EnvError, match="hespi.yaml"):
        store.entries()
    store.path.write_text("- a list\n")
    with pytest.raises(EnvError, match="hespi.yaml"):
        store.entries()
    with pytest.raises(EnvError, match="hespi.yaml"):
        store.store(PLAIN, "value")  # rather than replace what is there


def test_values_written_by_hand_are_read_as_text():
    store = PipelineEnv("hespi")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(f"variables:\n  {PLAIN}: 8\n")
    (entry,) = store.entries()
    assert entry.value == "8" and not entry.private


# Listing


def test_entries_are_sorted_and_say_where_each_value_comes_from(keyring, monkeypatch):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "plain")
    store.store(SECRET, "secret", private=True)
    monkeypatch.setenv(OTHER, "from the shell")
    store.store(OTHER, "stored")
    assert [
        (entry.name, entry.value, entry.private, entry.overridden)
        for entry in store.entries()
    ] == [
        (OTHER, "stored", False, True),
        (PLAIN, "plain", False, False),
        (SECRET, "secret", True, False),
    ]


def test_a_private_value_missing_from_the_keyring_is_listed_without_a_value(
    keyring,
):
    store = PipelineEnv("hespi")
    store.store(SECRET, "secret", private=True)
    keyring.delete_password("collectra:hespi", SECRET)
    (entry,) = store.entries()
    assert entry.private and entry.value is None


def test_private_values_are_listed_without_a_value_when_there_is_no_keyring(
    keyring, monkeypatch
):
    store = PipelineEnv("hespi")
    store.store(SECRET, "secret", private=True)
    monkeypatch.setattr(env_module, "keyring_backend", fail.Keyring)
    (entry,) = store.entries()
    assert entry.private and entry.value is None


# Loading


def test_loading_sets_variables_the_environment_does_not_have(keyring, monkeypatch):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "plain")
    store.store(SECRET, "secret", private=True)
    store.store(OTHER, "stored")
    for name in (PLAIN, SECRET):
        monkeypatch.delenv(name)
    monkeypatch.setenv(OTHER, "from the shell")

    loaded = PipelineEnv("hespi")
    assert loaded.load() == [PLAIN, SECRET]
    assert os.environ[PLAIN] == "plain"
    assert os.environ[SECRET] == "secret"
    assert os.environ[OTHER] == "from the shell"
    assert [entry.overridden for entry in loaded.entries()] == [True, False, False]


def test_loading_warns_about_a_private_value_it_cannot_read(keyring, monkeypatch):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "plain")
    store.store(SECRET, "secret", private=True)
    keyring.delete_password("collectra:hespi", SECRET)
    for name in (PLAIN, SECRET):
        monkeypatch.delenv(name)
    with pytest.warns(UserWarning, match=f"{SECRET}.*not in the keyring"):
        assert PipelineEnv("hespi").load() == [PLAIN]
    assert SECRET not in os.environ


def test_loading_warns_about_a_file_it_cannot_read():
    store = PipelineEnv("hespi")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("variables: [unclosed\n")
    with pytest.warns(UserWarning, match="hespi.yaml"):
        assert store.load() == []


def test_changes_reach_the_environment_of_a_running_session(keyring, monkeypatch):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "first")
    assert os.environ[PLAIN] == "first"
    store.store(PLAIN, "second")
    assert os.environ[PLAIN] == "second"
    store.remove(PLAIN)
    assert PLAIN not in os.environ

    # A value the shell gave is left alone.
    monkeypatch.setenv(OTHER, "from the shell")
    store.store(OTHER, "stored")
    store.remove(OTHER)
    assert os.environ[OTHER] == "from the shell"


# Removing


def test_removing_a_variable_removes_it_everywhere(keyring):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "plain")
    store.store(SECRET, "secret", private=True)
    store.remove(SECRET)
    assert stored(store) == {"variables": {PLAIN: "plain"}, "private": []}
    assert keyring.get_password("collectra:hespi", SECRET) is None
    assert SECRET not in os.environ


def test_removing_a_private_value_already_gone_from_the_keyring(keyring):
    store = PipelineEnv("hespi")
    store.store(SECRET, "secret", private=True)
    keyring.delete_password("collectra:hespi", SECRET)
    store.remove(SECRET)
    assert store.entries() == []


def test_removing_an_unknown_variable_is_an_error():
    with pytest.raises(EnvError, match=f"{PLAIN} is not stored for hespi"):
        PipelineEnv("hespi").remove(PLAIN)


def test_removing_everything_deletes_the_file(keyring):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "plain")
    store.store(SECRET, "secret", private=True)
    assert store.remove_all() == [PLAIN, SECRET]
    assert not store.path.exists()
    assert keyring.get_password("collectra:hespi", SECRET) is None
    assert PLAIN not in os.environ and SECRET not in os.environ
    assert store.remove_all() == []


# The pipeline


def test_a_pipeline_loads_its_variables_when_it_is_read(tmp_path, monkeypatch):
    PipelineEnv("hespi").store(PLAIN, "from hespi")
    PipelineEnv("other").store(OTHER, "from other")
    for name in (PLAIN, OTHER):
        monkeypatch.delenv(name)
    pipeline = Pipeline.from_file(write_pipeline(tmp_path / "hespi"))
    assert os.environ[PLAIN] == "from hespi"
    assert OTHER not in os.environ
    assert pipeline.env.name == "hespi"
    assert pipeline.env is pipeline.env


def test_the_variables_are_found_by_name_wherever_the_pipeline_is(
    tmp_path, monkeypatch
):
    Pipeline.from_file(write_pipeline(tmp_path / "a")).env.store(PLAIN, "shared")
    monkeypatch.delenv(PLAIN)
    Pipeline.from_file(write_pipeline(tmp_path / "elsewhere" / "b"))
    assert os.environ[PLAIN] == "shared"


def test_a_pipeline_made_in_code_does_not_load_variables(monkeypatch):
    PipelineEnv("hespi").store(PLAIN, "stored")
    monkeypatch.delenv(PLAIN)
    pipeline = Pipeline("hespi", "hp", "1")
    assert PLAIN not in os.environ
    assert pipeline.env.load() == [PLAIN]


# The command line


@pytest.fixture
def hespi(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")  # so rich does not wrap the table
    pipeline = Pipeline.from_file(write_pipeline(tmp_path / "hespi"))

    def run(*argv):
        return invoke(pipeline, ["env", *argv], name="hespi")

    return run


def test_set_stores_the_value_given(hespi):
    hespi("set", PLAIN, "/data/hf")
    assert PipelineEnv("hespi").entries()[0].value == "/data/hf"


def test_set_asks_for_a_value_without_showing_it(hespi, keyring, monkeypatch):
    prompts = []
    monkeypatch.setattr(
        env_module.getpass, "getpass", lambda prompt: prompts.append(prompt) or "sk-x"
    )
    hespi("set", SECRET, "--private")
    assert prompts == [f"Value for {SECRET}: "]
    assert keyring.get_password("collectra:hespi", SECRET) == "sk-x"


def test_set_reports_errors_without_a_traceback(hespi, no_keyring, capsys):
    with pytest.raises(cappa.Exit) as raised:
        hespi("set", SECRET, "value", "--private")
    assert raised.value.code == 1
    assert "No secure keyring" in capsys.readouterr().err


def test_show_lists_the_variables_and_masks_private_ones(
    hespi, keyring, monkeypatch, capsys
):
    hespi("set", PLAIN, "/data/hf")
    hespi("set", SECRET, "sk-ant-api03-abcdefgh", "--private")
    hespi("set", "COLLECTRA_TEST_PIN", "1234", "--private")
    monkeypatch.setenv(OTHER, "from the shell")
    hespi("set", OTHER, "stored")
    capsys.readouterr()

    hespi("show")
    out = capsys.readouterr().out
    assert str(PipelineEnv("hespi").path) in out
    assert "Environment variables for hespi" in out
    assert "/data/hf" in out
    assert "sk-a…efgh" in out and "abcd" not in out
    assert "••••" in out and "1234" not in out
    rows = {
        line.split()[1]: line for line in out.splitlines() if "COLLECTRA_TEST" in line
    }
    assert "keyring" in rows[SECRET] and "in use" in rows[SECRET]
    assert "file" in rows[PLAIN] and "in use" in rows[PLAIN]
    assert "overridden by the shell" in rows[OTHER]

    hespi("show", "--reveal")
    out = capsys.readouterr().out
    assert "sk-ant-api03-abcdefgh" in out and "1234" in out


def test_show_says_when_a_private_value_cannot_be_read(hespi, keyring, capsys):
    hespi("set", SECRET, "secret", "--private")
    keyring.delete_password("collectra:hespi", SECRET)
    hespi("show")
    assert "not in the keyring" in capsys.readouterr().out


def test_show_with_nothing_stored(hespi, capsys):
    hespi("show")
    assert "No environment variables are stored for hespi" in capsys.readouterr().out


def test_delete_removes_one_variable(hespi, capsys):
    hespi("set", PLAIN, "a")
    hespi("set", OTHER, "b")
    hespi("delete", PLAIN)
    assert [entry.name for entry in PipelineEnv("hespi").entries()] == [OTHER]
    assert f"Deleted {PLAIN}" in capsys.readouterr().out
    with pytest.raises(cappa.Exit):
        hespi("delete", PLAIN)


def test_clear_asks_before_deleting_everything(hespi, monkeypatch, capsys):
    hespi("set", PLAIN, "a")
    hespi("set", OTHER, "b")
    monkeypatch.setattr("builtins.input", lambda *prompt: "n")
    hespi("clear")
    assert len(PipelineEnv("hespi").entries()) == 2
    monkeypatch.setattr("builtins.input", lambda *prompt: "y")
    hespi("clear")
    assert PipelineEnv("hespi").entries() == []
    assert "Deleted 2 variables" in capsys.readouterr().out


def test_clear_with_yes_does_not_ask(hespi, monkeypatch, capsys):
    hespi("set", PLAIN, "a")
    monkeypatch.setattr("builtins.input", pytest.fail)
    hespi("clear", "--yes")
    assert PipelineEnv("hespi").entries() == []
    assert "Deleted 1 variable for hespi" in capsys.readouterr().out
    hespi("clear")
    assert "No environment variables are stored" in capsys.readouterr().out


def test_env_commands_appear_in_help(hespi, capsys):
    with pytest.raises(cappa.HelpExit):
        hespi("--help")
    out = capsys.readouterr().out
    assert all(name in out for name in ("set", "show", "delete", "clear"))


@pytest.mark.parametrize("command", ["show", "clear"])
def test_commands_report_a_file_they_cannot_read(hespi, command, capsys):
    store = PipelineEnv("hespi")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("variables: [unclosed\n")
    with pytest.raises(cappa.Exit):
        hespi(command)
    assert "hespi.yaml" in capsys.readouterr().err


def test_loading_says_which_stored_variables_the_shell_overrides(
    tmp_path, monkeypatch, capsys
):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "stored")
    store.store(OTHER, "stored")
    monkeypatch.delenv(PLAIN)
    monkeypatch.setenv(OTHER, "from the shell")
    capsys.readouterr()
    Pipeline.from_file(write_pipeline(tmp_path / "hespi"))
    out = capsys.readouterr().out
    assert f"{OTHER} is set in the shell" in out
    assert "stored value for hespi is not used" in out
    assert PLAIN not in out


def test_set_says_when_the_shell_overrides_the_new_value(hespi, monkeypatch, capsys):
    monkeypatch.setenv(OTHER, "from the shell")
    hespi("set", OTHER, "stored")
    assert f"{OTHER} is set in the shell" in capsys.readouterr().out
    hespi("set", PLAIN, "stored")
    assert "set in the shell" not in capsys.readouterr().out


def test_show_prints_values_as_given_not_as_rich_markup(hespi, capsys):
    hespi("set", PLAIN, "[bold]x[/bold]")
    capsys.readouterr()
    hespi("show")
    assert "[bold]x[/bold]" in capsys.readouterr().out


def test_whether_a_keyring_is_available(monkeypatch):
    assert PipelineEnv.keyring_available()
    monkeypatch.setattr(env_module, "keyring_backend", fail.Keyring)
    assert not PipelineEnv.keyring_available()


def test_an_entry_is_shown_masked_only_when_it_is_private():
    entry = env_module.EnvEntry
    assert entry("A", "/data/hf", False, False).display() == "/data/hf"
    assert entry("A", "sk-ant-api03-abcdefgh", True, False).display() == "sk-a…efgh"
    assert entry("A", "1234", True, False).display() == "••••"
    assert entry("A", "1234", True, False).display(reveal=True) == "1234"
    assert entry("A", None, True, False).display() == ""


# Overriding the shell


def test_an_override_variable_replaces_the_shells_value_when_loaded(
    tmp_path, monkeypatch, capsys
):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "stored", override=True)
    store.store(OTHER, "stored")
    assert stored(store) == {
        "variables": {PLAIN: "stored", OTHER: "stored"},
        "private": [],
        "override": [PLAIN],
    }
    monkeypatch.setenv(PLAIN, "from the shell")
    monkeypatch.setenv(OTHER, "from the shell")
    capsys.readouterr()

    pipeline = Pipeline.from_file(write_pipeline(tmp_path / "hespi"))
    assert os.environ[PLAIN] == "stored"
    assert os.environ[OTHER] == "from the shell"
    out = capsys.readouterr().out
    assert f"{PLAIN} for hespi replaces the value set in the shell" in out
    assert f"{OTHER} is set in the shell" in out
    entries = {entry.name: entry for entry in pipeline.env.entries()}
    assert entries[PLAIN].override and entries[PLAIN].replaces_shell
    assert not entries[PLAIN].overridden
    assert entries[OTHER].overridden and not entries[OTHER].replaces_shell


def test_an_override_variable_with_nothing_to_replace_is_simply_in_use():
    store = PipelineEnv("hespi")
    store.store(PLAIN, "stored", override=True)
    (entry,) = store.entries()
    assert entry.override and not entry.replaces_shell and not entry.overridden


def test_the_shells_value_comes_back_when_the_override_goes(monkeypatch):
    monkeypatch.setenv(PLAIN, "from the shell")
    store = PipelineEnv("hespi")
    store.store(PLAIN, "stored", override=True)
    assert os.environ[PLAIN] == "stored"
    store.store(PLAIN, "stored")  # set again without override
    assert os.environ[PLAIN] == "from the shell"
    store.store(PLAIN, "stored", override=True)
    store.remove(PLAIN)
    assert os.environ[PLAIN] == "from the shell"
    store.store(PLAIN, "stored", override=True)
    store.remove_all()
    assert os.environ[PLAIN] == "from the shell"


def test_override_can_be_switched_without_the_value(keyring, monkeypatch):
    monkeypatch.setenv(SECRET, "from the shell")
    store = PipelineEnv("hespi")
    store.store(SECRET, "secret", private=True)
    store.set_override(SECRET, True)
    assert os.environ[SECRET] == "secret"
    assert stored(store)["override"] == [SECRET]
    assert keyring.get_password("collectra:hespi", SECRET) == "secret"
    store.set_override(SECRET, False)
    assert os.environ[SECRET] == "from the shell"
    assert "override" not in stored(store)
    with pytest.raises(EnvError, match="not stored"):
        store.set_override(PLAIN, True)


def test_override_cannot_be_switched_on_a_value_the_keyring_has_lost(keyring):
    store = PipelineEnv("hespi")
    store.store(SECRET, "secret", private=True)
    keyring.delete_password("collectra:hespi", SECRET)
    store.set_override(SECRET, True)  # stored, applied when the value is back
    assert stored(store)["override"] == [SECRET]


def test_removing_an_override_variable_forgets_the_override(monkeypatch):
    store = PipelineEnv("hespi")
    store.store(PLAIN, "stored", override=True)
    store.store(OTHER, "stored")
    store.remove(PLAIN)
    assert "override" not in stored(store)


def test_set_override_on_the_command_line(hespi, monkeypatch, capsys):
    monkeypatch.setenv(PLAIN, "from the shell")
    hespi("set", PLAIN, "stored", "--override")
    out = capsys.readouterr().out
    assert f"{PLAIN} for hespi replaces the value set in the shell" in out
    assert os.environ[PLAIN] == "stored"
    hespi("override", PLAIN, "--off")
    assert f"{PLAIN} is set in the shell" in capsys.readouterr().out
    assert os.environ[PLAIN] == "from the shell"
    hespi("override", PLAIN)
    assert os.environ[PLAIN] == "stored"
    capsys.readouterr()
    hespi("show")
    row = next(line for line in capsys.readouterr().out.splitlines() if PLAIN in line)
    assert "overrides the shell" in row
    with pytest.raises(cappa.Exit):
        hespi("override", OTHER)


def test_override_with_nothing_to_replace_says_so_on_the_command_line(hespi, capsys):
    hespi("set", PLAIN, "stored")
    capsys.readouterr()
    hespi("override", PLAIN)
    assert f"{PLAIN} for hespi now overrides the shell" in capsys.readouterr().out
    hespi("override", PLAIN, "--off")
    assert f"{PLAIN} for hespi no longer overrides the shell" in capsys.readouterr().out


def test_show_marks_an_override_with_nothing_to_replace(hespi, capsys):
    hespi("set", PLAIN, "stored", "--override")
    capsys.readouterr()
    hespi("show")
    row = next(line for line in capsys.readouterr().out.splitlines() if PLAIN in line)
    assert "in use (override on)" in row
