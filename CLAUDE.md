# CLAUDE.md

## Development preferences

- **Test-first:** Always write the tests first, then write the code that makes them pass.
- **Coverage:** Keep test coverage at 100%. Measure with `coverage run -m pytest ; coverage report`.
- **Formatting:** Format all Python code with [black](https://black.readthedocs.io/).

## Terminal output

- **Use [rich](https://rich.readthedocs.io/) for anything printed to stdout.** The user likes clean, informative, coloured output: `rich.table.Table` for lists of things, short status lines with a coloured marker (green `✓` for done, yellow `!` for a warning, red for errors), and bold for the names of things. Keep messages short and in plain words.
- Print user-supplied values (paths, names, values) with `rich.text.Text` or `rich.markup.escape`, never as markup, so `[brackets]` in them print as typed.
- Long single lines such as file paths should not wrap: print them with `soft_wrap=True`.
