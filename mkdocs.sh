#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

# Build independently of the active Python environment and ML dependencies.
exec uv tool run --from 'sphinx>=5.0.0' \
    --with 'sphinx-rtd-theme>=1.0.0' \
    --with 'sphinx-copybutton>=0.4.0' \
    sphinx-build -E -b html "$@" docs docs/_build/html
