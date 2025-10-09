#!/bin/bash
coverage erase
coverage run --source=collectra -m pytest
coverage report
rm -r htmlcov
coverage html
open htmlcov/index.html