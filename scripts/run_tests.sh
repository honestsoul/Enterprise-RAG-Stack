#!/usr/bin/env bash
# Run the project’s test suite using pytest.  If no tests are found,
# pytest will exit with code 5 which we treat as success.

set -euo pipefail

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest is not installed. Installing..."
  pip install pytest
fi

pytest || test $? -eq 5
