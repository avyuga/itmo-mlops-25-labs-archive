#!/bin/bash

# Run ruff formatter
echo "Running ruff formatter..."
poetry run ruff format mlops

# # Run ruff checker with auto-fix
# echo "Running ruff checker with auto-fix..."
# poetry run ruff check --fix mlops

echo "Done!"
