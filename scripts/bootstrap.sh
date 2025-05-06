# scripts/bootstrap.sh
#!/bin/zsh
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
poetry install
