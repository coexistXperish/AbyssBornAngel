#!/usr/bin/env bash
set -e

REPO="https://github.com/coexistXperish/AbyssBornAngel.git"
HERMES_DIR="$HOME/.hermes/hermes-agent"

cd "$HERMES_DIR"
git remote set-url origin "$REPO"
git pull origin main
bash scripts/install.sh
grep -q "hermes_plus" ~/.hermes/config.yaml || echo "  hermes_plus: true" >> ~/.hermes/config.yaml
echo "HERMES++ active: $(grep hermes_plus ~/.hermes/config.yaml)"
