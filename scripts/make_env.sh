#!/usr/bin/env bash
# Create .env from .env.example: your user and group id, and a random value for every secret.
# Does nothing if .env already exists. The file is readable by you only (mode 600).
set -euo pipefail

[[ -f .env.example ]] || { echo "ERROR: run this from the repository root" >&2; exit 1; }
if [[ -f .env ]]; then
    echo ".env already exists; leaving it alone"
    exit 0
fi

cp .env.example .env
sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env
while grep -q 'change-me' .env; do
    secret="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    sed -i "0,/change-me/s//${secret}/" .env
done
chmod 600 .env
echo "created .env for user $(id -un) ($(id -u):$(id -g)) with random secrets (mode 600)"
