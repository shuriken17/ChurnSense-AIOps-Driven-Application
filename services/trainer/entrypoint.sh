#!/bin/sh
# The container runs as the host user id, which has no home folder inside the image.
# Create the folder named by $HOME (compose sets it to /tmp/home), then run the command.
set -e
mkdir -p "${HOME:-/tmp/home}"
exec "$@"
