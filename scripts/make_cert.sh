#!/usr/bin/env bash
# Create a self-signed TLS certificate for nginx (valid one year). Does nothing if one exists.
# The browser will warn once because no authority signed it; that is expected for a local demo.
set -euo pipefail

DIR="${TLS_DIR:-/srv/churnsense/secrets/tls}"
mkdir -p "$DIR"
if [[ -s "$DIR/tls.crt" && -s "$DIR/tls.key" ]]; then
    echo "certificate already exists: $DIR/tls.crt"
    exit 0
fi
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout "$DIR/tls.key" -out "$DIR/tls.crt" \
    -subj "/O=ChurnSense/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:churnsense.local,IP:127.0.0.1" 2>/dev/null
chmod 644 "$DIR/tls.crt"
chmod 644 "$DIR/tls.key"
echo "created $DIR/tls.crt (self-signed, 365 days)"
