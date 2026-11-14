#!/bin/sh
# Runs when the frontend container starts (nginx image entry-point hook): writes config.js so the
# page knows where its API is. API_BASE is a path on the same host (default) or a full address.
set -eu

API_BASE="${API_BASE:-/api/v1}"
HTML_DIR="${HTML_DIR:-/usr/share/nginx/html}"

case "$API_BASE" in
    *[!A-Za-z0-9_:/.-]*)
        echo "ERROR: API_BASE contains characters that are not allowed: $API_BASE" >&2
        exit 1
        ;;
esac

printf 'window.CHURNSENSE_CONFIG = { apiBase: "%s" };\n' "$API_BASE" > "$HTML_DIR/config.js"
echo "frontend configured: apiBase=$API_BASE"
