#!/usr/bin/env bash
# Download the IBM Telco Customer Churn CSV and verify it (checksum and row count).
# The expected checksum is read from data/raw/SOURCE.md, so there is one place to change it.
#
#   scripts/download_data.sh            download if missing or wrong, then verify
#
# Environment overrides (used by the tests): DATA_URL, TARGET, SOURCE_MD.
set -euo pipefail

URL="${DATA_URL:-https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv}"
TARGET="${TARGET:-data/raw/telco_churn.csv}"
SOURCE_MD="${SOURCE_MD:-data/raw/SOURCE.md}"
EXPECTED_LINES=7044   # 7,043 customers plus the header line

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$SOURCE_MD" ]] || die "cannot find $SOURCE_MD (run this from the repository root)"
EXPECTED_SHA="$(grep -Eio 'sha-?256:? *\|? *[0-9a-f]{64}|sha-?256[^0-9a-f]*[0-9a-f]{64}' "$SOURCE_MD" \
    | grep -Eo '[0-9a-f]{64}' | head -n 1 || true)"
[[ -n "$EXPECTED_SHA" ]] || die "no SHA-256 found in $SOURCE_MD"

check() {
    local file="$1" sha lines
    sha="$(sha256sum "$file" | cut -d' ' -f1)"
    lines="$(wc -l < "$file" | tr -d ' ')"
    [[ "$sha" == "$EXPECTED_SHA" ]] || { echo "checksum is $sha, expected $EXPECTED_SHA"; return 1; }
    [[ "$lines" == "$EXPECTED_LINES" ]] || { echo "file has $lines lines, expected $EXPECTED_LINES"; return 1; }
}

if [[ -f "$TARGET" ]] && check "$TARGET" >/dev/null 2>&1; then
    echo "already present and verified: $TARGET"
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
echo "downloading $URL"
curl -fsSL --retry 3 --retry-delay 2 -o "$TMP" "$URL" || die "download failed"
check "$TMP" || die "the downloaded file does not match $SOURCE_MD; it was NOT saved"
mv "$TMP" "$TARGET"
trap - EXIT
echo "saved and verified: $TARGET ($(wc -c < "$TARGET" | tr -d ' ') bytes)"
