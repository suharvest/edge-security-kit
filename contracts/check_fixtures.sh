#!/bin/sh
# Conformance gate (HUB_SPEC §10): every fixture must satisfy the schema.
#
# Dependency-free on purpose — validate_payload.py uses only the standard
# library, so this runs on a device, in CI, or on a laptop with no venv.
#
#   sh contracts/check_fixtures.sh
#
# Exits non-zero on the first failure, after reporting every file.
set -u

here=$(dirname "$0")
python=${PYTHON:-python3}
validator="$here/validate_payload.py"
fixtures="$here/fixtures"

if [ ! -d "$fixtures" ]; then
    echo "no fixture directory at $fixtures" >&2
    exit 1
fi

count=0
failed=0
for path in "$fixtures"/*.json; do
    [ -e "$path" ] || continue
    count=$((count + 1))
    if out=$("$python" "$validator" "$path" 2>&1); then
        printf 'PASS %-34s %s\n' "$(basename "$path")" "$out"
    else
        printf 'FAIL %-34s %s\n' "$(basename "$path")" "$out" >&2
        failed=$((failed + 1))
    fi
done

if [ "$count" -eq 0 ]; then
    echo "no fixtures found in $fixtures" >&2
    exit 1
fi

echo "---"
echo "$count fixture(s), $failed failure(s)"
[ "$failed" -eq 0 ] || exit 1
