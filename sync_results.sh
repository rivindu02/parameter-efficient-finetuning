#!/usr/bin/env bash
# Download Colab results from Google Drive (MyDrive/EN4554) into ./colab_results.
# One-way: Drive -> local. Never deletes or changes anything on Drive.
#
#   ./sync_results.sh          # metrics, traces, logs, plots (skips model weights)
#   ./sync_results.sh --all    # everything, including adapters / full-FT weights
#
# Needs an rclone remote named "gdrive" (set up once with: rclone config).
set -euo pipefail
cd "$(dirname "$0")"
REMOTE="${RCLONE_REMOTE:-gdrive}:EN4554"
DEST="colab_results"

FILTER=(--exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.pth")
[[ "${1:-}" == "--all" ]] && FILTER=()

rclone copy "$REMOTE" "$DEST" "${FILTER[@]}" --update --fast-list --stats-one-line --stats 0 -v 2>&1 \
  | grep -E "Copied|ERROR|Failed" || true
echo "synced $REMOTE -> $DEST"
find "$DEST" -type f -newermt "-10 minutes" 2>/dev/null | sed 's/^/  new: /' | head -40
