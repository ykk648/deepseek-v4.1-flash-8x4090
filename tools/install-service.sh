#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_NAME=deepseek-v4.1-flash.service
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

mkdir -p "$UNIT_DIR"
sed "s|@PROJECT_DIR@|$ROOT|g" "$ROOT/$UNIT_NAME" >"$UNIT_DIR/$UNIT_NAME"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"

echo "Installed $UNIT_DIR/$UNIT_NAME"
echo "Start with: systemctl --user start $UNIT_NAME"
