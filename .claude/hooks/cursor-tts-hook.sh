#!/usr/bin/env bash
# Queue a Cursor IDE afterAgentResponse for Afterwords TTS.
#
# Cursor 1.7+ fires afterAgentResponse when the agent completes a response,
# passing the full assistant text in the `text` field. This adapter extracts
# it, strips markdown, and appends it to the common private queue
# (~/.claude/afterwords-tts/queue/) that the existing tts-worker.sh drains. The
# worker and all shared helpers (strip-markdown.py, chunk-text.py,
# tts-queue.py) are read from ~/.claude/hooks/.
#
# Wire-up: copy this file and tts-queue.py to ~/.claude/hooks/ and configure
# ~/.cursor/hooks.json:
#   {
#     "version": 1,
#     "hooks": {
#       "afterAgentResponse": [
#         {
#           "command": "bash ~/.claude/hooks/cursor-tts-hook.sh",
#           "type": "command",
#           "timeout": 10,
#           "failClosed": false
#         }
#       ]
#     }
#   }
#
# Voice per project: add a .afterwords file at the repo root, e.g.:
#   default: rimmer
#   cursor: lister
#
# Run `bash setup.sh` to install automatically when Cursor is detected.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$HOME/.claude/afterwords-tts"
WORKER_PID="${STATE_DIR}/worker.pid"
WORKER="$HOME/.claude/hooks/tts-worker.sh"
STRIP_MARKDOWN="$HOME/.claude/hooks/strip-markdown.py"
QUEUE_HELPER="${SCRIPT_DIR}/tts-queue.py"
[ -f "$QUEUE_HELPER" ] || QUEUE_HELPER="$HOME/.claude/hooks/tts-queue.py"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

TEXT=$(printf '%s' "$INPUT" | jq -r '.text // empty' 2>/dev/null \
    | python3 "$STRIP_MARKDOWN" 2>/dev/null)
[ -z "$TEXT" ] && exit 0

# workspace_roots[0] is the project directory; fall back to $PWD.
PROJECT_DIR=$(printf '%s' "$INPUT" | jq -r '.workspace_roots[0] // empty' 2>/dev/null)
PROJECT_DIR="${PROJECT_DIR:-$PWD}"

AGENT="cursor"

python3 "$QUEUE_HELPER" enqueue "$STATE_DIR" queue "$PROJECT_DIR" "$AGENT" "$TEXT" || exit 1

if [ -f "$WORKER_PID" ]; then
    EXISTING=$(cat "$WORKER_PID" 2>/dev/null)
    if [ -n "$EXISTING" ] && kill -0 "$EXISTING" 2>/dev/null; then
        exit 0
    fi
    rm -f "$WORKER_PID"
fi

nohup bash "$WORKER" >/dev/null 2>&1 &
