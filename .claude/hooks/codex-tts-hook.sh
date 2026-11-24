#!/usr/bin/env bash
# Queue a Codex final answer for Afterwords TTS.
#
# Per-session isolation: queue and worker-pid paths include CODEX_THREAD_ID so
# multiple parallel Codex sessions each have their own queue and worker.
#
# Queue format: one JSON file per item in $STATE_DIR/queue-<thread>/. The state
# dir is private to this user (mode 0700); tts-queue.py refuses a directory it
# does not own and writes the item through the directory it checked.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION_ID="${CODEX_THREAD_ID:-global}"
STATE_DIR="$HOME/.codex/afterwords-tts"
QUEUE_NAME="queue-${SESSION_ID}"
WORKER_PID="${STATE_DIR}/worker-${SESSION_ID}.pid"
WORKER="${SCRIPT_DIR}/codex-tts-worker.sh"
QUEUE_HELPER="${SCRIPT_DIR}/tts-queue.py"

INPUT=$(cat)
TEXT=$(printf '%s' "$INPUT" | python3 "${REPO_DIR}/codex_session_hook.py" --strip-markdown 2>/dev/null)
[ -z "$TEXT" ] && exit 0

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
AGENT="${AGENT_TYPE:-}"

python3 "$QUEUE_HELPER" enqueue "$STATE_DIR" "$QUEUE_NAME" "$PROJECT_DIR" "$AGENT" "$TEXT" || exit 1

if [ -f "$WORKER_PID" ]; then
    EXISTING=$(cat "$WORKER_PID" 2>/dev/null)
    if [ -n "$EXISTING" ] && kill -0 "$EXISTING" 2>/dev/null; then
        exit 0
    fi
    rm -f "$WORKER_PID"
fi

nohup bash "$WORKER" >/dev/null 2>&1 &
