#!/usr/bin/env bash
# Queue a Gemini CLI assistant response for Afterwords TTS.
#
# Gemini fires AfterAgent (analogue of Claude Stop) with a JSON payload that
# differs from Claude's: it sends `.prompt_response` rather than
# `.last_assistant_message`, and `.hook_event_name` instead of agent_type. This
# adapter normalises and re-emits in the format the existing Claude TTS hook +
# worker already understand, then reuses the same private queue
# (~/.claude/afterwords-tts/queue/) and worker pid file so a single worker
# drains both sources.
#
# Wire-up: drop this in ~/.claude/hooks/ (or wherever your Claude tts-hook.sh
# lives) and reference it from ~/.gemini/settings.json. See README "With Gemini
# CLI" for the JSON snippet.
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

# Gemini sends prompt_response. Fall back to last_assistant_message in case the
# Gemini schema converges with Claude's in a future release.
TEXT=$(printf '%s' "$INPUT" | jq -r '.prompt_response // .last_assistant_message // empty' 2>/dev/null \
    | python3 "$STRIP_MARKDOWN" 2>/dev/null)
[ -z "$TEXT" ] && exit 0

# Gemini hook payload doesn't carry an agent_type; subagent semantics differ
# enough that we don't try to map them. AGENT is set to gemini to resolve
# voice via .afterwords mapping.
AGENT="gemini"

python3 "$QUEUE_HELPER" enqueue "$STATE_DIR" queue "$PWD" "$AGENT" "$TEXT" || exit 1

if [ -f "$WORKER_PID" ]; then
    EXISTING=$(cat "$WORKER_PID" 2>/dev/null)
    if [ -n "$EXISTING" ] && kill -0 "$EXISTING" 2>/dev/null; then
        exit 0
    fi
    rm -f "$WORKER_PID"
fi

nohup bash "$WORKER" >/dev/null 2>&1 &
