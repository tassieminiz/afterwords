#!/usr/bin/env bash
# Queue an Antigravity CLI (agy) assistant response for Afterwords TTS.
#
# Antigravity fires Stop with a JSON payload that contains transcriptPath and
# workspacePaths. This hook extracts the final answer from the transcript file,
# sets AGENT="agy", and appends it to the common Claude/Gemini/agy TTS queue.
#
set -uo pipefail

QUEUEDIR="/tmp/claude-tts-queue"
WORKER_PID="/tmp/claude-tts-worker.pid"
WORKER="$HOME/.claude/hooks/tts-worker.sh"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

TRANSCRIPT_PATH=$(printf '%s' "$INPUT" | jq -r '.transcriptPath // empty' 2>/dev/null)
[ -z "$TRANSCRIPT_PATH" ] && exit 0

PROJECT_DIR=$(printf '%s' "$INPUT" | jq -r '.workspacePaths[0] // empty' 2>/dev/null)
PROJECT_DIR="${PROJECT_DIR:-$PWD}"

# Resolve the absolute path of this script's directory to find the python helper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEXT=$(python3 "${SCRIPT_DIR}/agy-session-hook.py" "$TRANSCRIPT_PATH" 2>/dev/null \
    | python3 "${SCRIPT_DIR}/strip-markdown.py" 2>/dev/null)
[ -z "$TEXT" ] && exit 0

AGENT="agy"

# /tmp is shared with other local users: open the queue dir once without
# following symlinks, verify ownership on that descriptor, and create the item
# relative to it so the path is never resolved again after the check.
python3 -c '
import json, os, secrets, sys, time
qdir, project_dir, agent, text = sys.argv[1:5]
try:
    try:
        os.mkdir(qdir, 0o700)
    except FileExistsError:
        pass
    fd = os.open(qdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    if os.fstat(fd).st_uid != os.getuid():
        sys.exit("afterwords: %s is not a directory we own — refusing to use it" % qdir)
    os.fchmod(fd, 0o700)
    name = "%d-%s.json" % (time.time(), secrets.token_hex(4))
    out = os.open(name + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(out, "w") as f:
        json.dump({"project_dir": project_dir, "agent": agent, "text": text}, f)
        f.write("\n")
    os.rename(name + ".tmp", name, src_dir_fd=fd, dst_dir_fd=fd)
except OSError as e:
    sys.exit("afterwords: cannot use %s (%s) — refusing to queue" % (qdir, e.strerror))
' "$QUEUEDIR" "$PROJECT_DIR" "$AGENT" "$TEXT" || exit 1

if [ -f "$WORKER_PID" ]; then
    EXISTING=$(cat "$WORKER_PID" 2>/dev/null)
    if [ -n "$EXISTING" ] && kill -0 "$EXISTING" 2>/dev/null; then
        exit 0
    fi
    rm -f "$WORKER_PID"
fi

nohup bash "$WORKER" >/dev/null 2>&1 &
