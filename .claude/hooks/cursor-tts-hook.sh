#!/usr/bin/env bash
# Queue a Cursor IDE afterAgentResponse for Afterwords TTS.
#
# Cursor 1.7+ fires afterAgentResponse when the agent completes a response,
# passing the full assistant text in the `text` field. This adapter extracts
# it, strips markdown, and appends it to the common /tmp/claude-tts-queue/
# dir that the existing tts-worker.sh drains. The worker and all shared
# helpers (strip-markdown.py, chunk-text.py) are read from ~/.claude/hooks/.
#
# Wire-up: copy this file to ~/.claude/hooks/ and configure ~/.cursor/hooks.json:
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

QUEUEDIR="/tmp/claude-tts-queue"
WORKER_PID="/tmp/claude-tts-worker.pid"
WORKER="$HOME/.claude/hooks/tts-worker.sh"
STRIP_MARKDOWN="$HOME/.claude/hooks/strip-markdown.py"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

TEXT=$(printf '%s' "$INPUT" | jq -r '.text // empty' 2>/dev/null \
    | python3 "$STRIP_MARKDOWN" 2>/dev/null)
[ -z "$TEXT" ] && exit 0

# workspace_roots[0] is the project directory; fall back to $PWD.
PROJECT_DIR=$(printf '%s' "$INPUT" | jq -r '.workspace_roots[0] // empty' 2>/dev/null)
PROJECT_DIR="${PROJECT_DIR:-$PWD}"

AGENT="cursor"

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
