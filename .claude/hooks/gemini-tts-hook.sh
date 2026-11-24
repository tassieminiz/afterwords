#!/usr/bin/env bash
# Queue a Gemini CLI assistant response for Afterwords TTS.
#
# Gemini fires AfterAgent (analogue of Claude Stop) with a JSON payload that
# differs from Claude's: it sends `.prompt_response` rather than
# `.last_assistant_message`, and `.hook_event_name` instead of agent_type. This
# adapter normalises and re-emits in the format the existing Claude TTS hook +
# worker already understand, then reuses the same /tmp/claude-tts-queue/ dir and
# /tmp/claude-tts-worker.pid so a single worker drains both sources.
#
# Wire-up: drop this in ~/.claude/hooks/ (or wherever your Claude tts-hook.sh
# lives) and reference it from ~/.gemini/settings.json. See README "With Gemini
# CLI" for the JSON snippet.
set -uo pipefail

QUEUEDIR="/tmp/claude-tts-queue"
WORKER_PID="/tmp/claude-tts-worker.pid"
WORKER="$HOME/.claude/hooks/tts-worker.sh"
STRIP_MARKDOWN="$HOME/.claude/hooks/strip-markdown.py"

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
' "$QUEUEDIR" "$PWD" "$AGENT" "$TEXT" || exit 1

if [ -f "$WORKER_PID" ]; then
    EXISTING=$(cat "$WORKER_PID" 2>/dev/null)
    if [ -n "$EXISTING" ] && kill -0 "$EXISTING" 2>/dev/null; then
        exit 0
    fi
    rm -f "$WORKER_PID"
fi

nohup bash "$WORKER" >/dev/null 2>&1 &
