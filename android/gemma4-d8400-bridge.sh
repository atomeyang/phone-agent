#!/system/bin/sh
# Resident bridge for the on-device agent.
#
# The app drops request documents into $ROOT/agent/request.json and tails the
# append-only event log the runner writes there; the legacy single-shot demo
# path (app-request + app-run.log) is still served by the same process, so the
# screen-on/off benchmark keeps working without a second model instance.

ROOT=/data/local/tmp/gemma4-d8400
AGENT="$ROOT/agent"
REQUEST="$AGENT/request.json"
EVENTS="$AGENT/events.jsonl"
READY="$AGENT/service-ready"
START="$ROOT/app-service-start"
PID_FILE="$AGENT/runner.pid"
BRIDGE_LOG="$AGENT/bridge.log"
SERVICE_LOG="$AGENT/service.log"
# Legacy demo/benchmark files, kept for compatibility.
LEGACY_READY="$ROOT/app-service-ready"
LEGACY_LOG="$ROOT/app-run.log"
RUNNER=gemma4-d8400-runner

stop_runner() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        rm -f "$PID_FILE"
    fi
    for pid in $(pidof "$RUNNER" 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null
    done
}

runner_is_alive() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start_runner() {
    if runner_is_alive && [ -s "$READY" ]; then
        return
    fi
    stop_runner
    mkdir -p "$AGENT"
    chmod 777 "$AGENT" 2>/dev/null
    rm -f "$READY" "$START" "$REQUEST" "$EVENTS" "$AGENT/cancel"
    : > "$SERVICE_LOG"
    : > "$EVENTS"
    : > "$LEGACY_LOG"
    chmod 666 "$EVENTS" "$SERVICE_LOG" "$LEGACY_LOG"

    export GEMMA4_PLATFORM="$(cat "$ROOT/.platform" 2>/dev/null)"
    [ -n "$GEMMA4_PLATFORM" ] || GEMMA4_PLATFORM=mt6899
    ln -sf libneuron_runtime.so.9.3.1 "$ROOT/libneuron_runtime.so.9"
    ln -sf libneuron_adapter.so.9.3.1 "$ROOT/libneuron_adapter.so.9"

    "$ROOT/$RUNNER" --agent-service \
        "$ROOT/config.json" \
        "$ROOT/models/visual_front_w8a16_${GEMMA4_PLATFORM}.dla" \
        "$ROOT" \
        "$AGENT" \
        512 > "$SERVICE_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    # Wait for the runner to publish its ready file so the app never talks to a
    # half-loaded engine.
    count=0
    while [ ! -s "$READY" ] && [ "$count" -lt 1200 ]; do
        if ! runner_is_alive; then
            break
        fi
        sleep 0.5
        count=$((count + 1))
    done
    if [ -s "$READY" ]; then
        # The legacy path only checks that this file exists.
        cp -f "$READY" "$LEGACY_READY" 2>/dev/null
    fi
}

cd "$ROOT" || exit 1
export LD_LIBRARY_PATH="$ROOT"
chmod 777 "$ROOT" 2>/dev/null

while true; do
    if [ -f "$START" ]; then
        rm -f "$START"
        start_runner
    fi
    if ! runner_is_alive; then
        rm -f "$PID_FILE" "$READY" "$LEGACY_READY"
    fi
    sleep 0.2
done
