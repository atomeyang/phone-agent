#!/system/bin/sh

ROOT=/data/local/tmp/minicpmv46-neuropilot-letterbox
REQUEST="$ROOT/app-request"
STOP="$ROOT/app-stop"
LOG="$ROOT/app-run.log"
PROMPT="$ROOT/app-prompt.txt"
PID_FILE="$ROOT/app-runner.pid"
START="$ROOT/app-service-start"
READY="$ROOT/app-service-ready"
SERVICE_LOG="$ROOT/app-service.log"
RUNNER=minicpmv46-letterbox-runner

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
    if runner_is_alive; then
        touch "$READY"
        return
    fi

    stop_runner
    rm -f "$READY" "$REQUEST" "$STOP"
    : > "$LOG"
    : > "$SERVICE_LOG"
    chmod 666 "$LOG" "$SERVICE_LOG"
    ln -sf libneuron_runtime.so.9.3.1 "$ROOT/libneuron_runtime.so.9"
    ln -sf libneuron_adapter.so.9.3.1 "$ROOT/libneuron_adapter.so.9"
    "$ROOT/$RUNNER" --service \
        "$ROOT/config.json" \
        "$ROOT/models" \
        "$ROOT/app-input-fp32.bin" \
        "$PROMPT" \
        "$REQUEST" "$STOP" "$LOG" "$READY" 1024 > "$SERVICE_LOG" 2>&1 &
    echo $! > "$PID_FILE"
}

cd "$ROOT" || exit 1
export LD_LIBRARY_PATH="$ROOT"
chmod 777 "$ROOT"
touch "$LOG"
chmod 666 "$LOG"

while true; do
    if [ -f "$START" ]; then
        rm -f "$START"
        start_runner
    fi

    if ! runner_is_alive; then
        rm -f "$PID_FILE" "$READY"
    fi

    sleep 0.1
done
