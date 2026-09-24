#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE="com.example.minicpmv46.neuropilot.letterbox"
ACTIVITY="$PACKAGE/com.example.minicpmv46.neuropilot.MainActivity"
REMOTE_ROOT="/data/local/tmp/minicpmv46-neuropilot-letterbox"
RUNNER="minicpmv46-letterbox-runner"
BRIDGE="minicpmv46-letterbox-bridge.sh"
EXPECTED_MIB=1473
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if (( $# != 0 )); then
    fail "This script takes no arguments."
fi

find_windows_adb() {
    local candidate windows_path
    local -a candidates=()
    [[ -n "${ANDROID_SDK_ROOT:-}" ]] && \
        candidates+=("$ANDROID_SDK_ROOT/platform-tools/adb.exe")
    [[ -n "${ANDROID_HOME:-}" ]] && \
        candidates+=("$ANDROID_HOME/platform-tools/adb.exe")
    command -v adb.exe >/dev/null 2>&1 && candidates+=("$(command -v adb.exe)")

    if command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
        windows_path="$(cmd.exe /d /c 'echo %LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe' \
            2>/dev/null | tr -d '\r')"
        if [[ -n "$windows_path" ]]; then
            candidate="$(wslpath -u "$windows_path" 2>/dev/null || true)"
            [[ -n "$candidate" ]] && candidates+=("$candidate")
        fi
    fi

    for candidate in /mnt/?/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe; do
        [[ -e "$candidate" ]] && candidates+=("$candidate")
    done
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" && -x "$candidate" && "${candidate,,}" == *.exe ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

ADB="$(find_windows_adb)" || \
    fail "Windows Android SDK adb.exe was not found. Install Android SDK Platform-Tools."
APK=""
for candidate in \
    "$SCRIPT_DIR/PhoneVLM-Letterbox-v1.2.1.apk" \
    "$SCRIPT_DIR/MiniCPM-V-4.6-NPU-MNN-v1.2.0-letterbox.apk" \
    "$SCRIPT_DIR/app-letterbox-debug.apk" \
    "$SCRIPT_DIR/app/build/outputs/apk/letterbox/debug/app-letterbox-debug.apk"; do
    if [[ -f "$candidate" ]]; then
        APK="$candidate"
        break
    fi
done
[[ -n "$APK" ]] || fail "The PhoneVLM Letterbox APK was not found beside this script."
ADB_APK="$APK"
if command -v wslpath >/dev/null 2>&1; then
    ADB_APK="$(wslpath -w "$APK")"
fi

mapfile -t DEVICES < <("$ADB" devices | tr -d '\r' \
    | awk 'NR > 1 && $2 == "device" { print $1 }')
(( ${#DEVICES[@]} == 1 )) || \
    fail "Connect and authorize exactly one Android device."

"$ADB" wait-for-device >/dev/null
MODEL="$("$ADB" shell getprop ro.product.model | tr -d '\r')"
PLATFORM="$("$ADB" shell getprop ro.board.platform | tr -d '\r' \
    | tr '[:upper:]' '[:lower:]')"
[[ "$PLATFORM" == "mt6899" ]] || \
    fail "Unsupported platform '$PLATFORM' on '$MODEL'; this build requires MT6899."

ADB_UID="$("$ADB" shell id -u | tr -d '\r')"
if [[ "$ADB_UID" != "0" ]]; then
    printf 'Requesting root ADB access...\n'
    "$ADB" root >/dev/null 2>&1 || true
    "$ADB" wait-for-device >/dev/null
    ADB_UID="$("$ADB" shell id -u | tr -d '\r')"
fi
[[ "$ADB_UID" == "0" ]] || \
    fail "This phone does not provide the root ADB identity required by NeuroPilot."

printf 'ADB: %s\nDevice: %s (%s)\nInstalling Letterbox APK...\n' \
    "$ADB" "$MODEL" "$PLATFORM"
"$ADB" install -r -d "$ADB_APK"
"$ADB" shell "mkdir -p '$REMOTE_ROOT' && chmod -R 777 '$REMOTE_ROOT'" >/dev/null
"$ADB" shell "
    for pid in \$(pidof '$RUNNER' 2>/dev/null); do
        kill -9 \"\$pid\" 2>/dev/null || true
    done
    for pid in \$(pidof sh 2>/dev/null); do
        script=\$(tr '\\0' '\\n' < \"/proc/\$pid/cmdline\" 2>/dev/null | sed -n '2p')
        case \"\$script\" in
            ./$BRIDGE|*/$BRIDGE) kill -9 \"\$pid\" 2>/dev/null || true ;;
        esac
    done
    for file in '$REMOTE_ROOT/bridge.pid' '$REMOTE_ROOT/app-runner.pid'; do
        if [ -f \"\$file\" ]; then
            pid=\$(cat \"\$file\" 2>/dev/null)
            [ -n \"\$pid\" ] && kill -9 \"\$pid\" 2>/dev/null || true
            rm -f \"\$file\"
        fi
    done
    rm -f '$REMOTE_ROOT/app-service-ready' \
        '$REMOTE_ROOT/app-service-start' \
        '$REMOTE_ROOT/$BRIDGE' \
        '$REMOTE_ROOT/$RUNNER' \
        '$REMOTE_ROOT/models/visual_504x392_fp16_mt6899.dla'
" >/dev/null

"$ADB" shell am force-stop "$PACKAGE" >/dev/null
"$ADB" shell am start -S -W -f 0x10008000 -n "$ACTIVITY" >/dev/null

printf 'Extracting packaged runtime...\n'
deadline=$((SECONDS + 1200))
next_report=0
bridge_started=0
while (( SECONDS < deadline )); do
    if (( bridge_started == 0 )) && \
            "$ADB" shell "test -f '$REMOTE_ROOT/$BRIDGE' \
                && test -f '$REMOTE_ROOT/app-service-start'" >/dev/null 2>&1; then
        "$ADB" shell "
            chmod 755 '$REMOTE_ROOT/$BRIDGE'
            cd '$REMOTE_ROOT'
            setsid sh './$BRIDGE' > bridge-daemon.log 2>&1 < /dev/null &
            echo \$! > bridge.pid
        " >/dev/null
        bridge_started=1
        printf 'Runtime extracted; loading NPU and MNN models...\n'
    fi

    if "$ADB" shell "test -s '$REMOTE_ROOT/app-service-ready'" >/dev/null 2>&1; then
        READY_MESSAGE="$("$ADB" shell cat "$REMOTE_ROOT/app-service-ready" | tr -d '\r')"
        "$ADB" shell am force-stop "$PACKAGE" >/dev/null
        "$ADB" shell am start -S -W -f 0x10008000 -n "$ACTIVITY" >/dev/null
        printf 'Progress: 100%% (%s/%s MiB)\n' "$EXPECTED_MIB" "$EXPECTED_MIB"
        printf 'Ready: %s\nPhoneVLM Letterbox is installed and open.\n' "$READY_MESSAGE"
        exit 0
    fi

    if (( SECONDS >= next_report )); then
        CURRENT_MIB="$("$ADB" shell "du -sm '$REMOTE_ROOT' 2>/dev/null | cut -f1" \
            | tr -d '\r')"
        CURRENT_MIB="${CURRENT_MIB:-0}"
        PERCENT=$((CURRENT_MIB * 100 / EXPECTED_MIB))
        (( PERCENT > 99 )) && PERCENT=99
        printf 'Progress: %d%% (%s/%s MiB)\n' \
            "$PERCENT" "$CURRENT_MIB" "$EXPECTED_MIB"
        next_report=$((SECONDS + 5))
    fi
    sleep 1
done

printf 'Bridge log:\n' >&2
"$ADB" shell "tail -n 40 '$REMOTE_ROOT/bridge-daemon.log' 2>/dev/null" >&2 || true
printf 'Runtime log:\n' >&2
"$ADB" shell "tail -n 80 '$REMOTE_ROOT/app-service.log' 2>/dev/null" >&2 || true
fail "Timed out while waiting for the resident runtime."
