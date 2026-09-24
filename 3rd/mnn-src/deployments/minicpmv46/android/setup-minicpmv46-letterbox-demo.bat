@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PACKAGE=com.example.minicpmv46.neuropilot.letterbox"
set "ACTIVITY=%PACKAGE%/com.example.minicpmv46.neuropilot.MainActivity"
set "REMOTE_ROOT=/data/local/tmp/minicpmv46-neuropilot-letterbox"
set "RUNNER=minicpmv46-letterbox-runner"
set "BRIDGE=minicpmv46-letterbox-bridge.sh"
set "EXPECTED_MIB=1473"
set "SCRIPT_DIR=%~dp0"

call :find_adb
if not defined ADB (
    echo ERROR: Windows Android SDK adb.exe was not found. 1>&2
    exit /b 1
)
call :find_apk
if not defined APK (
    echo ERROR: PhoneVLM Letterbox APK was not found beside this script. 1>&2
    exit /b 1
)

set "STATE_FILE=%TEMP%\minicpmv46-letterbox-%RANDOM%-%RANDOM%.tmp"
set /a DEVICE_COUNT=0
set "DEVICE_SERIAL="
"%ADB%" devices >"%STATE_FILE%" || goto :adb_error
for /f "usebackq skip=1 tokens=1,2" %%A in ("%STATE_FILE%") do (
    if "%%B"=="device" (
        set /a DEVICE_COUNT+=1
        set "DEVICE_SERIAL=%%A"
    )
)
if not !DEVICE_COUNT! EQU 1 (
    echo ERROR: Connect and authorize exactly one Android device. 1>&2
    goto :failed
)

"%ADB%" wait-for-device >nul
"%ADB%" shell getprop ro.product.model >"%STATE_FILE%"
set /p "MODEL=" <"%STATE_FILE%"
"%ADB%" shell getprop ro.board.platform >"%STATE_FILE%"
set /p "PLATFORM=" <"%STATE_FILE%"
if /i not "!PLATFORM!"=="mt6899" (
    echo ERROR: Unsupported platform '!PLATFORM!' on '!MODEL!'; this build requires MT6899. 1>&2
    goto :failed
)

"%ADB%" shell id -u >"%STATE_FILE%"
set /p "ADB_UID=" <"%STATE_FILE%"
if not "!ADB_UID!"=="0" (
    echo Requesting root ADB access...
    "%ADB%" root >nul 2>&1
    "%ADB%" wait-for-device >nul
    "%ADB%" shell id -u >"%STATE_FILE%"
    set /p "ADB_UID=" <"%STATE_FILE%"
)
if not "!ADB_UID!"=="0" (
    echo ERROR: This phone does not provide the root ADB identity required by NeuroPilot. 1>&2
    goto :failed
)

echo ADB: %ADB%
echo Device: !MODEL! (!PLATFORM!)
echo Installing Letterbox APK...
"%ADB%" install -r -d "%APK%" || goto :adb_error
"%ADB%" shell "mkdir -p '%REMOTE_ROOT%' && chmod -R 777 '%REMOTE_ROOT%'" >nul || goto :adb_error
"%ADB%" shell "for pid in $(pidof '%RUNNER%' 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done; for pid in $(pidof sh 2>/dev/null); do script=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null ^| sed -n '2p'); case "$script" in ./%BRIDGE%^|*/%BRIDGE%) kill -9 "$pid" 2>/dev/null || true ;; esac; done; for file in '%REMOTE_ROOT%/bridge.pid' '%REMOTE_ROOT%/app-runner.pid'; do if [ -f "$file" ]; then pid=$(cat "$file" 2>/dev/null); [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true; rm -f "$file"; fi; done; rm -f '%REMOTE_ROOT%/app-service-ready' '%REMOTE_ROOT%/app-service-start' '%REMOTE_ROOT%/%BRIDGE%' '%REMOTE_ROOT%/%RUNNER%' '%REMOTE_ROOT%/models/visual_504x392_fp16_mt6899.dla'" >nul
"%ADB%" shell am force-stop "%PACKAGE%" >nul
"%ADB%" shell am start -S -W -f 0x10008000 -n "%ACTIVITY%" >nul || goto :adb_error

echo Extracting packaged runtime...
set /a BRIDGE_STARTED=0
for /l %%I in (1,1,1200) do (
    if !BRIDGE_STARTED! EQU 0 (
        "%ADB%" shell "test -f '%REMOTE_ROOT%/%BRIDGE%' && test -f '%REMOTE_ROOT%/app-service-start'" >nul 2>&1
        if not errorlevel 1 (
            "%ADB%" shell "chmod 755 '%REMOTE_ROOT%/%BRIDGE%'; cd '%REMOTE_ROOT%'; setsid sh './%BRIDGE%' > bridge-daemon.log 2>&1 < /dev/null & echo $! > bridge.pid" >nul
            set /a BRIDGE_STARTED=1
            echo Runtime extracted; loading NPU and MNN models...
        )
    )

    "%ADB%" shell "test -s '%REMOTE_ROOT%/app-service-ready'" >nul 2>&1
    if not errorlevel 1 goto :ready

    set /a REPORT_TICK=%%I %% 5
    if !REPORT_TICK! EQU 1 (
        set "CURRENT_MIB=0"
        "%ADB%" shell "du -sm '%REMOTE_ROOT%' 2>/dev/null" >"%STATE_FILE%"
        for /f "usebackq tokens=1" %%S in ("%STATE_FILE%") do set "CURRENT_MIB=%%S"
        set /a PERCENT=!CURRENT_MIB!*100/%EXPECTED_MIB%
        if !PERCENT! GTR 99 set "PERCENT=99"
        echo Progress: !PERCENT!%% ^(!CURRENT_MIB!/%EXPECTED_MIB% MiB^)
    )
    >nul ping 127.0.0.1 -n 2
)

echo Bridge log: 1>&2
"%ADB%" shell "tail -n 40 '%REMOTE_ROOT%/bridge-daemon.log' 2>/dev/null" 1>&2
echo Runtime log: 1>&2
"%ADB%" shell "tail -n 80 '%REMOTE_ROOT%/app-service.log' 2>/dev/null" 1>&2
echo ERROR: Timed out while waiting for the resident runtime. 1>&2
goto :failed

:ready
"%ADB%" shell cat "%REMOTE_ROOT%/app-service-ready" >"%STATE_FILE%"
set /p "READY_MESSAGE=" <"%STATE_FILE%"
"%ADB%" shell am force-stop "%PACKAGE%" >nul
"%ADB%" shell am start -S -W -f 0x10008000 -n "%ACTIVITY%" >nul
echo Progress: 100%% ^(%EXPECTED_MIB%/%EXPECTED_MIB% MiB^)
echo Ready: !READY_MESSAGE!
echo PhoneVLM Letterbox is installed and open.
del /q "%STATE_FILE%" >nul 2>&1
exit /b 0

:find_adb
set "ADB="
if defined ANDROID_SDK_ROOT if exist "%ANDROID_SDK_ROOT%\platform-tools\adb.exe" set "ADB=%ANDROID_SDK_ROOT%\platform-tools\adb.exe"
if not defined ADB if defined ANDROID_HOME if exist "%ANDROID_HOME%\platform-tools\adb.exe" set "ADB=%ANDROID_HOME%\platform-tools\adb.exe"
if not defined ADB if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" set "ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
if not defined ADB for /f "delims=" %%A in ('where adb.exe 2^>nul') do if not defined ADB set "ADB=%%A"
exit /b 0

:find_apk
set "APK="
if exist "%SCRIPT_DIR%PhoneVLM-Letterbox-v1.2.1.apk" set "APK=%SCRIPT_DIR%PhoneVLM-Letterbox-v1.2.1.apk"
if not defined APK if exist "%SCRIPT_DIR%MiniCPM-V-4.6-NPU-MNN-v1.2.0-letterbox.apk" set "APK=%SCRIPT_DIR%MiniCPM-V-4.6-NPU-MNN-v1.2.0-letterbox.apk"
if not defined APK if exist "%SCRIPT_DIR%app-letterbox-debug.apk" set "APK=%SCRIPT_DIR%app-letterbox-debug.apk"
if not defined APK if exist "%SCRIPT_DIR%app\build\outputs\apk\letterbox\debug\app-letterbox-debug.apk" set "APK=%SCRIPT_DIR%app\build\outputs\apk\letterbox\debug\app-letterbox-debug.apk"
exit /b 0

:adb_error
echo ERROR: An ADB command failed. 1>&2
:failed
del /q "%STATE_FILE%" >nul 2>&1
exit /b 1
