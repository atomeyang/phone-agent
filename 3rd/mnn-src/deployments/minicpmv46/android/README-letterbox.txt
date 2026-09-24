PhoneVLM Letterbox Android Demo
===============================

Version: 1.2.1-letterbox
Image path: proportional fit into a 392x504 canvas with centered padding.
Vision input: one tile and 63 visual tokens; no image content is cropped.
Target: MT6899 engineering device with root ADB enabled.

Windows:
1. Connect and authorize one phone over USB.
2. Double-click setup-minicpmv46-letterbox-demo.bat.
3. Wait until progress reaches 100% and the app opens.

WSL or Git Bash:
1. Connect and authorize one phone over USB.
2. Run: bash setup-minicpmv46-letterbox-demo.sh

The installer uses Windows Android SDK adb.exe, requires no arguments, and can
install beside the PhoneVLM Adaptive app. The two apps use separate
packages, runtime directories, bridge scripts, and runner processes.
