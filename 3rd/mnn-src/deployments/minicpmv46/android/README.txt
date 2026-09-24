PhoneVLM Adaptive Android Demo
==============================

Route: NeuroPilot FP16 vision + MNN Q4 language model.
Image mode: official dynamic multi-tile when supported, otherwise proportional
letterbox fitting into 392x504. No image content is cropped.
Version: 1.3.1
Target: MT6899 engineering device with root ADB enabled.
Default prompt: Please describe this image.

Windows:
1. Connect and authorize one phone over USB.
2. Double-click setup-minicpmv46-demo.bat.
3. Wait until the app opens and reports that both models are loaded.

WSL or Git Bash:
1. Connect and authorize one phone over USB.
2. Run: bash setup-minicpmv46-demo.sh

The setup scripts locate Windows Android SDK adb.exe automatically. The first
installation extracts the bundled runtime and model files, so it takes longer
than later launches. The models remain resident after setup. Any image size or
aspect ratio can be selected directly. Edit the prompt, then press Run. The
output is streamed while TTFT and stage metrics are displayed.
