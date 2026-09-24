# 全流程复现 - 纯 WSL Ubuntu 模式

> **结论: 全部 12 个步骤都可在 WSL Ubuntu 中完成**,无需切换到 Windows。

---

## ✅ WSL 已具备的能力

| 资源 | 状态 |
|------|------|
| Python 3.8 | ✓ 已安装 |
| pip3 | ✗ 需安装 (Step 1) |
| PyTorch / transformers / ONNX | ✗ 需安装 (Step 1) |
| adb (Linux 版) | ✓ **已装 1.0.39** (`/usr/bin/adb`) |
| 访问 Windows 文件 | ✓ `/mnt/c/...` 和 `/mnt/d/...` |
| 调用 Windows .exe | ✓ 直接 `/mnt/c/Users/.../adb.exe` |
| MT6899 真机连接 | ✓ USB 通过 WSL 透传 |

---

## 🚀 一键启动 (推荐)

### 在 Windows PowerShell 中:

```powershell
# 进入 WSL 并运行
wsl -e bash -c "bash /mnt/d/codes/Gallery/UDIdemos/PhoneVLM/phonevlm-letterbox/run_wsl.sh"
```

### 在 WSL Ubuntu 中:

```bash
bash /mnt/d/codes/Gallery/UDIdemos/PhoneVLM/phonevlm-letterbox/run_wsl.sh
```

脚本会询问:
- 从 Step 几开始? (1-12)
- 确认执行

---

## 📋 分步骤执行

```bash
# 进入 WSL
wsl

# 切换到项目目录
cd /mnt/d/codes/Gallery/UDIdemos/PhoneVLM/phonevlm-letterbox

# 给脚本执行权限
chmod +x step*.sh

# 依次执行
bash step1_install_deps.sh
bash step2_download_model.sh
...
bash step12_deploy.sh
```

---

## 🔑 关键事实

### 1. **adb 在 WSL 中可直连 MT6899**

USB 在 WSL 中已透传。直接:
```bash
adb devices
# 应该看到 MT6899
adb shell getprop ro.board.platform
# mt6899
```

**无需** Windows 端的 adb.exe,Linux 版 adb 1.0.39 已足矣。

### 2. **APK 路径在 WSL 中正确**

Gradle 构建的 APK 在 WSL 中构建完成,放在:
```
$WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk
```

直接用 WSL 的 adb 安装:
```bash
adb install -r -d $WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk
```

### 3. **NDK 编译在 WSL 中**

NDK r27b for Linux 完美支持 WSL Ubuntu:
```bash
unzip android-ndk-r27b-linux.zip
# toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
```

### 4. **NeuroPilot SDK 注意事项**

NeuroPilot SDK 是 ** MTK 专有的 NPU 运行时**:
- `libneuron_runtime.so.9.3.1` (NPU 驱动)
- `libneuron_adapter.so.9.3.1` (设备适配)
- `neuron compiler` (ONNX → DLA 工具)

**这是 MT6899 上跑 NPU 必需的,但 MNN CPU 模式可绕过**。

下载: https://neuropilot.mediatek.com (需注册)

---

## 📦 各步骤输出位置

```
~/minicpmv46-workdir/
├── npu_vision/                 # Step 6 输出
│   └── vision_504x392/
│       ├── visual.onnx
│       └── visual_fp16_mt6899.dla
├── npu_llm/                    # Step 7 输出
│   ├── prefill/layer_00~23/    # 24 个 prefill ONNX
│   ├── decode/layer_00~23/     # 24 个 decode ONNX
│   └── head/                   # LM head 分块输出
├── mnn_export/                 # Step 8 输出
│   ├── llm.mnn + llm.mnn.weight
│   ├── visual.mnn
│   └── tokenizer.mtok
├── build_mnn_android/libMNN.so # Step 9 输出
├── build/android/              # Step 9 输出
│   ├── minicpmv46-hybrid-runner
│   └── minicpmv46-letterbox-runner
├── runtime-assets/             # Step 10 输出 (Adaptive)
├── runtime-assets-letterbox/   # Step 10 输出 (Letterbox)
└── PhoneVLM-Letterbox-v1.2.1.apk  # Step 11 输出
```

设备端位置:
```
/data/local/tmp/minicpmv46-neuropilot-letterbox/
├── minicpmv46-letterbox-runner
├── minicpmv46-letterbox-bridge.sh
├── config.json
├── llm.mnn + llm.mnn.weight
├── llm_config.json
├── tokenizer.mtok
├── libMNN.so
├── libneuron_runtime.so.9.3.1
├── libneuron_adapter.so.9.3.1
└── models/
    └── visual_504x392_fp16_mt6899.dla
```

---

## ⚠️ 已知限制

| 限制 | 影响 | 解决方案 |
|------|------|----------|
| NeuroPilot SDK 受限 | 无 NPU 加速 | MNN Q4 CPU 模式仍可工作 |
| 模型导出慢 | 24 层 ONNX ~1h | 分批导出 + 探针验证 |
| WSL USB 偶尔需重启 | adb 设备断开 | `adb kill-server && adb start-server` |
| NDK 编译吃内存 | 可能 OOM | `make -j2` 而非 `-j$(nproc)` |

---

## 🎯 推荐执行顺序

| 阶段 | 时间 | 关键资源 |
|------|------|----------|
| Step 1 (依赖) | ~30 min | 网络带宽 |
| Step 2 (模型) | ~30 min | 磁盘 + 网络 |
| Step 4 (NDK) | ~10 min | 网络 |
| Step 5 (MNNConvert) | ~20 min | CPU |
| Step 6 (NPU Vision) | ~20 min | GPU 推荐 |
| Step 7 (NPU LLM) | ~60 min | GPU 推荐 |
| Step 8 (MNN Q4) | ~30 min | GPU 推荐 |
| Step 9 (Runner) | ~20 min | CPU |
| Step 11 (APK) | ~10 min | Gradle |
| Step 12 (部署) | ~5 min | USB + 真机 |

总计 ~4 小时 (含下载)。

---

## 📞 故障排查

### adb 找不到设备
```bash
# 重启 adb
adb kill-server
adb start-server
adb devices
```

### WSL 看不到 USB
```powershell
# 在 PowerShell 中 (管理员):
usbipd list
usbipd bind --busid <BUS_ID>
usbipd attach --wsl --busid <BUS_ID>
```

### NDK 编译失败
```bash
# 检查工具链
$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ --version

# 必须用 Linux 版,不能用 Windows .exe 版
```

### Gradle 构建失败
```bash
# 清理缓存
cd $MNN_REPO/deployments/minicpmv46/android
./gradlew clean
./gradlew :app:assembleLetterboxDebug
```

---

*文档由 Cursor AI 基于 `phonevlm-letterbox` 项目源码自动生成。*
*项目分支: `deploy/phonevlm-letterbox-v1.2.1`*