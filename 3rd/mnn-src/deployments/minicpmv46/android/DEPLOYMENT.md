# PhoneVLM Letterbox 1.2.1

- Git branch: `deploy/phonevlm-letterbox-v1.2.1`
- Android package: `com.example.minicpmv46.neuropilot.letterbox`
- Gradle flavor: `letterbox`
- Vision: MT6899 NeuroPilot FP16 `504x392` profile
- Language model: MNN Q4 on four performance CPU cores
- Image policy: proportional resize and centered padding to `504x392`
- Vision input: one tile and 63 visual tokens
- Source images are never cropped or stretched.

Build with external runtime assets:

```bash
gradle :app:assembleLetterboxDebug \
  -PletterboxRuntimeAssetsDir=/path/to/letterbox/runtime-assets
```

Generated models, proprietary runtime libraries, and APKs are not stored in
Git. The validated device package is `PhoneVLM-Letterbox-v1.2.1.apk`.
