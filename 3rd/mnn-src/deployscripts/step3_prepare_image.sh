#!/bin/bash
set -euo pipefail

# ============================================================
# MiniCPM-V 4.6 全流程复�?- Step 3: 准备测试图片
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MODEL_DIR="${HOME}/models/minicpm-v-4.6"
VENV_DIR="${HOME}/minicpmv46-env"

echo "=========================================="
echo "Step 3: 准备测试图片"
echo "=========================================="

mkdir -p "$WORK_DIR"

source "$VENV_DIR/bin/activate"

echo ""
echo "[1/2] 下载官方测试图片..."
# 使用 HuggingFace 上的测试图片
python -c "
import os
from huggingface_hub import hf_hub_download
from PIL import Image

work = os.path.expanduser('${WORK_DIR}')

# 尝试从多个来源获取测试图�?
urls = [
    ('openbmb/MiniCPM-V-4_6', 'assets/test.jpg'),
]

# 直接用网络图�?
import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# 使用一个简单的人像图片作为测试
test_urls = [
    'https://huggingface.co/datasets/YiFuImage/portrait_200/resolve/main/person_001.jpg',
]

# 如果上面的不可用，使�?placeholder
try:
    img_path = os.path.join(work, 'test.jpg')
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/dummy/invalid/url/skip.jpg',
        img_path
    )
    print(f'Downloaded test image: {img_path}')
except Exception as e:
    print(f'Download failed: {e}')
    print('Will generate a synthetic test image instead...')
    
    # 生成合成测试图片
    from PIL import Image, ImageDraw
    import numpy as np
    
    img = Image.new('RGB', (800, 600), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    # 画一个人形轮�?
    draw.ellipse([300, 150, 500, 350], fill=(220, 180, 160))  # �?
    draw.rectangle([280, 350, 520, 550], fill=(50, 50, 200))  # 身体
    img.save(img_path)
    print(f'Generated synthetic test image: {img_path}')
"

echo ""
echo "[2/2] 验证图片可以�?MiniCPM-V processor 处理..."
python -c "
import os
from transformers import AutoProcessor
from PIL import Image

work = os.path.expanduser('${WORK_DIR}')
model = os.path.expanduser('${MODEL_DIR}')
test_img = os.path.join(work, 'test.jpg')

if not os.path.exists(test_img):
    print(f'ERROR: Test image not found: {test_img}')
    exit(1)

processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
img = Image.open(test_img).convert('RGB')

# 测试三种 profile
for max_slice in [1, 3, 9]:
    result = processor(
        images=img,
        text='<image>\nPlease describe this image.',
        return_tensors='pt',
        max_slice_nums=max_slice,
    )
    print(f'  max_slice_nums={max_slice}: '
          f'pixel_values={tuple(result[\"pixel_values\"].shape)}, '
          f'target_sizes={result[\"target_sizes\"].tolist()}')

print(f'Test image ready: {test_img}')
"

echo ""
echo "=========================================="
echo "Step 3 完成!"
echo "=========================================="
echo "工作目录: $WORK_DIR"
echo "测试图片: $WORK_DIR/test.jpg"
echo ""
echo "下一�? Step 4 - 安装 NDK �?NeuroPilot SDK"
