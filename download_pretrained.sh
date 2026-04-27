#!/bin/bash
# Download GPT-SoVITS pretrained models
# Run from GPT-SoVITS root directory

set -e

echo "Downloading GPT-SoVITS Pretrained Models..."
echo ""

DEST_GPT="./GPT_SoVITS/pretrained_models"
DEST_SOVITS="./SoVITS_weights_v2Pro"

mkdir -p "$DEST_GPT/v2Pro" "$DEST_SOVITS"

# === GPT-SoVITS Models (text-to-semantic) ===
# Model: GPT-SoVITS v2 Pro

echo "[1/4] Downloading s1v3.ckpt (GPT model, 148MB)..."
curl -L "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt" \
  -o "$DEST_GPT/s1v3.ckpt" --progress-bar

echo "[2/4] Downloading Chinese RoBERTa (BERT model, 800MB)..."
curl -L "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large.zip" \
  -o /tmp/chinese-roberta.zip --progress-bar
unzip -o /tmp/chinese-roberta.zip -d "$DEST_GPT/"
rm /tmp/chinese-roberta.zip

echo "[3/4] Downloading s2Gv2Pro.pth (SoVITS Generator v2 Pro, 155MB)..."
curl -L "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2Gv2Pro.pth" \
  -o "$DEST_SOVITS/s2Gv2Pro.pth" --progress-bar

echo "[4/4] Downloading s2Dv2Pro.pth (SoVITS Discriminator v2 Pro, 121MB)..."
curl -L "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2Dv2Pro.pth" \
  -o "$DEST_SOVITS/s2Dv2Pro.pth" --progress-bar

echo ""
echo "Download complete!"
echo ""
echo "Files:"
ls -lh "$DEST_GPT/s1v3.ckpt" 2>/dev/null || echo "  s1v3.ckpt: MISSING"
ls -lh "$DEST_GPT/chinese-roberta-wwm-ext-large/" 2>/dev/null | head -1 || echo "  chinese-roberta: MISSING"
ls -lh "$DEST_SOVITS/"*.pth 2>/dev/null || echo "  SoVITS weights: MISSING"
echo ""
echo "Note: For v3/v4 models, download from:"
echo "  https://huggingface.co/lj1995/GPT-SoVITS/tree/main"
