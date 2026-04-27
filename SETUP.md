# KaedeTai's GPT-SoVITS Fork - Setup Guide

## Quick Setup

```bash
# 1. Clone this fork
git clone git@github.com:KaedeTai/GPT-SoVITS.git
cd GPT-SoVITS

# 2. Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download pretrained models
chmod +x download_pretrained.sh
./download_pretrained.sh
```

## Requirements

- Python 3.10 - 3.12
- PyTorch (with MPS support for Mac M1 Max)
- ffmpeg
- sox (for audio recording)

### Install system dependencies (macOS)

```bash
brew install ffmpeg sox
```

## Download Pretrained Models

Run `download_pretrained.sh` to download:
- `s1v3.ckpt` - GPT-SoVITS v3 model
- `chinese-roberta-wwm-ext-large/` - Chinese BERT
- `s2Gv2Pro.pth` / `s2Dv2Pro.pth` - SoVITS v2 Pro

Manual download from HuggingFace:
```
https://huggingface.co/lj1995/GPT-SoVITS/tree/main
```

## Training

### S2 (SoVITS) Fine-tuning
```bash
source .venv/bin/activate
nohup .venv/bin/python -u s2_train_mps.py -c configs/s2_tai_v2pro_finetune.json > ~/tts/s2_train.log 2>&1 &
```

### S1 (GPT) Fine-tuning
```bash
source .venv/bin/activate
nohup .venv/bin/python -u s1_train_mps.py -c configs/s1_tai_finetune.yaml > ~/tts/s1_train.log 2>&1 &
```

## Inference

### Command Line
```bash
source .venv/bin/activate
.venv/bin/python inference_cli.py --text "要轉換的文字" --ref_wav "參考音頻.wav" --ref_text "參考音頻的內容"
```

### Python API
```python
import sys
sys.path.insert(0, '/path/to/GPT-SoVITS')
# See inference_finetuned.py in the tts/ repo
```

## NLTK Data (required for English text processing)

```bash
.venv/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt')"
```

## Project Structure

```
GPT-SoVITS/
├── .venv/                  # Virtual environment (NOT committed)
├── GPT_SoVITS/
│   ├── pretrained_models/  # Downloaded separately (NOT committed)
│   ├── module/             # Core modules (with MPS fixes)
│   └── inference_webui.py  # Inference engine
├── configs/                 # Training configs
├── s1_train_mps.py        # S1 training (Mac MPS)
├── s2_train_mps.py         # S2 training (Mac MPS)
├── download_pretrained.sh    # Download pretrained models
└── inference_cli.py         # CLI inference
```

## MPS Compatibility

This fork includes fixes for Apple Silicon (M1/M2/M3) MPS training:
- `GPT_SoVITS/module/core_vq.py` - MPS quantization fix
- `GPT_SoVITS/module/ddp_utils.py` - MPS distributed training fix

## Voice Cloning Scripts

See https://github.com/KaedeTai/tts for voice cloning scripts:
- `inference_finetuned.py` - Fine-tuned TTS inference
- `inference_mixed.py` - Mixed language TTS (Chinese + English)
- `tts_auto_split.py` - Long text auto-split
- `record_sample.py` - Record voice samples
