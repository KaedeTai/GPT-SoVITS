#!/usr/bin/env python3
"""
GPT-SoVITS 推理測試
使用 Direction B：訓練好的 GPT + 官方 SoVITS v2Pro 預訓練模型
"""
import os
import sys
import torch

# 加入路徑
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")

from GPT_SoVITS.s1_train import (
    get_sovits_model,
    get_bert_model,
    get_phoneme_to_semantic,
)
from GPT_SoVITS.GPT_SoVITS.models import commons
from GPT_SoVITS.GPT_SoVITS.data.collate import collate_fn
import soundfile as sf
import numpy as np

# === 設定 ===
GPT_CKPT = "/Users/kaede/tts/GPT-SoVITS/GPT_weights_v2/TaiwanTTS-e100.ckpt"
SOVITS_CKPT = "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2s血肉至高-v2Pro.ckpt"
BERT_DIR = "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
DEVICE = "mps"

# 測試文字（繁體中文）
TEST_TEXTS = [
    "哈囉，大家好我是戴志洋，歡迎來到我的頻道。",
    "今天要跟大家分享關於未來世界的人才培育。",
    "區塊鏈技術正在改變我們的金融生態。",
]


def load_gpt_model(gpt_ckpt, sovits_ckpt, bert_dir, device):
    """載入 GPT + SoVITS 模型"""
    print(f"載入 SoVITS v2Pro from {sovits_ckpt}...")
    
    # SoVITS 模型
    sovits_cfg = torch.load(sovits_ckpt, map_location="cpu")
    # 嘗試取 config
    if "config" in sovits_cfg:
        sovits_config = sovits_cfg["config"]
    else:
        sovits_config = sovits_cfg
    
    sovits_model = get_sovits_model(sovits_config, device)
    
    # GPT 模型（Direction B: 只用 semantic token）
    print(f"載入 GPT from {gpt_ckpt}...")
    from pytorch_lightning import LightningModule
    
    # 建立模型
    from GPT_SoVITS.GPT_SoVITS.models import Text2Semantic
    model = Text2Semantic()
    state_dict = torch.load(gpt_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.to(device)
    model.eval()
    
    # BERT
    print("載入 BERT...")
    bert_model = get_bert_model(bert_dir, device)
    
    return model, sovits_model, bert_model


def inference(text, gpt_model, sovits_model, bert_model, device):
    """推理"""
    print(f"\n推理文字: {text}")
    
    # 1. 取得 phoneme
    from pypinyin import pypinyin
    from pypinyin.contrib.tone_convert import to_initials
    
    # 簡單的 pinyin conversion（用 GPT-SoVITS 內建）
    # 這裡需要參考原始代碼的 text2phoneme 邏輯
    # 暫時用 init + phone 轉換
    
    # 實際上 GPT-SoVITS 的推理流程更複雜
    # 建議用官方 inference-webui 或 tts_inference.py
    
    print("  [注意] 完整推理需要 text2phoneme，請用官方 inference 腳本")
    return None


def main():
    print("=" * 60)
    print("GPT-SoVITS 推理測試 (Direction B)")
    print("=" * 60)
    
    # 檢查檔案
    for f in [GPT_CKPT, SOVITS_CKPT]:
        if not os.path.exists(f):
            print(f"錯誤: 找不到 {f}")
            return
    
    print(f"GPT checkpoint: {GPT_CKPT}")
    print(f"SoVITS checkpoint: {SOVITS_CKPT}")
    print(f"Device: {DEVICE}")
    
    # 官方推理腳本更完整
    print("\n建議用官方推理方式:")
    print("  cd /Users/kaede/tts/GPT-SoVITS")
    print("  python GPT_SoVITS/inference.py \\")
    print("    --gpt_path GPT_weights_v2/TaiwanTTS-e100.ckpt \\")
    print("    --sovits_path GPT_SoVITS/pretrained_models/v2Pro/s2s血肉至高-v2Pro.ckpt \\")
    print("    --text '測試文字' --output test.wav")


if __name__ == "__main__":
    main()
