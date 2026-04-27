#!/usr/bin/env python3
"""
GPT-SoVITS 推理測試（直接 import，patch torchaudio）
"""
import sys
import os
import soundfile as sf
import torch

# === Patch torchaudio.load to use soundfile ===
import torchaudio
def _sf_load(path, *args, **kwargs):
    wav, sr = sf.read(path, dtype='float32')
    wav = torch.from_numpy(wav).float()
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    return wav, sr
torchaudio.load = _sf_load
print("torchaudio patched to use soundfile")

# Now import GPT-SoVITS
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")

GPT_CKPT = "/Users/kaede/tts/GPT-SoVITS/GPT_weights_v2/TaiwanTTS-e100.ckpt"
SOVITS_CKPT = "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2s血肉至高-v2Pro.ckpt"
REF_AUDIO = "/Users/kaede/tts/my_voice_data/processed/slices/2018 07 21 健康家庭文教基金會講座 戴志洋主講：未來世界的人才 _d5LVOs2nn_k__0001.wav"
REF_TEXT = "哈囉大家好，我是戴志洋，歡迎來到我的頻道。"
TARGET_TEXT = "今天要跟大家分享關於未來世界的人才培育與區塊鏈技術的發展。"
OUTPUT_DIR = "/tmp/gpt_sovits_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Load models
print("Loading GPT...")
change_gpt_weights(gpt_path=GPT_CKPT)
print("Loading SoVITS...")
change_sovits_weights(sovits_path=SOVITS_CKPT)
print("Models loaded!")

# Synthesize
print(f"\nRef: {REF_TEXT}")
print(f"Target: {TARGET_TEXT}")
print("Synthesizing...")

result = get_tts_wav(
    ref_wav_path=REF_AUDIO,
    prompt_text=REF_TEXT,
    prompt_language="Chinese",
    text=TARGET_TEXT,
    text_language="Chinese",
    top_p=1,
    temperature=1,
)

import soundfile as sf_out
import numpy as np
result_list = list(result)
if result_list:
    sr, audio = result_list[-1]
    out_path = os.path.join(OUTPUT_DIR, "output.wav")
    sf_out.write(out_path, audio, sr)
    print(f"\nSaved: {out_path}")
    print(f"Duration: {len(audio)/sr:.1f}s, Sample rate: {sr}")
else:
    print("No audio output!")
