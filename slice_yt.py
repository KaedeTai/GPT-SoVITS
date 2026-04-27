#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import os, sys
sys.path.insert(0, '/Users/kaede/tts/GPT-SoVITS')
from tools.slicer2 import Slicer
from tools.my_utils import load_audio

def slice_folder(input_dir, output_dir, sr=32000):
    os.makedirs(output_dir, exist_ok=True)
    slicer = Slicer(sr=sr, threshold=-40, min_length=3000, min_interval=200, hop_size=20, max_sil_kept=2000)
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.wav', '.m4b', '.flac'))])
    total = 0
    
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        name = os.path.splitext(fname)[0]
        try:
            audio = load_audio(fpath, sr)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                out_name = f"{name}_{start:010d}_{end:010d}.wav"
                wavfile.write(os.path.join(output_dir, out_name), sr, (chunk * 32767).astype(np.int16))
                total += 1
            print(f"  {fname}: OK ({total} total)")
        except Exception as e:
            print(f"  {fname}: FAIL - {e}")
    return total

if __name__ == "__main__":
    datasets = [
        ("/Users/kaede/tts/training_data/YouTube-Taiwan", "/Users/kaede/tts/training_data/sliced/YouTube-Taiwan"),
        ("/Users/kaede/tts/training_data/YouTube-HeroStory", "/Users/kaede/tts/training_data/sliced/YouTube-HeroStory"),
        ("/Users/kaede/tts/training_data/YouTube-JinYong", "/Users/kaede/tts/training_data/sliced/YouTube-JinYong"),
    ]
    
    for inp, out in datasets:
        if os.path.exists(inp):
            n = len([f for f in os.listdir(inp) if f.endswith(('.mp3', '.wav', '.m4b', '.flac'))])
            if n > 0:
                print(f"Slicing {inp} ({n} files)...")
                slice_folder(inp, out)
                print(f"  -> {out}")
            else:
                print(f"SKIP {inp}: no audio files")
        else:
            print(f"SKIP {inp}: not found")