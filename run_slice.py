#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import os

# Use the slicer2 directly
from tools.slicer2 import Slicer
from tools.my_utils import load_audio

def slice_folder(input_dir, output_dir, sr=32000):
    os.makedirs(output_dir, exist_ok=True)
    slicer = Slicer(sr=sr, threshold=-40, min_length=3000, min_interval=200, hop_size=20, max_sil_kept=2000)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.wav', '.m4b', '.flac'))]
    files = sorted(files)
    total = 0
    
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        name = fname.rsplit('.', 1)[0]
        try:
            audio = load_audio(fpath, sr)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * 0.9 * 0.2) + (1 - 0.2) * chunk
                out_name = f"{name}_{start:010d}_{end:010d}.wav"
                wavfile.write(os.path.join(output_dir, out_name), sr, (chunk * 32767).astype(np.int16))
                total += 1
            print(f"  {fname}: OK")
        except Exception as e:
            print(f"  {fname}: FAIL - {e}")
    return total

if __name__ == "__main__":
    datasets = [
        ("/Users/kaede/tts/training_data/LibriVox-Chinese/LunYu", "/Users/kaede/tts/training_data/sliced/LunYu"),
        ("/Users/kaede/tts/training_data/ChineseBible-Catholic", "/Users/kaede/tts/training_data/sliced/Bible"),
        ("/Users/kaede/tts/training_data/LibriVox-Chinese/AnalectsChinese", "/Users/kaede/tts/training_data/sliced/Analects"),
    ]
    
    for inp, out in datasets:
        if os.path.exists(inp):
            print(f"Slicing {inp}...")
            n = slice_folder(inp, out)
            print(f"  -> {n} files created")