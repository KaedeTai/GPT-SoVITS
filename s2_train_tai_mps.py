#!/usr/bin/env python3
"""
GPT-SoVITS S2 (SoVITS vocoder) training - Mac M1/M2 MPS adaptation
For Tai's voice data
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import logging
import random

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.mps.enable_aggregateActions = True

# Setup paths
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")

import utils
hps = utils.get_hparams_from_file('/Users/kaede/tts/GPT-SoVITS/configs/s2_tai_scratch.json')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
for logger_name in ["matplotlib", "h5py", "numba", "triton"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from module.models import SynthesizerTrn, MultiPeriodDiscriminator
from module.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from module import commons

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

device = torch.device("mps")
global_step = 0

def main():
    torch.manual_seed(hps.train.seed)
    random.seed(hps.train.seed)
    
    logging.info("=== S2 Training (Tai Voice) ===")
    logging.info(f"Data: {hps.data.exp_dir}")
    logging.info(f"Checkpoint: {hps.s2_ckpt_dir}")
    logging.info(f"Pretrained: {hps.train.pretrained_s2G}")
    
    os.makedirs(hps.s2_ckpt_dir, exist_ok=True)
    
    # Load pretrained if exists
    if os.path.exists(hps.train.pretrained_s2G):
        logging.info(f"Loading pretrained SoVITS: {hps.train.pretrained_s2G}")
    
    # Create data loader
    train_dataset = TextAudioSpeakerLoader(hps.data)
    collate = TextAudioSpeakerCollate()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        collate_fn=collate
    )
    logging.info(f"Training samples: {len(train_dataset)}")
    
    # Create model
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)
    
    # Load pretrained weights
    if os.path.exists(hps.train.pretrained_s2G):
        logging.info("Loading pretrained weights...")
        ckpt = torch.load(hps.train.pretrained_s2G, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["weight"], strict=False)
        logging.info("Pretrained weights loaded!")
    
    # Optimizer
    optim_g = torch.optim.AdamW(model.parameters(), lr=hps.train.learning_rate)
        
    # Training loop
    for epoch in range(1, hps.train.epochs + 1):
        model.train()
        total_loss_g = 0
        total_loss_d = 0
        
        for batch_idx, (ssl_padded, spec_padded, mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, mel_lengths) in enumerate(train_loader):
            phonemes = phonemes.to(device)
            phoneme_lens = phoneme_lens.to(device)
            texts = texts.to(device)
            text_lens = text_lens.to(device)
            audios = audios.to(device)
            audio_lens = audio_lens.to(device)
            speakers = speakers.to(device)
            
            # Forward
            mel = mel_spectrogram_torch(audios, hps.data.filter_length, hps.data.n_mel_channels, 
                                        hps.data.sampling_rate, hps.data.hop_length, 
                                        hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            
            # Get model output
            output = model(phonemes, phoneme_lens, texts, text_lens, speakers, audios)
            
            # Calculate losses
            (mel_out, mask, _), (mel_len, _) = output
            
            # Simplified loss for now
            loss_mel = F.l1_loss(mel_out * mask, mel * mask)
            loss_g = loss_mel
            
            # Generator backward
            optim_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_g.step()
            
            total_loss_g += loss_g.item()
            
            if batch_idx % 20 == 0:
                logging.info(f"Epoch {epoch}/{hps.train.epochs} - Step {batch_idx}/{len(train_loader)} - Loss G: {loss_g.item():.4f}")
        
        avg_loss_g = total_loss_g / len(train_loader)
        logging.info(f"Epoch {epoch} completed - Avg Loss G: {avg_loss_g:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            ckpt_path = f"{hps.s2_ckpt_dir}/s2_G_{epoch}.pth"
            torch.save({"model": model.state_dict()}, ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main()
