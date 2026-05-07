#!/usr/bin/env python3
"""
GPT-SoVITS S2 (SoVITS vocoder) training - Mac M1/M2 MPS adaptation
Based on s2_train.py but modified for MPS (no DDP, single GPU)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
import logging
import random
import json

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.mps.enable_aggregateActions = True

# Setup paths
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")

import utils
import sys
config_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/kaede/tts/GPT-SoVITS/configs/s2_tai_v2pro_finetune.json'
hps = utils.get_hparams_from_file(config_path)


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

device = torch.device("mps")  # DEBUG: using CPU instead of MPS
global_step = 0

def main():
    torch.manual_seed(hps.train.seed)
    random.seed(hps.train.seed)
    
    logging.info("=== S2 Training (MPS) ===")
    logging.info(f"Config: {hps.data.exp_dir}")
    
    # Data
    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.version)
    collate_fn = TextAudioSpeakerCollate(version=hps.version)
    
    # Simple batch sampler (no DistributedBucketSampler)
    from torch.utils.data import DataLoader
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    
    logging.info(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    
    # Initialize dummy distributed process group for single-process training
    # (Some model code uses dist.get_rank())
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="tcp://localhost:29501", 
                               world_size=1, rank=0)
    
    # Models - MPS version
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(
        hps.model.use_spectral_norm, 
        version=hps.version
    ).to(device)
    
    logging.info("Models initialized on MPS")
    
    # Optimizers with different LR for text encoder
    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )
    
    optim_g = torch.optim.AdamW(
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {
                "params": net_g.enc_p.text_embedding.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.encoder_text.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.mrte.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
        ],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    
    # Check for existing checkpoint to resume from
    ckpt_dir = hps.s2_ckpt_dir
    existing_ckpts = []
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith('s2_G_') and f.endswith('.pth'):
                epoch_num = int(f.replace('s2_G_', '').replace('.pth', ''))
                existing_ckpts.append(epoch_num)
    
    if existing_ckpts:
        resume_epoch = max(existing_ckpts)
        epoch_str = resume_epoch + 1   # start training at NEXT epoch — don't overwrite the resumed ckpt
        g_ckpt = os.path.join(ckpt_dir, f's2_G_{resume_epoch}.pth')
        d_ckpt = os.path.join(ckpt_dir, f's2_D_{resume_epoch}.pth')
        full_ckpt = os.path.join(ckpt_dir, f's2_full_{resume_epoch}.pth')
        logging.info(f'Resuming from epoch {resume_epoch}')

        # Prefer s2_full_NN.pth for G — it has enc_q and opt_g (no warm-up needed).
        # Fall back to s2_G_NN.pth (slim savee format) if full state missing.
        if os.path.exists(full_ckpt):
            g_state = torch.load(full_ckpt, map_location='cpu', weights_only=False)
            net_g.load_state_dict(g_state['model'], strict=False)
            if 'opt_g' in g_state and isinstance(g_state['opt_g'], dict):
                try:
                    optim_g.load_state_dict(g_state['opt_g'])
                    logging.info('Restored opt_g state — no optimizer warm-up needed')
                except Exception as e:
                    logging.warning(f'opt_g restore failed (will warm up): {e}')
            global_step = g_state.get('step', resume_epoch * len(train_loader))
            logging.info(f'Loaded G full state from {full_ckpt} (step={global_step})')
        elif os.path.exists(g_ckpt):
            from GPT_SoVITS.process_ckpt import load_sovits_new
            g_state = load_sovits_new(g_ckpt)
            if 'model' in g_state:
                net_g.load_state_dict(g_state['model'], strict=False)
            elif 'weight' in g_state:
                net_g.load_state_dict(g_state['weight'], strict=False)
            logging.warning(f'No s2_full_{resume_epoch}.pth found — opt_g and enc_q reset; expect a brief warm-up.')
            logging.info(f'Loaded G slim from {g_ckpt}')
            global_step = resume_epoch * len(train_loader)

        if os.path.exists(d_ckpt):
            d_state = torch.load(d_ckpt, map_location='cpu', weights_only=False)
            if 'model' in d_state:
                net_d.load_state_dict(d_state['model'], strict=False)
            elif 'weight' in d_state:
                net_d.load_state_dict(d_state['weight'], strict=False)
            if 'opt_d' in d_state and isinstance(d_state['opt_d'], dict):
                try:
                    optim_d.load_state_dict(d_state['opt_d'])
                    logging.info('Restored opt_d state')
                except Exception as e:
                    logging.warning(f'opt_d restore failed (will warm up): {e}')
            logging.info(f'Loaded D from epoch {resume_epoch}')

        logging.info(f'Resuming from step {global_step}')
    else:
        epoch_str = 1
        global_step = 0
    
    # Load pretrained weights if available (only if no checkpoint to resume from)
    if not existing_ckpts and hps.train.pretrained_s2G and os.path.exists(hps.train.pretrained_s2G):
        logging.info(f"Loading pretrained SoVITS G from {hps.train.pretrained_s2G}")
        pretrained_state = torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)
        if "weight" in pretrained_state:
            net_g.load_state_dict(pretrained_state["weight"], strict=False)
        else:
            net_g.load_state_dict(pretrained_state, strict=False)
        logging.info("Loaded pretrained SoVITS G")
    else:
        logging.info("Training SoVITS from scratch")
    
    if not existing_ckpts and hps.train.pretrained_s2D and os.path.exists(hps.train.pretrained_s2D):
        logging.info(f"Loading pretrained SoVITS D from {hps.train.pretrained_s2D}")
        pretrained_state = torch.load(hps.train.pretrained_s2D, map_location="cpu", weights_only=False)
        if "weight" in pretrained_state:
            net_d.load_state_dict(pretrained_state["weight"], strict=False)
        else:
            net_d.load_state_dict(pretrained_state, strict=False)
        logging.info("Loaded pretrained SoVITS D")
    elif existing_ckpts:
        logging.info("Skipping pretrained S2D load (resuming from existing ckpt)")
    
    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
    
    # Step schedulers to resume from correct point
    for _ in range(epoch_str - 1):
        scheduler_g.step()
        scheduler_d.step()
    
    # Training loop
    batch_size = hps.train.batch_size
    log_interval = hps.train.log_interval
    save_interval = 1000  # save every 1000 steps
    
    logging.info(f"=== Starting training: {hps.train.epochs} epochs ===")
    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size}, Log interval: {log_interval}")
    import sys; sys.stdout.flush(); sys.stderr.flush()
    
    for epoch in range(epoch_str, hps.train.epochs + 1):
        net_g.train()
        net_d.train()
        logging.info(f"Epoch {epoch} started")
        import sys; sys.stdout.flush()
        
        for batch_idx, data in enumerate(train_loader):
            print(f'Batch {batch_idx}: got data', flush=True)
            if hps.version in {"v2Pro", "v2ProPlus"}:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
            else:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data
            print(f'Batch {batch_idx}: unpacked', flush=True)
            
            # Move to MPS
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            text, text_lengths = text.to(device), text_lengths.to(device)
            if hps.version in {"v2Pro", "v2ProPlus"}:
                sv_emb = sv_emb.to(device)
            
            # Forward pass - Generator
            if hps.version in {"v2Pro", "v2ProPlus"}:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask, 
                 (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths, sv_emb
                )
            else:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask,
                 (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths
                )
            
            # Mel spectrogram
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
            
            # Discriminator update
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
            
            optim_d.zero_grad()
            loss_disc_all.backward()
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            optim_d.step()
            
            # Generator update
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
            
            optim_g.zero_grad()
            loss_gen_all.backward()
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            optim_g.step()
            
            global_step += 1
            
            if global_step % 10 == 0:  # Log every 10 steps for monitoring
                import sys
                sys.stdout.flush()
                lr = optim_g.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"step={global_step} "
                    f"loss_G={loss_gen_all.item():.4f} "
                    f"loss_D={loss_disc_all.item():.4f} "
                    f"loss_mel={loss_mel.item():.4f} "
                    f"lr={lr:.6f}"
                )
            
            if global_step % save_interval == 0:
                save_checkpoint(epoch, net_g, net_d, optim_g, optim_d)
        
        scheduler_g.step()
        scheduler_d.step()
        logging.info(f"=== Epoch {epoch} completed ===")
        
        # Save at end of epoch
        save_checkpoint(epoch, net_g, net_d, optim_g, optim_d)
    
    logging.info("=== Training completed ===")
    save_checkpoint("final", net_g, net_d, optim_g, optim_d)

def save_checkpoint(epoch, net_g, net_d, optim_g, optim_d):
    """Save G in inference-compatible format via savee(); D as raw torch.save (only used to resume training)."""
    ckpt_dir = hps.s2_ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save G via savee() so inference_webui auto-detects v2Pro and loads correctly.
    # savee() writes to f"{hps.save_weight_dir}/{name}.pth", so we set save_weight_dir
    # to ckpt_dir for this call (mirrors GPT-SoVITS upstream behavior).
    from GPT_SoVITS.process_ckpt import savee
    prev_save_weight_dir = getattr(hps, "save_weight_dir", None)
    hps.save_weight_dir = ckpt_dir
    try:
        result = savee(
            net_g.state_dict(),
            f"s2_G_{epoch}",
            epoch,
            global_step,
            hps,
            model_version=getattr(hps.model, "version", "v2Pro"),
        )
        if result != "Success.":
            logging.warning(f"savee() returned non-success for G: {result}")
    finally:
        if prev_save_weight_dir is not None:
            hps.save_weight_dir = prev_save_weight_dir

    # Save FULL G state for resume — savee() drops enc_q and opt_g, which
    # are needed to continue training without warming up from scratch.
    # Naming: s2_full_NN.pth (raw torch.save, NOT inference-compatible).
    ckpt_path_full = os.path.join(ckpt_dir, f"s2_full_{epoch}.pth")
    torch.save({
        "model": net_g.state_dict(),       # includes enc_q
        "opt_g": optim_g.state_dict(),     # optimizer state
        "step": global_step,
        "epoch": epoch,
    }, ckpt_path_full)

    # Save D in raw torch.save format — D is only needed to resume training,
    # not for inference, so format compatibility doesn't matter here.
    ckpt_path_d = os.path.join(ckpt_dir, f"s2_D_{epoch}.pth")
    torch.save({
        "model": net_d.state_dict(),
        "opt_d": optim_d.state_dict(),
    }, ckpt_path_d)

    logging.info(f"Saved: s2_G_{epoch}.pth (savee/v2Pro), s2_full_{epoch}.pth, s2_D_{epoch}.pth")

if __name__ == "__main__":
    import torch.nn.functional as F
    main()
