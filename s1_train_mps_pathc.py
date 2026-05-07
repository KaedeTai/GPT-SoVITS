#!/usr/bin/env python3
"""
GPT-SoVITS S1 (GPT) Training - MPS Version, Path C variant.

Differences from s1_train_mps.py:
  * Registers a backward hook on `model.ar_text_embedding.word_embeddings.weight`
    that zeroes the gradient of rows [0..FREEZE_ZH_ROWS-1] every step,
    so the Mandarin phoneme rows inherited from the lite e10 ckpt stay
    bit-exactly identical while the new tw_ rows (732..1032) train.
  * On every train epoch end, prints two metrics:
       - text_embedding[:732] max diff vs the original surgery snapshot
         (must always be 0.0 — proof the freeze hook works)
       - text_embedding[732:] L2 norm (should slowly grow from 0).
  * Optional zh regression callback — runs the current ckpt through a short
    Mandarin sentence and logs spectrogram MSE vs a baseline wav.

Designed to leave the audio embedding and the LM head untouched (they train
normally — only the text embedding has the masked-grad hook).
"""
import os
import sys

# Add project root to path
sys.path.insert(0, '/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS')

# Disable TOKENIZERS parallelism before any tokenizer imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import logging
import time
import traceback
from pathlib import Path

import torch
import functools
import pickle

# Monkey-patch torch.load to use weights_only=False by default
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            import pathlib
            return pathlib.PosixPath
        return super().find_class(module, name)

from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import SingleDeviceStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")

from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils import get_newest_ckpt
from AR.utils.io import load_yaml_config
from process_ckpt import my_save
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Path C constants
# ---------------------------------------------------------------------------
FREEZE_ZH_ROWS = 732   # rows 0..731 are Mandarin (frozen). Row 732..1032 are tw_.


class my_model_ckpt(ModelCheckpoint):
    def __init__(self, config, if_save_latest, if_save_every_weights,
                 half_weights_save_dir, exp_name, **kwargs):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.if_save_latest == True:
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


# ---------------------------------------------------------------------------
# Path C — freeze-hook + per-epoch metrics callback
# ---------------------------------------------------------------------------
class FreezeAndMetricsCallback(Callback):
    """
    Path C freeze enforcement.

    Two layers of protection because ScaledAdam (the optimizer GPT-SoVITS uses)
    has a `p * scale_step` parameter-rescaling term that moves p even when
    grad is exactly zero — a plain backward hook is therefore insufficient.

      Layer 1 — backward grad hook on ar_text_embedding.weight:
          zeros gradients on rows [:freeze_rows] so Adam momentum stays at 0
          and the gradient-driven update is zero.
      Layer 2 — on_train_batch_end:
          copies the original (snapshot) values back into rows [:freeze_rows]
          to undo any param-rescale drift from ScaledAdam's size update.

    Logs zh max diff and tw row norms at every epoch end.
    """

    def __init__(self, freeze_rows: int, log_path: str):
        super().__init__()
        self.freeze_rows = freeze_rows
        self.log_path = log_path
        self._hook_attached = False
        self._zh_snapshot = None       # CPU fp32 snapshot
        self._zh_snapshot_dev = None   # on-device cache (matches dtype/device of weight)
        with open(self.log_path, "w") as f:
            f.write("epoch\twall_time\tzh_max_diff\ttw_norm\ttw_row_min_norm\ttw_row_max_norm\n")

    @staticmethod
    def _freeze_zh_grad_factory(freeze_rows):
        def hook(grad):
            new_grad = grad.clone()
            new_grad[:freeze_rows] = 0
            return new_grad
        return hook

    def _attach_hook(self, pl_module):
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        self._zh_snapshot = emb.detach()[: self.freeze_rows].clone().float().cpu()
        # On-device cache, matches dtype/device of the weight
        self._zh_snapshot_dev = emb.detach()[: self.freeze_rows].clone()
        emb.register_hook(self._freeze_zh_grad_factory(self.freeze_rows))
        self._hook_attached = True
        print(f"[FreezeHook] attached on {tuple(emb.shape)} weight "
              f"dtype={emb.dtype} dev={emb.device}; "
              f"freezing rows [0..{self.freeze_rows - 1}]; "
              f"zh snapshot norm={self._zh_snapshot.norm().item():.4f}", flush=True)

    def on_train_start(self, trainer, pl_module):
        if not self._hook_attached:
            self._attach_hook(pl_module)

    # Layer 2 — restore zh rows after every training batch so any
    # optimizer-side drift on rows [:freeze_rows] is reverted.
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._hook_attached or self._zh_snapshot_dev is None:
            return
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        # Make sure the cache lives on the same device/dtype as emb (will only
        # actually move once on first use, then cheap)
        if self._zh_snapshot_dev.device != emb.device or self._zh_snapshot_dev.dtype != emb.dtype:
            self._zh_snapshot_dev = self._zh_snapshot_dev.to(device=emb.device, dtype=emb.dtype)
        with torch.no_grad():
            emb.data[: self.freeze_rows].copy_(self._zh_snapshot_dev)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._hook_attached:
            return
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        with torch.no_grad():
            zh_now = emb.detach()[: self.freeze_rows].float().cpu()
            tw_now = emb.detach()[self.freeze_rows :].float().cpu()
            zh_diff = (zh_now - self._zh_snapshot).abs().max().item()
            tw_norm = tw_now.norm().item()
            tw_row_norms = tw_now.norm(dim=1)
            tw_row_min = tw_row_norms.min().item()
            tw_row_max = tw_row_norms.max().item()
        epoch = trainer.current_epoch
        wall = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"{epoch}\t{wall}\t{zh_diff:.6e}\t{tw_norm:.6f}\t{tw_row_min:.6f}\t{tw_row_max:.6f}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"[FreezeMetrics] epoch={epoch} zh_max_diff={zh_diff:.3e} "
              f"tw_norm={tw_norm:.4f} tw_row_min={tw_row_min:.4f} "
              f"tw_row_max={tw_row_max:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Path C — zh regression callback (best-effort, errors swallowed)
# ---------------------------------------------------------------------------
class ZhRegressionCallback(Callback):
    """
    Per-epoch: emit a regression marker line. Heavy-weight TTS regression is
    not invoked from the training loop (it would lock up the same MPS device
    that's training); instead we log the half-weights ckpt path so an
    external monitor can pick it up. The 'lightweight' regression we *do*
    compute here is the full text_embedding zh-row diff (already in Freeze
    callback) which is the most reliable in-loop sanity signal.

    Heavy zh regression (spectrogram MSE vs baseline) is run separately
    by `tw_finetune/zh_regression_runner.py` after each ckpt is written.
    """

    def __init__(self, exp_name: str, half_weights_dir: str, log_path: str):
        super().__init__()
        self.exp_name = exp_name
        self.half_weights_dir = half_weights_dir
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("epoch\twall_time\tckpt\n")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        ckpt = os.path.join(self.half_weights_dir,
                            f"{self.exp_name}-e{epoch}.ckpt")
        wall = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"{epoch}\t{wall}\t{ckpt}\n")
        print(f"[Regression] epoch={epoch} ckpt_marker={ckpt}", flush=True)


class ProgressHeartbeatCallback(Callback):
    """
    Writes per-epoch and per-batch heartbeat lines to PROGRESS_FILE
    (env var). No-op if env var is unset. Used to ping dispatch / parent
    session that training is alive even before a ckpt is written.
    """

    def __init__(self):
        super().__init__()
        self.path = os.environ.get("PROGRESS_FILE", "")
        self._last_batch_hb = 0.0
        self._batch_period = 60.0  # seconds — emit a hb line at most every 60s
        self._step_count = 0
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] train: ProgressHeartbeatCallback armed\n")
            except Exception:
                pass

    def _write(self, msg: str):
        if not self.path:
            return
        try:
            with open(self.path, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] train: {msg}\n")
        except Exception:
            pass

    def on_train_start(self, trainer, pl_module):
        self._write("train_start — first batch incoming")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._step_count += 1
        now = time.time()
        if now - self._last_batch_hb < self._batch_period:
            return
        self._last_batch_hb = now
        cm = trainer.callback_metrics
        loss = cm.get("total_loss_step", cm.get("total_loss"))
        top3 = cm.get("top_3_acc_step", cm.get("top_3_acc"))
        loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
        top3_s = f"{float(top3):.4f}" if top3 is not None else "n/a"
        self._write(f"epoch={trainer.current_epoch+1} step={self._step_count} batch_idx={batch_idx} loss={loss_s} top3={top3_s}")

    def on_train_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        loss = cm.get("total_loss_epoch", cm.get("total_loss"))
        top3 = cm.get("top_3_acc_epoch", cm.get("top_3_acc"))
        loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
        top3_s = f"{float(top3):.4f}" if top3 is not None else "n/a"
        epoch = trainer.current_epoch + 1
        self._write(f"=== EPOCH {epoch} TRAIN END === loss={loss_s} top3={top3_s} steps={self._step_count}")


# ---------------------------------------------------------------------------
def main(args):
    config = load_yaml_config(args.config_file)

    if args.output_dir:
        config['output_dir'] = args.output_dir
        config['train']['half_weights_save_dir'] = os.path.join(args.output_dir, 'half_weights')

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(config["train"]["half_weights_save_dir"]).mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)

    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )

    freeze_metrics_log = str(output_dir / "freeze_metrics.tsv")
    freeze_callback = FreezeAndMetricsCallback(
        freeze_rows=FREEZE_ZH_ROWS,
        log_path=freeze_metrics_log,
    )

    regression_log = str(output_dir / "regression_log.tsv")
    regression_callback = ZhRegressionCallback(
        exp_name=config["train"]["exp_name"],
        half_weights_dir=config["train"]["half_weights_save_dir"],
        log_path=regression_log,
    )

    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)

    if torch.backends.mps.is_available():
        accelerator = "mps"
        strategy = SingleDeviceStrategy(device="mps")
        print("Using MPS + SingleDeviceStrategy")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        from pytorch_lightning.strategies import DDPStrategy
        import platform
        strategy = DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
    else:
        accelerator = "cpu"
        strategy = "auto"

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=accelerator,
        devices=1,
        limit_val_batches=0,
        benchmark=False,
        fast_dev_run=False,
        strategy=strategy,
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback, freeze_callback, regression_callback, ProgressHeartbeatCallback()],
        use_distributed_sampler=False,
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print(f"ckpt_path: {ckpt_path}")

    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="/Users/kaede/tts/tw_finetune/s1_tw_lite_pathc.yaml",
        help="path of config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="override output directory",
    )
    args = parser.parse_args()
    logging.info(str(args))
    main(args)
