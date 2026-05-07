#!/usr/bin/env python3
"""
GPT-SoVITS S1 (GPT) Training - MPS Version, Path D variant.

Path D = "freeze EVERYTHING except text_embedding rows 732..1032".

Why:
  Path C (transformer + audio_emb + LM head trainable; only text_emb rows
  frozen via grad-mask) catastrophically forgot Mandarin within 1 epoch
  (mel_mse 700-1300 vs zh-v1.0 baseline). Even mixing 24% zh data (Path C
  mix) couldn't keep mel_mse below the 200 threshold.

  Hypothesis: the freeze hook on text_embedding works perfectly
  (zh_max_diff bit-exactly 0.0 across all epochs of path-c). What was
  drifting was the transformer blocks + audio embedding + LM head.

  Path D removes that drift entirely:
    - ALL parameters: requires_grad = False
    - EXCEPT model.ar_text_embedding.word_embeddings.weight
    - And within that, rows 0..731 are kept bit-exact (Layer 1: grad mask
      hook; Layer 2: per-batch snapshot restore — same belt-and-braces
      as Path C).

  Result: Mandarin output is mathematically identical to the surgery
  ckpt (because the only thing that changed is rows 732..1032 of the
  text embedding, which Mandarin phoneme IDs never index). New
  Taiwanese tokens learn to map into the existing phoneme→audio space
  the lite e10 ckpt already mastered.

Trainable parameter count: 301 rows * 512 dims = 154,112 fp32 params.
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
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam
from AR.utils import get_newest_ckpt
from AR.utils.io import load_yaml_config
from process_ckpt import my_save
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Path D constants
# ---------------------------------------------------------------------------
FREEZE_ZH_ROWS = 732   # rows 0..731 are Mandarin (frozen). Row 732..1032 are tw_.


# ---------------------------------------------------------------------------
# Path D — Lightning module subclass that:
#   1. Freezes everything except the text-embedding weight
#   2. Builds an optimizer that only contains that single tensor
# ---------------------------------------------------------------------------
class Text2SemanticLightningModulePathD(Text2SemanticLightningModule):
    """
    Subclass with frozen network. Only `model.ar_text_embedding.word_embeddings.weight`
    is trainable. The optimizer is built with that single parameter.
    """

    def _freeze_all_except_text_embedding(self):
        unfrozen_names = []
        emb_param = self.model.ar_text_embedding.word_embeddings.weight
        for name, p in self.named_parameters():
            if p is emb_param:
                p.requires_grad = True
                unfrozen_names.append(name)
            else:
                p.requires_grad = False
        # Sanity print
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[PathD] froze {n_total - n_train:,} params, "
            f"{n_train:,} remain trainable; trainable tensor names={unfrozen_names}",
            flush=True,
        )
        return [emb_param], unfrozen_names

    def configure_optimizers(self):
        params, names = self._freeze_all_except_text_embedding()
        # ScaledAdam expects parameters_names = list-of-list, one inner list per param group
        lm_opt = ScaledAdam(
            params,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=[names],
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }


# ---------------------------------------------------------------------------
# ModelCheckpoint copy from path-c (unchanged)
# ---------------------------------------------------------------------------
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
                if self.if_save_latest:
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except Exception:
                            pass
                if self.if_save_every_weights:
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
# Path D — freeze-hook + per-epoch metrics callback
# ---------------------------------------------------------------------------
class FreezeAndMetricsCallback(Callback):
    """
    Path D row-mask enforcement INSIDE the only trainable tensor.

    Even though the param is trainable, we still need rows 0..731 to remain
    bit-exact (Mandarin must not move). Two-layer protection:

      Layer 1 — backward grad hook on ar_text_embedding.weight:
          zeros gradients on rows [:freeze_rows] so Adam momentum stays at 0.
      Layer 2 — on_train_batch_end:
          copies the original snapshot back into rows [:freeze_rows] so any
          ScaledAdam param-rescale drift is reverted.

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
        self._zh_snapshot_dev = emb.detach()[: self.freeze_rows].clone()
        emb.register_hook(self._freeze_zh_grad_factory(self.freeze_rows))
        self._hook_attached = True
        print(
            f"[FreezeHook] attached on {tuple(emb.shape)} weight "
            f"dtype={emb.dtype} dev={emb.device}; "
            f"freezing rows [0..{self.freeze_rows - 1}]; "
            f"zh snapshot norm={self._zh_snapshot.norm().item():.4f}",
            flush=True,
        )

    def on_train_start(self, trainer, pl_module):
        if not self._hook_attached:
            self._attach_hook(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._hook_attached or self._zh_snapshot_dev is None:
            return
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
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
            tw_now = emb.detach()[self.freeze_rows:].float().cpu()
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
        print(
            f"[FreezeMetrics] epoch={epoch} zh_max_diff={zh_diff:.3e} "
            f"tw_norm={tw_norm:.4f} tw_row_min={tw_row_min:.4f} "
            f"tw_row_max={tw_row_max:.4f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Path D — zh regression callback (best-effort marker only)
# ---------------------------------------------------------------------------
class ZhRegressionCallback(Callback):
    """
    Per-epoch: emit a regression marker line. Heavyweight TTS regression is
    run by the watcher process so as not to lock up the training MPS device.
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
    """Per-epoch and per-batch heartbeat lines to PROGRESS_FILE (env var)."""

    def __init__(self):
        super().__init__()
        self.path = os.environ.get("PROGRESS_FILE", "")
        self._last_batch_hb = 0.0
        self._batch_period = 60.0
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
        self._write(
            f"epoch={trainer.current_epoch + 1} step={self._step_count} "
            f"batch_idx={batch_idx} loss={loss_s} top3={top3_s}"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        loss = cm.get("total_loss_epoch", cm.get("total_loss"))
        top3 = cm.get("top_3_acc_epoch", cm.get("top_3_acc"))
        loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
        top3_s = f"{float(top3):.4f}" if top3 is not None else "n/a"
        epoch = trainer.current_epoch + 1
        self._write(
            f"=== EPOCH {epoch} TRAIN END === loss={loss_s} top3={top3_s} steps={self._step_count}"
        )


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

    # IMPORTANT: use the Path-D subclass, not the original
    model: Text2SemanticLightningModulePathD = Text2SemanticLightningModulePathD(config, output_dir)

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
        default="/Users/kaede/tts/tw_finetune/s1_tw_lite_pathd.yaml",
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
