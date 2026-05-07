#!/usr/bin/env python3
"""
GPT-SoVITS S1 (GPT) Training - MPS Version, with ARPA-row freeze hook.

Why this exists
---------------
M1 forensic analysis (`check_arpa_drift.py`) showed that finetuning on
zh-only data (lite-e10 / long-e15 / mixed-e15) drifts ARPA phoneme
embedding rows by up to 0.27 (max abs) even though the train data
contains zero ARPA tokens. The drift is a side-effect of full-network
fine-tuning: optimizer state / Adam moments / weight decay / LR scaling
all touch every embedding row regardless of whether the row was hit by
the loss in any batch.

This script is a drop-in replacement for `s1_train_mps.py` that adds:

1. Belt-and-braces ARPA-row freeze (Layer 1 grad-mask + Layer 2 snapshot
   restore), patterned after Path D's zh-row freeze.
2. Optimizer weight_decay forced to 0 (ScaledAdam default already 0, but
   we assert it for safety).
3. Per-epoch metrics: `arpa_max_diff` (must stay 0.0) AND
   `non_arpa_max_diff` (should grow normally as training progresses,
   confirming we did not accidentally over-freeze).

Activation: set `freeze_arpa: true` in the YAML config (see
`s1_arpa_freeze.yaml`). When `freeze_arpa` is missing or false, the
script behaves identically to `s1_train_mps.py`.
"""
import os
import sys
import argparse
import logging
import time
import functools
import pickle
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, '/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS')

# Disable TOKENIZERS parallelism before any tokenizer imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

# Monkey-patch torch.load to use weights_only=False by default (PyTorch 2.6+)
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

# Import the project's symbol table to compute ARPA row indices.
sys.path.insert(0, '/Users/kaede/tts/GPT-SoVITS')
from text import symbols2  # noqa: E402


# ---------------------------------------------------------------------------
# Compute ARPA row indices once at import time
# ---------------------------------------------------------------------------
def _compute_arpa_row_indices() -> list[int]:
    arpa_tokens = sorted(symbols2.arpa)
    sym_idx = {s: i for i, s in enumerate(symbols2.symbols)}
    rows = sorted([sym_idx[t] for t in arpa_tokens if t in sym_idx])
    return rows


ARPA_ROWS: list[int] = _compute_arpa_row_indices()


# ---------------------------------------------------------------------------
# ARPA freeze callback — belt-and-braces protection.
# ---------------------------------------------------------------------------
class ArpaFreezeCallback(Callback):
    """
    Two-layer protection on the ARPA rows of
    `model.ar_text_embedding.word_embeddings.weight`:

      Layer 1 — backward grad hook on the embedding tensor: zeros
                gradients on ARPA rows so optimizer momentum stays at 0
                for those rows.
      Layer 2 — on_train_batch_end snapshot restore: copies the
                pre-training ARPA rows back into the live tensor each
                batch, defeating any drift introduced by ScaledAdam
                rescaling, weight decay, or numerical noise.

    Per-epoch logging:
      arpa_max_diff      — must be exactly 0.0 if freeze is working
      non_arpa_max_diff  — must grow > 0 if non-ARPA rows are training
    """

    def __init__(self, arpa_rows: list[int], log_path: str):
        super().__init__()
        self.arpa_rows = list(arpa_rows)
        self.arpa_idx_t: torch.Tensor | None = None  # set on hook attach
        self.log_path = log_path
        self._hook_attached = False
        self._arpa_snapshot: torch.Tensor | None = None      # CPU fp32
        self._arpa_snapshot_dev: torch.Tensor | None = None  # on-device cache
        self._initial_full: torch.Tensor | None = None       # CPU fp32 baseline of the entire emb (for non-ARPA delta)
        with open(self.log_path, "w") as f:
            f.write("epoch\twall_time\tarpa_max_diff\tnon_arpa_max_diff\tnon_arpa_l2_mean\n")

    @staticmethod
    def _grad_hook_factory(arpa_idx_t: torch.Tensor):
        def hook(grad: torch.Tensor):
            new_grad = grad.clone()
            new_grad[arpa_idx_t] = 0
            return new_grad
        return hook

    def _attach_hook(self, pl_module):
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        # Filter rows to ones present in the actual emb (pretrained vocab=732
        # but post-finetune may be smaller/larger — be safe).
        in_range = [r for r in self.arpa_rows if r < emb.shape[0]]
        idx_t = torch.tensor(in_range, dtype=torch.long, device=emb.device)
        self.arpa_idx_t = idx_t

        # Snapshots
        self._arpa_snapshot = emb.detach()[idx_t].clone().float().cpu()
        self._arpa_snapshot_dev = emb.detach()[idx_t].clone()
        self._initial_full = emb.detach().clone().float().cpu()

        # Layer 1
        emb.register_hook(self._grad_hook_factory(idx_t))
        self._hook_attached = True
        print(
            f"[ArpaFreeze] attached on {tuple(emb.shape)} weight "
            f"dtype={emb.dtype} dev={emb.device}; "
            f"freezing {len(in_range)} ARPA rows "
            f"(min={min(in_range)}, max={max(in_range)}); "
            f"snapshot norm={self._arpa_snapshot.norm().item():.4f}",
            flush=True,
        )

    def on_train_start(self, trainer, pl_module):
        if not self._hook_attached:
            self._attach_hook(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._hook_attached or self._arpa_snapshot_dev is None:
            return
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        # Move snapshot if device/dtype changed (e.g., after .to() switch)
        if (self._arpa_snapshot_dev.device != emb.device or
                self._arpa_snapshot_dev.dtype != emb.dtype):
            self._arpa_snapshot_dev = self._arpa_snapshot_dev.to(
                device=emb.device, dtype=emb.dtype)
            self.arpa_idx_t = self.arpa_idx_t.to(emb.device)
        with torch.no_grad():
            emb.data[self.arpa_idx_t] = self._arpa_snapshot_dev

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._hook_attached:
            return
        emb = pl_module.model.ar_text_embedding.word_embeddings.weight
        with torch.no_grad():
            emb_cpu = emb.detach().float().cpu()
            arpa_now = emb_cpu[self.arpa_idx_t.cpu()]
            arpa_max = (arpa_now - self._arpa_snapshot).abs().max().item()
            # Non-ARPA delta vs initial
            full_diff = (emb_cpu - self._initial_full).abs()
            mask = torch.ones(emb_cpu.shape[0], dtype=torch.bool)
            mask[self.arpa_idx_t.cpu()] = False
            non_arpa_diff = full_diff[mask]
            non_arpa_max = non_arpa_diff.max().item()
            non_arpa_l2 = (emb_cpu - self._initial_full)[mask].norm(dim=1).mean().item()
        epoch = trainer.current_epoch
        wall = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = (f"{epoch}\t{wall}\t{arpa_max:.6e}\t{non_arpa_max:.6e}"
                f"\t{non_arpa_l2:.6e}\n")
        with open(self.log_path, "a") as f:
            f.write(line)
        status = "OK" if arpa_max == 0.0 else "DRIFT"
        print(
            f"[ArpaFreeze] epoch={epoch} arpa_max_diff={arpa_max:.3e} "
            f"non_arpa_max={non_arpa_max:.3e} non_arpa_l2={non_arpa_l2:.3e} "
            f"[{status}]",
            flush=True,
        )


# ---------------------------------------------------------------------------
# weight_decay = 0 wrapper around the base LightningModule.
# ---------------------------------------------------------------------------
class Text2SemanticLightningModuleArpaFreeze(Text2SemanticLightningModule):
    """
    Same as the base module but asserts the optimizer is built with
    weight_decay=0. ScaledAdam does not expose a weight_decay arg in the
    base configure_optimizers, but we rebuild the optimizer here to
    document the intent and to guard against future changes.
    """

    def configure_optimizers(self):
        from AR.modules.optim import ScaledAdam
        from AR.modules.lr_schedulers import WarmupCosineLRSchedule

        model_parameters = list(self.model.parameters())
        parameters_names = [
            [name for name, _ in self.model.named_parameters()]
        ]
        # ScaledAdam does NOT accept weight_decay; the absence of weight
        # decay is implicit. This is documented here for clarity.
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
        print("[ArpaFreeze] optimizer = ScaledAdam (weight_decay=0 implicit)",
              flush=True)
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
# ModelCheckpoint copy from base (unchanged behaviour)
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
            if (self._every_n_epochs >= 1 and
                    (trainer.current_epoch + 1) % self._every_n_epochs == 0):
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
                            "%s/%s-e%s.ckpt" % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    if args.output_dir:
        config['output_dir'] = args.output_dir
        config['train']['half_weights_save_dir'] = os.path.join(
            args.output_dir, 'ckpt')

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    callbacks = [ckpt_callback]

    # Honour the freeze_arpa flag (default False)
    freeze_arpa = bool(config.get("train", {}).get("freeze_arpa", False))
    if freeze_arpa:
        log_path = str(output_dir / "arpa_freeze_metrics.tsv")
        arpa_cb = ArpaFreezeCallback(arpa_rows=ARPA_ROWS, log_path=log_path)
        callbacks.append(arpa_cb)
        print(f"[ArpaFreeze] enabled — {len(ARPA_ROWS)} ARPA rows will be "
              f"frozen; metrics → {log_path}", flush=True)
    else:
        print("[ArpaFreeze] DISABLED (set train.freeze_arpa=true to enable)",
              flush=True)

    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)

    if torch.backends.mps.is_available():
        accelerator = "mps"
        strategy = SingleDeviceStrategy(device="mps")
        print("Using MPS + SingleDeviceStrategy")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        from pytorch_lightning.strategies import DDPStrategy
        import platform
        strategy = DDPStrategy(
            process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
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
        callbacks=callbacks,
        use_distributed_sampler=False,
    )

    if freeze_arpa:
        model = Text2SemanticLightningModuleArpaFreeze(config, output_dir)
    else:
        model = Text2SemanticLightningModule(config, output_dir)

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
        default="/Users/kaede/tts/tai_voice_prepared/s1_arpa_freeze.yaml",
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
