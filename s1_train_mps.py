#!/usr/bin/env python3
"""
GPT-SoVITS S1 (GPT) Training - MPS Version
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
from pathlib import Path

import torch
import functools
import pickle

# Monkey-patch torch.load to use weights_only=False by default (fix for PyTorch 2.6+ with PL checkpoints containing pathlib objects)
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

# Also patch pickle.Unpickler to allow PosixPath
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            import pathlib
            return pathlib.PosixPath
        return super().find_class(module, name)

from pytorch_lightning import Trainer, seed_everything
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


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
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


def main(args):
    config = load_yaml_config(args.config_file)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
        config['train']['half_weights_save_dir'] = os.path.join(args.output_dir, 'ckpt')

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
    
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    
    # MPS detection
    if torch.backends.mps.is_available():
        accelerator = "mps"
        device = "mps"
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        device = "cuda"
        print(f"Using CUDA GPU")
    else:
        accelerator = "cpu"
        device = "cpu"
        print(f"Using CPU")
    
    # For MPS, use SingleDeviceStrategy instead of DDP
    if accelerator == "mps":
        strategy = SingleDeviceStrategy(device="mps")
        print("Using SingleDeviceStrategy for MPS")
    elif accelerator == "gpu" and torch.cuda.is_available():
        # Keep DDP for CUDA
        from pytorch_lightning.strategies import DDPStrategy
        import platform
        strategy = DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
    else:
        strategy = "auto"
    
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=accelerator,
        devices=1,  # MPS doesn't support multi-device
        limit_val_batches=0,
        benchmark=False,
        fast_dev_run=False,
        strategy=strategy,
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
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
        default="/Users/kaede/tts/GPT-SoVITS/configs/s1_tai_finetune.yaml",
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
