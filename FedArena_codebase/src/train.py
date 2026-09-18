import sys
from pathlib import Path

import hydra
import signal
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.utils import handle_main_process_sigterm
from utils.logging_utils import redirect_stdout_to_log


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    redirect_stdout_to_log()

    # cuDNN autotune + TF32 for fixed-size image inputs.
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[DEBUG] Loading train dataset...", flush=True)
    df = instantiate(cfg.train_dataset, cfg=cfg, mode="train", _recursive_=False)
    print("[DEBUG] Dataset loaded, init federated...", flush=True)

    trainer = instantiate(cfg.federated_method, _recursive_=False)
    trainer._init_federated(cfg, df)

    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_main_process_sigterm(signum, frame, trainer),
    )

    trainer.begin_train()


if __name__ == "__main__":
    train()
