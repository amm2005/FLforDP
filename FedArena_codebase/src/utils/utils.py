import sys
import os
import signal
import subprocess
from hydra.core.hydra_config import HydraConfig


def handle_main_process_sigterm(signum, frame, trainer):
    """Handle SIGTERM in the main process by stopping all child processes."""
    print("Parent process received SIGTERM. Terminate all child processes...")
    trainer.manager.stop_train()
    sys.exit(0)


def handle_client_process_sigterm(signum, frame, rank):
    """Handle SIGTERM in a child process by forwarding it to the parent."""
    print(f"Child process {rank} received SIGTERM. Send it to parent...")

    os.kill(os.getppid(), signal.SIGTERM)


def get_repository_info(do_print: bool = False) -> dict:
    """Return commit hash and repo name for the current repository state."""
    git_info = {
        "commit_hash": subprocess.run(
            "git rev-parse --verify HEAD".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .replace("\n", ""),
        "repo_name": subprocess.run(
            "git rev-parse --show-toplevel".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .split("/")[-1]
        .replace("\n", ""),
        "project_name": subprocess.run(
            "git rev-parse --show-toplevel".split(), capture_output=True
        )
        .stdout.decode("utf-8")
        .split("/")[-1]
        .replace("\n", ""),
    }
    return git_info


def get_run_command():
    script_name = sys.argv[0]
    args = " ".join(sys.argv[1:])
    run_dir = HydraConfig.get().run.dir
    run_command = f"python {script_name} {args} > {run_dir}/train_log.txt &"
    return run_command


def create_model_info(model_state, metrics, checkpoint_path, cfg):

    model_info = {
        "git": get_repository_info(),
        "run_command": get_run_command(),
        "model": model_state,
        "metrics": {"metrics": metrics[0], "threshold": metrics[1]},
        "config_file": cfg,
    }

    return model_info
