import os
import sys
import hydra


def redirect_stdout_to_log():
    main_log_file = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/output.txt"
    )

    f = open(main_log_file, "w")
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = f
    sys.stderr = f

    orig_cwd = hydra.utils.get_original_cwd()
    link_path = os.path.join(orig_cwd, "output", "file.txt")
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(main_log_file, link_path)

    print("Information about files:")
    print(f"File to logging: {main_log_file}")
    print(f"Link file: {link_path}")
    print()
