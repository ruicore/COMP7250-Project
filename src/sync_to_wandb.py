import yaml
from pathlib import Path
import subprocess

PROJECT = "COMP7250-Project"
LOG_DIR = "../running/0412/logs"
ENTITY = 'ruihe'


def generate_config_file(directory, config):
    config_path = directory / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Generated config file at {config_path}")


def sync_to_wandb():
    for trick_dir in Path(LOG_DIR).iterdir():
        if not trick_dir.is_dir():
            continue

        for trick_value_dir in trick_dir.iterdir():
            if not trick_value_dir.is_dir():
                continue

            run_name = f"ResNet18_{trick_dir.name}_{trick_value_dir.name}".replace(" ", "")
            print('Processing run:', run_name)

            if not any(trick_value_dir.glob("events.out.tfevents*")):
                print(f"⚠️ Skipping {run_name}: No TensorBoard logs found")
                continue
            generate_config_file(trick_value_dir, {
                "model": run_name,
            })

            cmd = (
                f"wandb sync --sync-tensorboard --include-offline --mark-synced "
                f"-p {PROJECT} "
                f"--id '{run_name}' "
            )
            if ENTITY:
                cmd += f"-e {ENTITY} "
            cmd += f'"{trick_value_dir}"'

            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Synced {run_name}")


if __name__ == "__main__":
    sync_to_wandb()
