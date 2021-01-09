import yaml
import torch
import argparse
from pathlib import Path
from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config: dict) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    log("Load train and test set:")
    train_test_set = load_train_test_set(config)
    log(f"{train_test_set.shape}")

    import pdb; pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    output_path = Path(config["output_dir_path"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    run(config)


if __name__ == "__main__":
    main()
