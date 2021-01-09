import yaml
import argparse
from src import log, set_out, span
from src.utils import seed_everything


def run(config: dict) -> None:
    seed_everything(71)
    print("a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    run(config)


if __name__ == "__main__":
    main()
