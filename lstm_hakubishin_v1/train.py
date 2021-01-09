import yaml
import argparse
from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything


def run(config: dict) -> None:
    seed_everything(71)
    train_test_set = load_train_test_set(config)
    print(train_test_set.shape)
    import pdb; pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    run(config)


if __name__ == "__main__":
    main()
