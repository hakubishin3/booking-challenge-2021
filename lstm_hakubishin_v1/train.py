import os
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Type

from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything
from src.dataset import Dataset, collate_fn
from src.models import BookingLSTM
from src.runner import CustomRunner


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, Type[torch.nn.Module]] = {
    cls.__name__: cls
    for cls in [
        BookingLSTM,
    ]
}


def run(config: dict) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    log("Load train and test set:")
    train_test_set = load_train_test_set(config)
    log(f"{train_test_set.shape}")

    log("Preprocessing:")
    target_le = preprocessing.LabelEncoder()
    train_test_set["city_id"] = target_le.fit_transform(train_test_set["city_id"])

    train = train_test_set[train_test_set["row_num"].isnull()]
    test = train_test_set[~train_test_set["row_num"].isnull()]

    train_trips = train.groupby("utrip_id")["city_id"].apply(list)
    test_trips = test.query("city_id!=0").groupby("utrip_id")["city_id"].apply(list)

    x_train = pd.DataFrame(train_trips)
    x_test = pd.DataFrame(test_trips)

    x_train["n_trips"] = x_train["city_id"].map(lambda x: len(x))
    x_train = x_train.query("n_trips > 2").sort_values("n_trips").reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    log(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

    log("Get folds:")
    cv = StratifiedKFold(
        n_splits=config["fold"]["n_splits"],
        shuffle=config["fold"]["shuffle"],
        random_state=config["seed"],
    )
    folds = cv.split(x_train, pd.cut(x_train["n_trips"], 5, labels=False))

    log("Training:")
    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        log(f"Fold = {i_fold}")
        x_trn = x_train.loc[trn_idx, :]
        x_val = x_train.loc[val_idx, :]
        train_dataset = Dataset(x_trn, is_train=True)
        valid_dataset = Dataset(x_trn, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["params"]["bacth_size"],
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=True,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config["params"]["bacth_size"],
            num_workers=os.cpu_count(),
            pin_memory=True,
            shuffle=False,
        )
        model_cls = MODELS[config["model_name"]]
        model = model_cls(len(target_le.classes_))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6
        )
        logdir = Path(config["output_dir_path"]) / config["exp_name"] / f"fold{i_fold}"
        loaders = {"train": train_dataloader, "valid": valid_dataloader}
        runner = CustomRunner(device=DEVICE)
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=config["params"]["num_epochs"],
            verbose=True,
        )

        score = 0
        y_val = x_val['city_id'].map(lambda x: x[-1])
        for loop_i, prediction in enumerate(runner.predict_loader(
                                            loader=valid_loader,
                                            resume=f'{logdir}/checkpoints/best.pth',
                                            model=model,)):
            correct = y_val.values[loop_i] in np.argsort(prediction.cpu().numpy()[-1, :])[-4:]
            score += int(correct)
        score /= len(y_val)
        print('acc@4', score)

        pred = np.array(list(map(lambda x: x.cpu().numpy()[-1, :],
                                    runner.predict_loader(
                                        loader=test_loader,
                                        resume=f'{logdir}/checkpoints/best.pth',
                                        model=model,),)))
        print(pred.shape)
        np.save(f'y_pred_fold{fold_id}', pred)


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
