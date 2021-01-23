import os
import json
import codecs
import logging
import random
import pathlib
import numpy as np
import joblib
from typing import Any, Union, Dict
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def adversarial_validation(
    x_train: pd.DataFrame, x_test: pd.DataFrame, config: dict
) -> Dict[str, Dict[str, float]]:
    feature_name = x_train.columns
    train_adv = x_train.copy()
    test_adv = x_test.copy()
    train_adv["target"] = 0
    test_adv["target"] = 1
    train_test_adv = pd.concat([train_adv, test_adv], axis=0, sort=False).reset_index(
        drop=True
    )
    train_set, val_set = train_test_split(
        train_test_adv, test_size=0.33, random_state=71, shuffle=True
    )
    x_train_adv = train_set[feature_name]
    y_train_adv = train_set["target"]
    x_val_adv = val_set[feature_name]
    y_val_adv = val_set["target"]
    train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    val_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)
    clf = lgb.train(
        config["adversarial_validation"]["model_params"],
        train_lgb,
        valid_sets=[train_lgb, val_lgb],
        valid_names=["train", "valid"],
        **config["adversarial_validation"]["train_params"],
    )
    feature_importances = pd.DataFrame(
        sorted(zip(clf.feature_importance(importance_type="gain"), feature_name)),
        columns=["value", "feature"],
    )
    feature_importances = (
        feature_importances.set_index("feature")
        .sort_values(by="value", ascending=False)
        .head(30)
        .to_dict()["value"]
    )
    evals_result = {
        "adversarial_validation": {
            "train_auc": clf.best_score["train"]["auc"],
            "val_auc": clf.best_score["valid"]["auc"],
            "feature_importances": feature_importances,
        }
    }
    return evals_result


def seed_everything(seed: int = 71, gpu_mode: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class MyEncoder(json.JSONEncoder):
    """encode numpy objects
    https://wtnvenga.hatenablog.com/entry/2018/05/27/113848
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def json_dump(dict_object: dict, save_path: str) -> None:
    f = codecs.open(save_path, "w", "utf-8")
    json.dump(dict_object, f, indent=4, cls=MyEncoder, ensure_ascii=False)


class Pkl(object):
    """https://github.com/ghmagazine/kagglebook/blob/master/ch04-model-interface/code/util.py"""

    @classmethod
    def dump(cls, value: Any, path: Union[pathlib.PosixPath, str]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path: Union[pathlib.PosixPath, str]) -> None:
        return joblib.load(path)


def get_logger(
    module_name: str = None,
    save_path: Union[pathlib.PosixPath, str] = None,
) -> logging.Logger:
    logger = logging.getLogger(module_name)
    formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    logger.setLevel(logging.DEBUG)

    if save_path is None:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        handler = logging.FileHandler(save_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

