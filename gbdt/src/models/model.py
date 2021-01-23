import pathlib
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Union


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self, run_fold_name: str, params: dict, model_output_dir: pathlib.PosixPath
    ) -> None:
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None
        self.model_output_dir = model_output_dir
        self.iteration = 0

    @abstractmethod
    def train(
        self,
        x_trn: pd.DataFrame,
        y_trn: Union[pd.Series, np.ndarray],
        x_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def save_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

