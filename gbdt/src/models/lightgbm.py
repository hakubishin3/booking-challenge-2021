import numpy as np
import pandas as pd
import lightgbm as lgb
from .model import BaseModel
from src.utils import Pkl
from typing import Union


class ModelLightGBM(BaseModel):
    def train(
        self,
        x_trn: pd.DataFrame,
        y_trn: Union[pd.Series, np.ndarray],
        x_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray],
    ) -> None:
        validation_flg = x_val is not None

        # Setting model parameters
        lgb_model_params = self.params["model_params"]
        lgb_train_params = self.params["train_params"]

        # Training
        if validation_flg:
            self.model = lgb.LGBMClassifier(**lgb_model_params)
            self.model.fit(
                x_trn,
                y_trn,
                eval_set=[(x_trn, y_trn), (x_val, y_val)],
                eval_names=["train", "valid"],
                **lgb_train_params,
            )
        else:
            self.model = lgb.LGBMClassifier(**lgb_model_params)
            self.model.fit(
                x_trn,
                y_trn,
                eval_set=[(x_trn, y_trn)],
                eval_names=["train"],
                **lgb_train_params,
            )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_best_iteration(self) -> int:
        return self.model.best_iteration_

    def save_model(self) -> None:
        model_path = self.model_output_dir / f"{self.run_fold_name}.pkl"
        Pkl.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = self.model_output_dir / f"{self.run_fold_name}.pkl"
        self.model = Pkl.load(model_path)

