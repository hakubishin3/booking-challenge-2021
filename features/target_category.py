import pandas as pd
from base import BaseFeature
from typing import Optional, List, Tuple
import tempfile
import os

class TargetCategory(BaseFeature):
    def import_columns(self):
        return [
            "city_id",
            "row_number() over(partition by utrip_id order by checkin) as trip_order",
            "utrip_id",
        ]

    def make_features(self, df_train_input, df_test_input):
        col = "city_id"

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        train_target_idx = df_train_input.groupby("utrip_id")["trip_order"].idxmax()
        df_train_features["last_" + col] = df_train_input.loc[train_target_idx, :][col].values

        return df_train_features, df_test_features

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        self._logger.info(f"Running with debugging={self.debugging}")
        with tempfile.TemporaryDirectory() as tempdir:
            test_path = os.path.join(tempdir, f"{self.name}_test.ftr")
            train_path = os.path.join(tempdir, f"{self.name}_training.ftr")
            self.read_and_save_features(
                self.train_table, self.test_table, train_path, test_path,
            )
            self._upload_to_gs([train_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_output_path: str,
    ) -> None:
        df_train_input = self._read_from_bigquery(train_table_name)
        df_test_input = pd.DataFrame()   # dummy

        df_train_features, _ = self.make_features(
            df_train_input, df_test_input
        )

        df_train_features.columns = f"{self.name}_" + df_train_features.columns

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)


if __name__ == "__main__":
    TargetCategory.main()
