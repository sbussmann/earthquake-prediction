import json
import uuid
from uuid import UUID

from src.features.feature_constructor import FeatureConstructor
from src.data.load_data import training_data, submission_sample
from src.models.train_model import ModelTrainer, cross_validation
import os


if __name__ == "__main__":
    train_data = training_data(n_rows=1_500_000)
    stat_types = [
        "add_basic_stats",
        "add_rolling_stats",
        "add_trend_stats",
        "add_filtered_stats",
        "add_sta_lta_stats",
        "add_n_sigma_stats",
        "add_delta_stats",
        "add_start_end_stats",
        "add_ewma_stats",
    ]
    submission_df = submission_sample()

    for stat_type in stat_types[:1]:
        feature_constructor = FeatureConstructor(train_data, submission_df)
        feature_constructor.add_features([stat_type], scaled=True)
        feature_constructor.write_features()

        folds = cross_validation()

        model_trainer = ModelTrainer(
            feature_constructor.train_features,
            feature_constructor.test_features,
            feature_constructor.train_label,
            folds,
        )

        model_algorithms = {
            # "LGB": ["001"],
            # "XGB": ["001"],
            # "CatBoost": ["001"],
            "NuSVR": ["001", "002"]
        }
        for model_algorithm in model_algorithms:

            json_path = os.path.join("models", "{}_params.json".format(model_algorithm))

            with open(json_path, "r") as read_file:
                model_json = json.load(read_file)

                model_ids = model_algorithms[model_algorithm]

                for model_id in model_ids:
                    model_params = model_json[model_id]

                    model_trainer.train_and_predict(model_algorithm, model_params)
                    model_uuid = uuid.uuid1()

                    write_path = os.path.join("data", "processed", "submission-{}.json".format(model_uuid))
                    meta_data = {
                        "features_uuid": str(feature_constructor.uuid),
                        "algorithm": model_algorithm,
                        "iteration": model_id,
                        "params": model_params
                    }

                    with open(write_path, "w") as write_file:
                        json.dump(meta_data, write_file)
                    submission_path = os.path.join("data", "processed", "submission-{}.csv".format(model_uuid))
                    submission_df["time_to_failure"] = model_trainer.test_prediction
                    submission_df.to_csv(submission_path)
