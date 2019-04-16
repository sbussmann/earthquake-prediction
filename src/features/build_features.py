from src.features.feature_constructor import FeatureConstructor
from src.data.load_data import training_data
import os


if __name__ == "__main__":
    train_data = training_data(n_rows=300_000)
    feature_constructor = FeatureConstructor(train_data=train_data)
    all_stat_types = [
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
    feature_constructor.add_features(all_stat_types[:1])
    features_path = os.path.join("data", "interim", "train_features_basic_stats.csv")
    feature_constructor.train_features.to_csv(features_path)
