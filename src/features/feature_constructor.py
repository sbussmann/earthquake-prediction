import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features.feature_utils import (
    calc_change_rate,
    add_trend_feature,
    classic_sta_lta,
)
from src.data import load_data
from scipy.signal import hilbert, convolve
from scipy.signal.windows import hann
from scipy import stats
from sklearn.preprocessing import StandardScaler


class FeatureConstructor(object):
    def __init__(self, train_data, submission_example):
        self.train_data = train_data

        self.rows = 150_000
        self.segment_indexes = int(np.floor(train_data.shape[0] / self.rows))
        self.train_features = pd.DataFrame(
            index=range(self.segment_indexes), dtype=np.float64
        )

        self.train_label = pd.DataFrame(
            index=range(self.segment_indexes),
            dtype=np.float64,
            columns=["time_to_failure"],
        )

        self.test_features = pd.DataFrame(
            columns=self.train_features.columns,
            dtype=np.float64,
            index=submission_example.index,
        )
        self.test_label = None
        self.train_scaled_features = None
        self.test_scaled_features = None

    def add_train_features(self, stat_types):
        for segment_index in tqdm(range(self.segment_indexes)):
            train_segment = self.train_data.iloc[
                segment_index * self.rows : segment_index * self.rows + self.rows
            ]
            self.process_segment(train_segment, segment_index, stat_types)

            time_to_failure = train_segment["time_to_failure"].values[-1]
            self.train_label.loc[segment_index, "time_to_failure"] = time_to_failure

    def add_test_features(self, stat_types):
        for segment_index in tqdm(self.test_features.index):
            test_segment = load_data.test_segment(segment_index)
            self.process_segment(test_segment, segment_index, stat_types)

    def process_segment(self, segment, segment_index, stat_types):
        acoustic_data = pd.Series(segment["acoustic_data"].values)
        for stat_type in stat_types:
            if stat_type == "add_rolling_stats":
                for window_size in [10, 100, 1000]:
                    stats_result = globals()[stat_type](acoustic_data, window_size)
                    for stat_key in stats_result:
                        self.train_features.loc[segment_index, stat_key] = stats_result[
                            stat_key
                        ]
            elif stat_type == "add_n_sigma_stats":
                stats_result = globals()[stat_type](acoustic_data, n_sigma=3)
                for stat_key in stats_result:
                    self.train_features.loc[segment_index, stat_key] = stats_result[
                        stat_key
                    ]
            else:
                stats_function = globals()[stat_type]
                stats_result = stats_function(acoustic_data)
                for stat_key in stats_result:
                    self.train_features.loc[segment_index, stat_key] = stats_result[
                        stat_key
                    ]

    def add_features(self, stat_types):
        self.add_train_features(stat_types)
        self.add_test_features(stat_types)

    def add_scaled_features(self):
        standard_scaler = StandardScaler()
        train_features = self.train_features
        standard_scaler.fit(train_features)
        train_scaled_features = pd.DataFrame(
            standard_scaler.transform(train_features), columns=train_features.columns
        )
        self.train_scaled_features = train_scaled_features

        test_features = self.test_features
        test_scaled_features = pd.DataFrame(
            standard_scaler.transform(test_features), columns=test_features.columns
        )
        self.test_scaled_features = test_scaled_features


def add_basic_stats(segment):
    basic_stats = {
        "mean": segment.mean(),
        "std": segment.std(),
        "max": segment.max(),
        "min": segment.min(),
        "abs_max": np.abs(segment).max(),
        "abs_min": np.abs(segment).min(),
        "abs_mean": np.abs(segment).mean(),
        "abs_std": np.abs(segment).std(),
        "sum": segment.sum(),
        "q95": np.quantile(segment, 0.95),
        "q99": np.quantile(segment, 0.99),
        "q05": np.quantile(segment, 0.05),
        "q01": np.quantile(segment, 0.01),
        "abs_q95": np.quantile(np.abs(segment), 0.95),
        "abs_q99": np.quantile(np.abs(segment), 0.99),
        "abs_q05": np.quantile(np.abs(segment), 0.05),
        "abs_q01": np.quantile(np.abs(segment), 0.01),
        "mad": segment.mad(),
        "kurt": segment.kurtosis(),
        "skew": segment.skew(),
        "med": segment.median(),
        "q999": np.quantile(segment, 0.999),
        "q001": np.quantile(segment, 0.001),
        "iqr": np.subtract(*np.percentile(segment, [75, 25])),
        "ave10": stats.trim_mean(segment, 0.1),
        "max_to_min": segment.max() / np.abs(segment.min()),
        "max_to_min_diff": segment.max() - np.abs(segment.min()),
        "count_big": len(segment[np.abs(segment) > 500]),
    }
    return basic_stats


def add_rolling_stats(segment, window_size):

    x_roll_std = segment.rolling(window_size).std().dropna().values
    x_roll_mean = segment.rolling(window_size).mean().dropna().values

    rolling_stats = {
        "ave_roll_std_{}".format(window_size): x_roll_std.mean(),
        "std_roll_std_{}".format(window_size): x_roll_std.std(),
        "max_roll_std_{}".format(window_size): x_roll_std.max(),
        "min_roll_std_{}".format(window_size): x_roll_std.min(),
        "q01_roll_std_{}".format(window_size): np.quantile(x_roll_std, 0.01),
        "q05_roll_std_{}".format(window_size): np.quantile(x_roll_std, 0.05),
        "q95_roll_std_{}".format(window_size): np.quantile(x_roll_std, 0.95),
        "q99_roll_std_{}".format(window_size): np.quantile(x_roll_std, 0.99),
        "av_change_abs_roll_std_{}".format(window_size): np.mean(np.diff(x_roll_std)),
        "av_change_rate_roll_std_{}".format(window_size): np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0]
        ),
        "abs_max_roll_std_{}".format(window_size): np.abs(x_roll_std).max(),
        "ave_roll_mean_{}".format(window_size): x_roll_mean.mean(),
        "std_roll_mean_{}".format(window_size): x_roll_mean.std(),
        "max_roll_mean_{}".format(window_size): x_roll_mean.max(),
        "min_roll_mean_{}".format(window_size): x_roll_mean.min(),
        "q01_roll_mean_{}".format(window_size): np.quantile(x_roll_mean, 0.01),
        "q05_roll_mean_{}".format(window_size): np.quantile(x_roll_mean, 0.05),
        "q95_roll_mean_{}".format(window_size): np.quantile(x_roll_mean, 0.95),
        "q99_roll_mean_{}".format(window_size): np.quantile(x_roll_mean, 0.99),
        "av_change_abs_roll_mean_{}".format(window_size): np.mean(np.diff(x_roll_mean)),
        "av_change_rate_roll_mean_{}".format(window_size): np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0]
        ),
        "abs_max_roll_mean_{}".format(window_size): np.abs(x_roll_mean).max(),
    }
    return rolling_stats


def add_trend_stats(segment):
    trend_stats = {
        "trend": add_trend_feature(segment),
        "abs_trend": add_trend_feature(segment, abs_values=True),
    }
    return trend_stats


def add_filtered_stats(segment):
    filtered_stats = {
        "Hilbert_mean": np.abs(hilbert(segment)).mean(),
        "Hann_window_mean": (
            convolve(segment, hann(150), mode="same") / sum(hann(150))
        ).mean(),
    }
    return filtered_stats


def add_sta_lta_stats(segment):
    sta_lta_params = [
        {"sta": 500, "lta": 10000},
        {"sta": 5000, "lta": 100_000},
        {"sta": 3333, "lta": 6666},
        {"sta": 10000, "lta": 25000},
        {"sta": 50, "lta": 1000},
        {"sta": 100, "lta": 5000},
        {"sta": 333, "lta": 666},
        {"sta": 4000, "lta": 10000},
    ]
    sta_lta_stats = {}
    for i, sta_lta_param in enumerate(sta_lta_params):
        sta_lta_stats = {
            "classic_sta_lta{}_mean".format(i + 1): classic_sta_lta(
                segment, sta_lta_param["sta"], sta_lta_param["lta"]
            ).mean()
        }

    return sta_lta_stats


def add_n_sigma_stats(segment, n_sigma=3):
    moving_average_700_mean = segment.rolling(window=700).mean().mean(skipna=True)
    ma_700ma_std_mean = segment.rolling(window=700).std().mean()
    ma_400ma_std_mean = segment.rolling(window=400).std().mean()
    n_sigma_stats = {
        "MA_700MA_std_mean": ma_700ma_std_mean,
        "MA_700MA_BB_high_mean": (
            moving_average_700_mean + n_sigma * ma_700ma_std_mean
        ).mean(),
        "MA_700MA_BB_low_mean": (
            moving_average_700_mean - n_sigma * ma_700ma_std_mean
        ).mean(),
        "MA_400MA_std_mean": ma_400ma_std_mean,
        "MA_400MA_BB_high_mean": (
            moving_average_700_mean + n_sigma * ma_400ma_std_mean
        ).mean(),
        "MA_400MA_BB_low_mean": (
            moving_average_700_mean - n_sigma * ma_400ma_std_mean
        ).mean(),
        "MA_1000MA_std_mean": (segment.rolling(window=1000).std().mean()),
    }
    return n_sigma_stats


def add_delta_stats(segment):
    delta_stats = {
        "mean_change_abs": np.mean(np.diff(segment)),
        "mean_change_rate": calc_change_rate(segment),
    }
    return delta_stats


def add_start_end_stats(segment):
    start_end_stats = {
        "std_first_50000": segment[:50000].std(),
        "std_last_50000": segment[-50000:].std(),
        "std_first_10000": segment[:10000].std(),
        "std_last_10000": segment[-10000:].std(),
        "avg_first_50000": segment[:50000].mean(),
        "avg_last_50000": segment[-50000:].mean(),
        "avg_first_10000": segment[:10000].mean(),
        "avg_last_10000": segment[-10000:].mean(),
        "min_first_50000": segment[:50000].min(),
        "min_last_50000": segment[-50000:].min(),
        "min_first_10000": segment[:10000].min(),
        "min_last_10000": segment[-10000:].min(),
        "max_first_50000": segment[:50000].max(),
        "max_last_50000": segment[-50000:].max(),
        "max_first_10000": segment[:10000].max(),
        "max_last_10000": segment[-10000:].max(),
        "mean_change_rate_first_50000": calc_change_rate(segment[:50000]),
        "mean_change_rate_last_50000": calc_change_rate(segment[-50000:]),
        "mean_change_rate_first_10000": calc_change_rate(segment[:10000]),
        "mean_change_rate_last_10000": calc_change_rate(segment[-10000:]),
    }
    return start_end_stats


def add_ewma_stats(segment):
    ewma = pd.Series.ewm
    ewma_stats = {
        "exp_Moving_average_300_mean": (ewma(segment, span=300).mean()).mean(
            skipna=True
        ),
        "exp_Moving_average_3000_mean": (
            ewma(segment, span=3000).mean().mean(skipna=True)
        ),
        "exp_Moving_average_30000_mean": (
            ewma(segment, span=30000).mean().mean(skipna=True)
        ),
    }
    return ewma_stats
