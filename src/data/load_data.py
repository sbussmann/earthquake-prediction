import os
import pandas as pd
import numpy as np


def training_data(n_rows=None):
    data_dir = os.path.join("data")
    raw_dir = os.path.join(data_dir, "raw")
    train = pd.read_csv(
        os.path.join(raw_dir, "train", "train.csv"),
        nrows=n_rows,
        dtype={"acoustic_data": np.int16, "time_to_failure": np.float64},
    )
    return train


def test_segment(segment_id):
    data_dir = os.path.join("data")
    raw_dir = os.path.join(data_dir, "raw")
    test_path = os.path.join(raw_dir, "test")
    segment = pd.read_csv(os.path.join(test_path, "{}.csv".format(segment_id)))
    return segment


def submission_sample():
    data_dir = os.path.join("data")
    submission_path = os.path.join(data_dir, "processed", "sample_submission.csv")
    submission = pd.read_csv(submission_path, index_col='seg_id')
    submission = submission[::100]
    return submission
