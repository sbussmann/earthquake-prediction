import numpy as np
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import NuSVR


class ModelTrainer(object):
    def __init__(self, train_features, test_features, label_features):
        self.train_features = train_features
        self.test_features = test_features
        self.label_features = label_features

        self.folds = None

        self.out_of_fold = np.zeros(len(train_features))
        self.prediction = np.zeros(len(test_features))

        self.scores = []

    def cross_validation(self):
        n_fold = 5
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
        self.folds = folds

    def train_and_predict(self, model_type, params=None):

        train_features = self.train_features
        test_features = self.test_features
        label_features = self.label_features
        folds = self.folds
        for fold_n, (train_index, valid_index) in enumerate(folds.split(train_features)):
            print("Fold", fold_n, "started at", time.ctime())
            X_train, X_valid = (
                train_features.iloc[train_index],
                train_features.iloc[valid_index],
            )
            y_train, y_valid = (
                label_features.iloc[train_index],
                label_features.iloc[valid_index],
            )

            ModelFitClass = globals()["{}Fitter".format(model_type)]
            model_fitter = ModelFitClass(X_train, y_train, X_valid, y_valid, params)
            model_fitter.fit()
            model_fitter.predict()

            self.out_of_fold[valid_index] = model_fitter.y_predict_valid.reshape(-1)
            self.scores.append(model_fitter.score())

            y_predict = model_fitter.predict(test_features)

            self.prediction += y_predict

        self.prediction /= len(folds)


class ModelFitter(object):
    def __init__(self, X_train, y_train, X_valid, y_valid, params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.params = params

        self.model = None

        self.y_predict_valid = None
        self.y_predict = None

        self.score = None

    def score(self):
        return mean_absolute_error(self.y_valid, self.y_predict_valid)


class LGBFitter(ModelFitter):
    def __init__(self, X_train, y_train, X_valid, y_valid, params):
        ModelFitter.__init__(self, X_train, y_train, X_valid, y_valid, params)

    def fit(self):
        model = lgb.LGBMRegressor(**self.params, n_estimators=50000, n_jobs=-1)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)],
            eval_metric="mae",
            verbose=10000,
            early_stopping_rounds=200,
        )

        self.model = model

    def predict(self, test_features):
        model = self.model
        self.y_predict_valid = model.predict(self.X_valid)
        self.y_predict = model.predict(
            test_features, num_iteration=model.best_iteration_
        )


class XGBModelFitter(ModelFitter):
    def __init__(self, X_train, y_train, X_valid, y_valid, params):
        ModelFitter.__init__(self, X_train, y_train, X_valid, y_valid, params)

    def fit(self):

        train_data = xgb.DMatrix(
            data=self.X_train,
            label=self.y_train,
            feature_names=self.params["feature_names"],
        )
        valid_data = xgb.DMatrix(
            data=self.X_valid,
            label=self.y_valid,
            feature_names=self.params["feature_names"],
        )

        watchlist = [(train_data, "train"), (valid_data, "valid_data")]
        model = xgb.train(
            dtrain=train_data,
            num_boost_round=20000,
            evals=watchlist,
            early_stopping_rounds=200,
            verbose_eval=500,
            params=self.params,
        )
        self.model = model

    def predict(self):
        model = self.model
        self.y_predict_valid = model.predict(
            xgb.DMatrix(self.X_valid, feature_names=self.params["feature_names"]),
            ntree_limit=model.best_ntree_limit,
        )


class NuSVRFitter(ModelFitter):
    def __init__(self, X_train, y_train, X_valid, y_valid, params):
        ModelFitter.__init__(self, X_train, y_train, X_valid, y_valid, params)

    def fit(self):
        model = NuSVR(**self.params)
        model.fit(self.X_train, self.y_train)

    def predict(self):
        model = self.model
        self.y_predict_valid = model.predict(self.X_valid).reshape(-1)


class CatBoostFitter(ModelFitter):
    def __init__(self, X_train, y_train, X_valid, y_valid, params):
        ModelFitter.__init__(self, X_train, y_train, X_valid, y_valid, params)

    def fit(self):
        model = CatBoostRegressor(iterations=20000, eval_metric="MAE", **self.params)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_valid, self.y_valid),
            cat_features=[],
            use_best_model=True,
            verbose=False,
        )

    def predict(self):
        model = self.model
        self.y_predict_valid = model.predict(self.X_valid)
