import numpy as np
import time
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold




def cross_validation():
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    return folds


def lgb_fit(X_train, y_train, X_valid, y_valid, test_features, params):
    model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="mae",
        verbose=10000,
        early_stopping_rounds=200,
    )

    y_pred_valid = model.predict(X_valid)
    y_pred = model.predict(test_features, num_iteration=model.best_iteration_)
    return y_pred, y_pred_valid


def train_and_predict(
    train_features,
    test_features,
    label_features,
    folds,
    params=None,
    model_type="lgb"
):

    out_of_fold = np.zeros(len(train_features))
    prediction = np.zeros(len(test_features))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_features)):
        print("Fold", fold_n, "started at", time.ctime())
        X_train, X_valid = train_features.iloc[train_index], train_features.iloc[valid_index]
        y_train, y_valid = label_features.iloc[train_index], label_features.iloc[valid_index]

        if model_type == "lgb":
            y_pred, y_pred_valid = lgb_fit(X_train, y_train, X_valid, y_valid, test_features, params)

        if model_type == "xgb":
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=train_features.columns
            )
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=train_features.columns
            )

            watchlist = [(train_data, "train"), (valid_data, "valid_data")]
            model = xgb.train(
                dtrain=train_data,
                num_boost_round=20000,
                evals=watchlist,
                early_stopping_rounds=200,
                verbose_eval=500,
                params=params,
            )
            y_pred_valid = model.predict(
                xgb.DMatrix(X_valid, feature_names=train_features.columns),
                ntree_limit=model.best_ntree_limit,
            )
            y_pred = model.predict(
                xgb.DMatrix(test_features, feature_names=train_features.columns),
                ntree_limit=model.best_ntree_limit,
            )

        if model_type == "sklearn":
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f"Fold {fold_n}. MAE: {score:.4f}.")
            print("")

            y_pred = model.predict(test_features).reshape(-1)

        if model_type == "cat":
            model = CatBoostRegressor(iterations=20000, eval_metric="MAE", **params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                cat_features=[],
                use_best_model=True,
                verbose=False,
            )

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(test_features)

        out_of_fold[valid_index] = y_pred_valid.reshape(-1)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

        if model_type == "lgb":
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = train_features.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )

    prediction /= len(folds)

    print(
        "CV mean score: {0:.4f}, std: {1:.4f}.".format(np.mean(scores), np.std(scores))
    )
    return out_of_fold, prediction