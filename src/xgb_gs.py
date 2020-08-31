import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config

from attributes_cleaner import NumAttributesCleaner
from attributes_cleaner import CatAttributesCleaner

import joblib
import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import xgboost



def run():
    df_train = pd.read_csv(config.AIRBNB_TRAIN_FILE).drop("id", axis=1)
    df_test = pd.read_csv(config.AIRBNB_TEST_FILE).drop("id", axis=1)

    attribs = [f for f in df_train.columns if f not in ["price", "kfold"]]
    cat_attribs = ["zipcode",
                   "neighbourhood_cleansed",
                   "room_type",
                   "bed_type",
                   "cancellation_policy",
                   "security_deposit"]

    num_attribs = [f for f in attribs if f not in cat_attribs]


    X_train, y_train = df_train[attribs].copy(), df_train["price"].copy()
    X_test, y_test = df_test[attribs].copy(), df_test["price"].copy()


    num_transformer = Pipeline([
        ("num_cleaner", NumAttributesCleaner(num_attribs))])

    cat_transformer = Pipeline([
        ("cat_clener", CatAttributesCleaner(cat_attribs)),
        ("ohe", OneHotEncoder())])

    transformer = ColumnTransformer([
        ("num", num_transformer, num_attribs),
        ("cat", cat_transformer, cat_attribs)])


    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    #k_best = SelectKBest(f_regression, k=20)
    #rf = RandomForestRegressor(max_depth=2, random_state=42)
    #selector = SelectFromModel(estimator=rf, max_features=20)

    #selector = SelectKBest(f_regression, k=5)

    #X_train = selector.fit_transform(X_train, y_train)
    #X_valid = selector.transform(X_valid)


    param_dist = {'learning_rate':0.1,
                  'objective':'reg:squarederror',
                  "random_state":42}
    model = xgboost.XGBRegressor(**param_dist)

    param_grid = [{'n_estimators': [200, 300, 400],
                   'max_depth': [2, 4, 8],
                   'learning_rate': [.03, 0.05, .07,.1]}]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

    grid_search.fit(X_train, y_train,eval_set=[(X_test, y_test)], early_stopping_rounds=2, verbose=0.)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"xgb_ms.bin"))


if __name__ == "__main__":
        run()
