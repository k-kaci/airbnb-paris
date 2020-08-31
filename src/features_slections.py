import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config

from attributes_cleaner import NumAttributesCleaner
from attributes_cleaner import CatAttributesCleaner
from attributes_cleaner import CatAttributesEncoder

import joblib
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def run():
    df_train = pd.read_csv(config.AIRBNB_TRAIN_FILE).drop("id", axis=1)

    attribs = [f for f in df_train.columns if f not in ["price"]]
    cat_attribs = ["zipcode",
                   "neighbourhood_cleansed",
                   "room_type",
                   "bed_type",
                   "cancellation_policy",
                   "security_deposit"]

    num_attribs = [f for f in attribs if f not in cat_attribs]

    X, y = df_train[attribs].copy(), df_train.price


    num_transformer = Pipeline([
        ("num_cleaner", NumAttributesCleaner(num_attribs))])

    cat_transformer = Pipeline([
        ("cat_cleaner", CatAttributesCleaner(cat_attribs)),
        ("cat_encoder", CatAttributesEncoder(cat_attribs))])

    transformer = ColumnTransformer([
        ("num", num_transformer, num_attribs),
        ("cat", cat_transformer, cat_attribs)])

    X_tr = X.copy()
    X_tr = NumAttributesCleaner(num_attribs).transform(X_tr)
    X_tr = CatAttributesCleaner(cat_attribs).transform(X_tr)
    X_tr = CatAttributesEncoder(cat_attribs).transform(X_tr)

    k_best = SelectKBest(f_regression, k=10)
    k_best.fit(X_tr, y)

    k_best_scores = pd.DataFrame({"attribut":  X_tr.columns, "score": k_best.scores_ })
    k_best_scores = k_best_scores.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    print(k_best_scores.head(20))





if __name__ == "__main__":
        run()
