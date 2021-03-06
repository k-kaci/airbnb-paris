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


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def run(fold):
    df = pd.read_csv(config.AIRBNB_TRAIN_FOLDS_FILE).drop("id", axis=1)

    attribs = [f for f in df.columns if f not in ["price", "kfold"]]
    cat_attribs = ["zipcode",
                   "neighbourhood_cleansed",
                   "room_type",
                   "bed_type",
                   "cancellation_policy",
                   "security_deposit"]

    num_attribs = [f for f in attribs if f not in cat_attribs]

    df_train, df_valid = df[df.kfold != fold], df[df.kfold == fold]

    X_train, y_train = df_train[attribs].copy(), df_train["price"].copy()
    X_valid, y_valid = df_valid[attribs].copy(), df_valid["price"].copy()


    num_transformer = Pipeline([
        ("num_cleaner", NumAttributesCleaner(num_attribs)),
        ("std_scaler", StandardScaler())])

    cat_transformer = Pipeline([
        ("cat_cleaner", CatAttributesCleaner(cat_attribs)),
        ("cat_encoder", CatAttributesEncoder(cat_attribs))])

    transformer = ColumnTransformer([
        ("num", num_transformer, num_attribs),
        ("cat", cat_transformer, cat_attribs)])


    X_train_tr = transformer.fit_transform(X_train)
    X_valid_tr = transformer.transform(X_valid)



    model = LinearRegression()
    model.fit(X_train_tr, y_train)
    train_preds = model.predict(X_train_tr)
    valid_preds = model.predict(X_valid_tr)

    train_scores = {"r2_score": metrics.r2_score(y_train, train_preds),
                    "rmse_score": np.sqrt(metrics.mean_squared_error(y_train, train_preds))}

    valid_scores = {"r2_score": metrics.r2_score(y_valid, valid_preds),
                    "rmse_score": np.sqrt(metrics.mean_squared_error(y_valid, valid_preds))}

    print(f"Fold={fold} (train): ",
          f"r2_score = {train_scores['r2_score'].round(2)}", "--- ",
          f"rmse_score = {train_scores['rmse_score'].round(2)}")

    print(f"Fold={fold} (valid): ",
          f"r2_score = {valid_scores['r2_score'].round(2)}", "--- ",
          f"rmse_score = {valid_scores['rmse_score'].round(2)}")
    print("")
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"lr_{fold}.bin"))

if __name__ == "__main__":
    for fold in range(5):
        run(fold)



