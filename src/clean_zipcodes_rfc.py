import os

import config

import joblib
import pandas as pd
import numpy as np


from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def run():
    df = pd.read_csv(config.AIRBNB_TRAIN_FOLDS_FILE, usecols=["latitude", "longitude","zipcode"])
    X = df[df.zipcode.notna()][["latitude", "longitude"]].values
    y = df[df.zipcode.notna()]["zipcode"].values.astype(np.int64)

    rfc = RandomForestClassifier(random_state=42)

    parameters = {'n_estimators': [10, 20, 40, 60, 80, 100],
                  'max_depth': [8, 16, 32]}

    clf = GridSearchCV(rfc, parameters, cv=5, scoring='accuracy')
    clf.fit(X, y)

    print(f"best_params_ : {clf.best_params_}")
    print(f"accuracy : {clf.best_score_}")

    joblib.dump(clf.best_estimator_, os.path.join(config.MODEL_OUTPU,"clean_zipcodes_rf.bin"))

if __name__ == "__main__":

    predict_missing_zipcodes(df)



