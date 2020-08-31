# stratified-kfold for regression
import config

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import FunctionTransformer


def create_folds(df):
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # bin target
    df.loc[:, "bins"] = pd.cut(df["price"], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[val_, "kfold"] = fold

    # drop the bins column
    df = df.drop("bins", axis=1)

    return df

CreateFolds = FunctionTransformer(create_folds)

if __name__ == "__main__":
    df = pd.read_csv(config.AIRBNB_TRAIN_FILE)
    df = create_folds(df=df)
    df.to_csv(config.AIRBNB_TRAIN_FOLDS_FILE, index=False)
    print(df.sample(10))