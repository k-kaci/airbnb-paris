
import pandas as pd
import numpy as np


from sklearn.preprocessing import FunctionTransformer
from sklearn.base import  BaseEstimator
from sklearn.base import  TransformerMixin
from sklearn.utils import check_array


def attributes_cleaner(df):

    # zipcode : fill with "NONE" and create "RARE" category
    # zipcode will be converted to string

    paris_zipcodes = np.arange(75001, 75021, 1, dtype=np.int64)
    paris_zipcodes = np.insert(paris_zipcodes, 16, 75116)
    paris_districts = {zipcode:str(zipcode)[-2:]+"_dist" for zipcode in paris_zipcodes}
    paris_districts[-1] = "NONE"

    df["zipcode"] = df["zipcode"].fillna(-1)
    df["zipcode"] = df["zipcode"].map(lambda x: "RARE"if x not in paris_districts.keys() else paris_districts[x] )

    # bedrooms : fill with 1
    df.loc[:, "bedrooms"] = df["bedrooms"].fillna(1)


    # beds : fill with 1 if bedrooms = 0 else fill with bedrooms
    fill_beds = df["bedrooms"].map(lambda x: 1 if x == 0 else x)
    df.loc[:, "beds"] = [fill_beds[i] if pd.isna(df.loc[i, "beds"])
                         else df.loc[i, "beds"]
                         for i in df.index]

    # bathrooms : fill with 1
    df.loc[:, "bathrooms"] = df["bathrooms"].fillna(1)

    # cancellation_policy : fill with "NONE"
    df["cancellation_policy"] = df["cancellation_policy"].fillna("NONE")

    # host_is_superhost : fill with 0
    df["host_is_superhost"] = df["host_is_superhost"].fillna(0).astype('int64')

    # host_is_superhost :fill with 1
    df["host_total_listings_count"] = df["host_total_listings_count"].fillna(1).astype('int64')

    return df

class CatAttributesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        # zipcode : fill with "NONE" and create "RARE" category
        # zipcode will be converted to string
        if "zipcode" in self.attributes_names:
            paris_zipcodes = np.arange(75001, 75021, 1, dtype=np.int64)
            paris_zipcodes = np.insert(paris_zipcodes, 16, 75116)
            paris_districts = {zipcode: str(zipcode)[-2:] + "_dist" for zipcode in paris_zipcodes}
            paris_districts[-1] = "NONE_dist"

            X_["zipcode"] = X_["zipcode"].fillna(-1)
            X_["zipcode"] = X_["zipcode"].map(lambda x: "RARE_dist" if x not in paris_districts.keys() else paris_districts[x])


        # cancellation_policy : fill with  default "strict"
        if "cancellation_policy" in self.attributes_names:
            X_["cancellation_policy"] = X_["cancellation_policy"].fillna("strict")



        return X_

class NumAttributesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()

        # host_total_listings :fill with default 1
        if "host_total_listings" in self.attributes_names:
            X_["host_total_listings"] = X_["host_total_listings"].fillna(1)
            #X_["host_total_listings"] = X_["host_total_listings"].fillna(1)

        # bedrooms : fill with default 1
        if "bedrooms" in self.attributes_names:
            X_.loc[:, "bedrooms"] = X_["bedrooms"].fillna(1)

        # beds : fill with 1 if bedrooms = 0 else fill with bedrooms
        if "beds" in self.attributes_names:
            fill_beds = X_["bedrooms"].map(lambda x: 1 if x == 0 else x)
            X_.loc[:, "beds"] = [fill_beds[i] if pd.isna(X_.loc[i, "beds"])
                                 else X_.loc[i, "beds"]
                                 for i in X_.index]

        # bathrooms : fill with default 1
        if "bathrooms" in self.attributes_names:
            X_.loc[:, "bathrooms"] = X_["bathrooms"].fillna(1)

        # is_location_exact : fill with default 1
        if "is_location_exact" in self.attributes_names:
            X_["is_location_exact"] = X_["is_location_exact"].fillna(1)

        # host_is_superhost : fill with default 0
        if "host_is_superhost" in self.attributes_names:
            X_["host_is_superhost"] = X_["host_is_superhost"].fillna(0)

        # host_since_years : fill with default 0
        if "host_since_years" in self.attributes_names:
            host_since = X_["host_since_years"].median()
            X_["host_since_years"] = X_["host_since_years"].fillna(host_since)

        # reviews_per_month : fill with default median
        if "reviews_per_month" in self.attributes_names:
            reviews_pm = X_["number_of_reviews"] / (X_["host_since_years"] * 30)
            X_["reviews_per_month"] = X_["reviews_per_month"].fillna(reviews_pm)

        # cleaning_fee : fill with default median
        if "cleaning_fee" in self.attributes_names:
            cleaning_fee = X_["cleaning_fee"].median()
            X_["cleaning_fee"] = X_["cleaning_fee"].fillna(cleaning_fee)


        return X_


class CatAttributesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if "zipcode" in self.attributes_names:
            df_dum= pd.get_dummies(X_["zipcode"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("zipcode", axis=1)

        if "neighbourhood_cleansed" in self.attributes_names:
            df_dum = pd.get_dummies(X_["neighbourhood_cleansed"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("neighbourhood_cleansed", axis=1)

        if "room_type" in self.attributes_names:
            df_dum = pd.get_dummies(X_["room_type"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("room_type", axis=1)

        if "bed_type" in self.attributes_names:
            df_dum = pd.get_dummies(X_["bed_type"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("bed_type", axis=1)

        if "cancellation_policy" in self.attributes_names:
            df_dum = pd.get_dummies(X_["cancellation_policy"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("cancellation_policy", axis=1)
            #X_["cancellation_policy"] = X_["cancellation_policy"].map({"flexible": 1,
            #                                                           "moderate": 2,
            #                                                          "strict": 3 })
        if "security_deposit" in self.attributes_names:
            df_dum = pd.get_dummies(X_["security_deposit"])
            for col in df_dum.columns:
                X_[col] = df_dum[col]
            X_ = X_.drop("security_deposit", axis=1)
        return X_













