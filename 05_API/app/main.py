# Prediction requierements
import joblib
import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)

# FastAPI requierements
from fastapi import FastAPI
import uvicorn
from typing import Union

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         Importations         #
#______________________________#

path = os.path.dirname(__file__)

# importation model
model = joblib.load(path+"/src/model.joblib")

# importation dataset
dataset = pd.read_csv(path+"/src/get_around_pricing_project_clean.csv")
target_name = 'rental_price_per_day'

# separate target and explain value 
Y = dataset[:][target_name]
X = dataset.drop(columns= [target_name])

# split in train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       Preprocessing X        #
#______________________________#

# Create pipeline for numeric features
numeric_features = X.select_dtypes([np.number]).columns # Automatically detect positions of numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # missing values will be replaced by columns' median
    ('scaler', StandardScaler())
])

# Create pipeline for categorical features
categorical_features = X.select_dtypes("object").columns # Automatically detect positions of categorical columns
categorical_transformer = Pipeline(
    steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # missing values will be replaced by most frequent value
    ('encoder', OneHotEncoder(drop='first')) # first column will be dropped to avoid creating correlations between features
    ])

# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train) # Preprocessing influenceur
X_test = preprocessor.transform(X_test) # Preprocessing copieur


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       Prediction Model       #
#______________________________#

# Apply columns transformations
X_pred = dataset.drop(columns= ["rental_price_per_day"])
X_pred = X_pred[0:1]
X_pred = preprocessor.transform(X_pred)

# Prediction 
Y_pred = model.predict(X_pred)[0]

print(Y_pred)

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       FastAPI run app.       #
#______________________________#

#
## Init fastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


