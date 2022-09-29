# Prediction requierements
import joblib
import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)

print(pd.__version__)
print(np.__version__)
print(sklearn.__version__)
print(joblib.__version__)

# FastAPI requierements
from fastapi import FastAPI
import uvicorn
from typing import Union

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       Prediction Model       #
#______________________________#

# importation necessary (Model & Preprocessor)
path = os.path.dirname(__file__)
preprocessor = joblib.load(path+"/src/custom_transformer.pkl")
model = joblib.load(path+"/src/model.joblib")

# importation 1 line for test 
dataset = pd.read_csv(path+"/src/get_around_pricing_project_clean.csv")
dataset = dataset[0:1]

# Apply columns transformations
X_pred = dataset.drop(columns= ["rental_price_per_day"])
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


