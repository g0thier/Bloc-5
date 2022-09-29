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
from typing import Literal, List, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import json

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



class ModelCars(BaseModel):
    model_key: Literal['Citroën', 'Peugeot', 'Renault', 'Audi', 'BMW', 'Mercedes', 
                       'Volkswagen', 'Nissan', 'Mitsubishi', 'SEAT', 'Subaru', 'Toyota'] = 'Renault'
    mileage: int = 142056
    engine_power: int = 120
    fuel: Literal['diesel', 'petrol'] = 'diesel'
    paint_color: Literal['black', 'brown', 'grey', 'white', 'silver', 'blue'] = 'black'
    car_type: Literal['estate', 'hatchback', 'sedan', 'subcompact', 'suv'] = 'estate'
    private_parking_available: bool = True
    has_gps: bool = True
    has_air_conditioning: bool = False
    automatic_car: bool = False
    has_getaround_connect: bool = True
    has_speed_regulator: bool = False
    winter_tires: bool = True



@app.get("/get_predict/{model_key}")
async def get_predict(model_key: str = 'Renault', mileage: int = 142056, engine_power: int = 120, 
                      fuel: str = 'diesel', paint_color: str = 'black', car_type: str = 'estate', 
                      private_parking_available: bool = True, has_gps: bool = True, 
                      has_air_conditioning: bool = False, automatic_car: bool = False, 
                      has_getaround_connect: bool = True, has_speed_regulator: bool = False, 
                      winter_tires: bool = True):
    data_car = {
        'model_key': [model_key], 'mileage': [mileage],
        'engine_power': [engine_power], 'fuel': [fuel],
        'paint_color': [paint_color], 'car_type': [car_type],
        'private_parking_available': [private_parking_available],
        'has_gps': [has_gps], 'has_air_conditioning': [has_air_conditioning],
        'automatic_car': [automatic_car], 'has_getaround_connect': [has_getaround_connect],
        'has_speed_regulator': [has_speed_regulator], 'winter_tires': [winter_tires]
    }
    # From Dict to Pandas
    X_pred = pd.DataFrame.from_dict(data_car)
    # From Normal to preprocessed
    X_pred = preprocessor.transform(X_pred)
    # Prediction 
    Y_pred = model.predict(X_pred)[0]
    # Round 
    Y_pred = [int(x) for x in Y_pred]
    # Format response
    response = {"prediction": Y_pred}
    return response



@app.post("/predict")
async def predict(model_car: ModelCars):
    # From API input to dict based on ModelCars
    data_car = {
        'model_key': [model_car.model_key],
        'mileage': [model_car.mileage],
        'engine_power': [model_car.engine_power],
        'fuel': [model_car.fuel],
        'paint_color': [model_car.paint_color],
        'car_type': [model_car.car_type],
        'private_parking_available': [model_car.private_parking_available],
        'has_gps': [model_car.has_gps],
        'has_air_conditioning': [model_car.has_air_conditioning],
        'automatic_car': [model_car.automatic_car],
        'has_getaround_connect': [model_car.has_getaround_connect],
        'has_speed_regulator': [model_car.has_speed_regulator],
        'winter_tires': [model_car.winter_tires],
    }
    # From Dict to Pandas
    X_pred = pd.DataFrame.from_dict(data_car)
    # From Normal to preprocessed
    X_pred = preprocessor.transform(X_pred)
    # Prediction 
    Y_pred = model.predict(X_pred)
    # Round 
    Y_pred = [int(x) for x in Y_pred]
    # Format response
    response = {"prediction": Y_pred}
    return response


@app.post("/predict_json/")
async def predict_json(file: UploadFile = File(...)):
    # From Dict to Pandas
    X_pred = pd.read_json(file.file)
    # From Normal to preprocessed
    X_pred = preprocessor.transform(X_pred)
    # Prediction 
    Y_pred = model.predict(X_pred)
    # Round 
    Y_pred = [int(x) for x in Y_pred]
    # Format response
    response = {"prediction": Y_pred}
    return response