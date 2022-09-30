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
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
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
#       FastAPI describe       #
#______________________________#


description = """
## Predict the rental price per day of your car
 * __[Bloc n°5](https://github.com/g0thier/Bloc-5)__ : Industrialization of a machine learning algorithm and automation of decision-making processes.
"""

app = FastAPI(
    title="🚗 Get Around Analysis",
    description=description,
    version="0.1",
    #contact={
    #    "name": "Gauthier Rammault",
    #    "url": "https://www.linkedin.com/in/gauthier-rammault/",
    #},
    openapi_tags= [
        {
            "name": "Home",
            "description": "🚗 Get Around API homepage."
        },
        {
            "name": "Predicts",
            "description": "🚕 Get Around API with POST or GET method."
        }
    ]
)

html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>🚗 Get Around API</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="🚗 Get Around API">
    <meta name="author" content="Gauthier Rammault">

    <!-- Le styles -->
    <link href="https://getbootstrap.com/2.3.2/assets/css/bootstrap.css" rel="stylesheet">
    <style type="text/css">
      body {
        padding-top: 20px;
        padding-bottom: 40px;
      }

      /* Custom container */
      .container-narrow {
        margin: 0 auto;
        max-width: 700px;
      }
      .container-narrow > hr {
        margin: 30px 0;
      }

      /* Main marketing message and sign up button */
      .jumbotron {
        margin: 60px 0;
        text-align: center;
      }
      .jumbotron h1 {
        font-size: 72px;
        line-height: 1;
      }
      .jumbotron .btn {
        font-size: 21px;
        padding: 14px 24px;
      }

      /* Supporting marketing content */
      .marketing {
        margin: 60px 0;
      }
      .marketing p + h4 {
        margin-top: 28px;
      }
    </style>
    <link href="https://getbootstrap.com/2.3.2/assets/css/bootstrap-responsive.css" rel="stylesheet">

    <!-- HTML5 shim, for IE6-8 support of HTML5 elements -->
    <!--[if lt IE 9]>
      <script src="https://getbootstrap.com/2.3.2/assets/js/html5shiv.js"></script>
    <![endif]-->

    <!-- Fav and touch icons -->
    <link rel="apple-touch-icon-precomposed" sizes="144x144" href="https://getbootstrap.com/2.3.2/assets/ico/apple-touch-icon-144-precomposed.png">
    <link rel="apple-touch-icon-precomposed" sizes="114x114" href="https://getbootstrap.com/2.3.2/assets/ico/apple-touch-icon-114-precomposed.png">
      <link rel="apple-touch-icon-precomposed" sizes="72x72" href="https://getbootstrap.com/2.3.2/assets/ico/apple-touch-icon-72-precomposed.png">
                    <link rel="apple-touch-icon-precomposed" href="https://getbootstrap.com/2.3.2/assets/ico/apple-touch-icon-57-precomposed.png">
                                   <link rel="shortcut icon" href="https://getbootstrap.com/2.3.2/assets/ico/favicon.png">
  </head>

  <body>

    <div class="container-narrow">

      <div class="masthead">
        <ul class="nav nav-pills pull-right">
          <li class="active"><a href="#">Home</a></li>
          <li><a href="/docs#/">Docs</a></li>
        </ul>
        <h3 class="muted">Project Get Around</h3>
      </div>

      <hr>

      <div class="jumbotron">
        <h1>Project 🚗<br>Get Around API</h1>
        <p class="lead">Predict the rental price per day of your car. This <b>/</b> is the most simple and default endpoint. If you want to learn more, check out documentation of the api at <b>/docs</b></p>
        <a class="btn btn-large btn-success" href="/docs#/">See /docs</a>
      </div>

      <hr>

      <div class="footer">
        <p>Projet <a href="https://github.com/g0thier/Bloc-5">Bloc n°5</a> Jedha by <a href="https://www.linkedin.com/in/gauthier-rammault/">Gauthier Rammault</a>, the guy dreams to wanna be a real Data Scientist.</p>
        <p>This page come from bootcamp template <a href="https://getbootstrap.com/2.3.2/examples/marketing-narrow.html#">on this page</a></p>
      </div>

    </div> <!-- /container -->

    <!-- Le javascript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script src="https://getbootstrap.com/2.3.2/assets/js/jquery.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-transition.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-alert.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-modal.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-dropdown.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-scrollspy.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-tab.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-tooltip.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-popover.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-button.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-collapse.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-carousel.js"></script>
    <script src="https://getbootstrap.com/2.3.2/assets/js/bootstrap-typeahead.js"></script>

  </body>
</html>

"""

@app.get("/", tags=["Home"]) # here we categorized this endpoint as part of "Name_1" tag
async def get():
    return HTMLResponse(html)



#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       FastAPI run app.       #
#______________________________#


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



@app.get("/get_predict/{model_key}", tags=["Predicts"])
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
    Y_pred = model.predict(X_pred)
    # Round 
    Y_pred = [int(x) for x in Y_pred]
    # Format response
    response = {"prediction": Y_pred}
    return response



@app.post("/predict", tags=["Predicts"])
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


@app.post("/predict_json/", tags=["Predicts"])
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


