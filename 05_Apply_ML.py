
import joblib
import os
import pandas as pd

import sklearn
'''
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
'''

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        Execution Code        #
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
Y_pred = model.predict(X_pred)

print(Y_pred)