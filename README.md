# Bloc n°5 : Industrialisation d'un algorithme d'apprentissage automatique et automatisation des processus de décision.
## Contact 

[voguant-cal0n@icloud.com](mailto:voguant-cal0n@icloud.com)

## Video explain

[Bloc n°5 : Industrialisation d'un algorithme d'apprentissage automatique et automatisation des processus de décision.](https://youtu.be/aTxdoclEj9c "Bloc n°5")

## Links 

* [API on Heroku](https://getaroundapi-rg.herokuapp.com)
* [EDA on Heroku](https://getaroundweb-rg.herokuapp.com)

## Goals

In order to mitigate those issues we’ve decided to implement a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.

It solves the late checkout issue but also potentially hurts Getaround/owners revenues: we need to find the right trade off.

 * __Our Product Manager still needs to decide :__
    1. **threshold:** how long should the minimum delay be?
    2. **scope:** should we enable the feature for all cars?, only Connect cars?

* Web dashboard : First build a dashboard that will help the product Management team with the above questions. 

* Machine Learning - `/predict` endpoint :The Data Science team is working on *pricing optimization*. They have gathered some data to suggest optimum prices for car owners using Machine Learning. 

 * Documentation page : You need to provide the users with a **documentation** about your API.

 * Online production : You have to **host your API online**. 


## Informations about files:

1. EDA_delay_analysis.ipynb is an Exploratory Data Analysis of the delay_analysis dataset
2. 02_EDA_pricing.ipynb is an Exploratory Data Analysis of the princing_project dataset
3. 03_ML_pricing.ipynb prepare the random forest model.
4. 04_Application
    1. app.py is the steamlit application page. 
    2. Dockerfile is the procedure for the docker creation
    3. PushMe.py is the file to run the create and push on heroku procedure.
    4. README.md explain the PushMe utilisation.
    5. requierements.txt contain the library to include in the docker.

5. 05_API
    1. app : Folder of the app, model and the original dataset. 
    2. Dockerfile is the procedure for the docker creation
    3. PushMe.py is the file to run the create and push on heroku procedure.
    4. README.md explain the PushMe utilisation.
    5. requierements.txt contain the library to include in the docker.

6. 06_Test_API.ipynb is a jupiter notebook for test the API