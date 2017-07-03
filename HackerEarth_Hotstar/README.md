# IndiaHacks ML challenge on HackerEarth, data provided by Hotstar

### Problem Statement
- Identify users in a target segment using watch patterns.

### Dataset 
- The [data](https://he-s3.s3.amazonaws.com/media/hackathon/machine-learning-indiahacks-2017/5f828822-4--4-hotstar_dataset.zip) is provided as a JSON file.
- The columns titles, genres, cities, tod and dow are the show, the genre, the city, time of day and day of the week.
- The data in the above columns is in the following format title1:20 => the user watched title1 for 20 secs.
- Evaluation criteria - AUC-ROC score.

### Approach
 - Language used for the solution is R.
 - The following columns were added to the data:
 		- all the unique genres watched by the users
 		- top 20 cities where most of the users used Hotstar
 		- top 400 titles that were watched by the users
 		- time of day 
 		- days of the week 
 - I have used XGBoost library for training the model.
 - Packages used are ggplot2, data.table, caret, xgboost, e1071, tidyr, dplyr.

### Model Training
 - Model 6 and 7 contain 2 and 3 way feature interactions from XGBFi
 - The final model is and ensemble of all the above models and is again trained on XGBoost.
