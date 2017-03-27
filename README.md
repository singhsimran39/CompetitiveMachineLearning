# Machine Learning Challenege hosted on HackerEarth

### Problem Statement
- Figure out a way to find who will default on loan payments.

### Dataset 
- There are 45 features in the dataset. 27 features are numerical while other are either factors are characters.
- The training set contains 532428 observations. 
- The test set contains 354951 observations.
- Participants have to predict loan_status variable which is 0 (not a defaulter) or 1 (defaulter).
- Evaluation criteria - AUC-ROC score.

### Approach
 - Language used for the solution is R.
  - First I have done some Exploratory Data Analysis. I have plotted various graphs of different variables against loan_status (to be predicted).
  - I have used XGBoost library for training the model.
  - Packages used are ggplot2, data.table, caret, xgboost, e1071, tidyr, dplyr.

### Model Training
 - I started with the default XGBoost parameters and then after reading through the [Parameter Tuning](http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html) guide from the XGBoost website i was able to achieve a score of .94 on the leaderboard that put me in the top 90 out of nearly 3000 submissions.