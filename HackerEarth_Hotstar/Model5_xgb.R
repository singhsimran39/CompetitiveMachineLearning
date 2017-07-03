library(xgboost)
library(data.table)

#2 way interaction from XGBfi
train2[, CricketRomance := Cricket * Romance]
test2[, CricketRomance := Cricket * Romance]

train2[, CricketTotal := Cricket * totalTime]
test2[, CricketTotal := Cricket * totalTime]

train2[, CricketIshqbaaz := Cricket * Ishqbaaaz]
test2[, CricketIshqbaaz := Cricket * Ishqbaaaz]

train2[, CricketDrama := Cricket * Drama]
test2[, CricketDrama := Cricket * Drama]

train2[, KoffeTotal := KoffeeWithKaran * totalTime]
test2[, KoffeTotal := KoffeeWithKaran * totalTime]

train2[, CricketKoffee := Cricket * KoffeeWithKaran]
test2[, CricketKoffee := Cricket * KoffeeWithKaran]

train2[, DramaKoffee := Drama * KoffeeWithKaran]
test2[, DramaKoffee := Drama * KoffeeWithKaran]

train2[, totalsq := totalTime * totalTime]
test2[, totalsq := totalTime * totalTime]

train2[, CricketDay4 := Cricket * DayFour]
test2[, CricketDay4 := Cricket * DayFour]

train2[, CricketFamily := Cricket * Family]
test2[, CricketFamily := Cricket * Family]

train2[, KoffeeRomance := KoffeeWithKaran * Romance]
test2[, KoffeeRomance := KoffeeWithKaran * Romance]

train2[, CricketRishta := Cricket * YehRishtaKyaKehlataHai]
test2[, CricketRishta := Cricket * YehRishtaKyaKehlataHai]

train2[, KhokaRomance := KhokaBabu * Romance]
test2[, KhokaRomance := KhokaBabu * Romance]

train2[, KoffeeRishta := KoffeeWithKaran * YehRishtaKyaKehlataHai]
test2[, KoffeeRishta := KoffeeWithKaran * YehRishtaKyaKehlataHai]

train2[, CrciketFourteen := Cricket * `HourFourteen:`]
test2[, CrciketFourteen := Cricket * `HourFourteen:`]

train2[, DramaIshqbaaz := Drama * Ishqbaaaz]
test2[, DramaIshqbaaz := Drama * Ishqbaaaz]

train2[, IshqbaazKoffee := Ishqbaaaz * KoffeeWithKaran]
test2[, IshqbaazKoffee := Ishqbaaaz * KoffeeWithKaran]

train2[, CricketNaamkaran := Cricket * Naamkarann]
test2[, CricketNaamkaran := Cricket * Naamkarann]

train2[, CricketNach := Cricket * NachBaliye]
test2[, CricketNach := Cricket * NachBaliye]

train2[, Cricket23 := Cricket * `HourTwentyThree:`]
test2[, Cricket23 := Cricket * `HourTwentyThree:`]

train2[, KoffeeAIB := KoffeeWithKaran * OnAirWithAIB]
test2[, KoffeeAIB := KoffeeWithKaran * OnAirWithAIB]

train2[, ActionTotal := Action * totalTime]
test2[, ActionTotal := Action * totalTime]


xgbTrain <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbVal <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbTest <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))

params5 <- list(booster = "gbtree", 
                 objective = "binary:logistic", 
                 eval_metric = "auc", 
                 eta = 0.1, 
                 max_depth = 4, 
                 subsample = .8, 
                 max_delta_step = 5
)

set.seed(1234)
xgbcv5 <- xgb.cv(params = params5, 
                  data = xgbTrain, 
                  nrounds = 800, 
                  nfold = 5, 
                  showsd = T, 
                  stratified = T,
                  print_every_n = 10, 
                  early_stopping_rounds = 5, 
                  maximize = T, 
                  prediction = T
)
#0.825290

xgb5 <- xgb.train(params = params5, 
                   data = xgbTrain, 
                   nrounds = xgbcv5$best_iteration, 
                   print_every_n = 10, 
                   early_stopping_rounds = 2,
                   watchlist = list(val = xgbVal, train = xgbTrain), 
                   maximize = T
)
#0.829895

xgb5.preds <- as.data.table(predict(xgb5, xgbTest))
xgb5.preds <- cbind(ID = test2$ID, segment = xgb5.preds$V1)
xgb5.preds <- as.data.table(xgb5.preds)
fwrite(xgb5.preds, "xgb5.csv")

