library(data.table)
library(xgboost)
library(stringr)

#Add days column to train
days <- c("1:", "2:", "3:", "4:", "5:", "6:", "7:")

train[, d1 := lapply(dow, function(x) str_extract_all(x, "\\d+\\:"))]
train[, d1 := lapply(d1, unlist, use.names = F)]

toDayColumns <- function(data, variables) {       #this adds days as columns(Reference - starter script on HackerEarth)
  
  for(i in variables){
    
    data[, paste0(i) := lapply(d1, function(x) any(match(i,x)))]
    
  }
  return (data)
  
}

toDayColumns(train, days)       #Call the function

for(i in days) {       #add day and time watched
  
  idx <- which(train[, paste(i), with = F] == TRUE)
  pat <- paste0(i, "[0-9]+")
  train[idx, paste(i) := regmatches(dow, regexpr(pattern = pat, dow))]
  
}

train[, (days) := lapply(.SD, function(x) substring(x, 3)), .SDcols = days]
train[, d1 := NULL]
colnames(train)[87:93] <- c("DayOne", "DayTwo", "DayThree", "DayFour", "DayFive", "DaySix", "DaySeven")
days <- c("DayOne", "DayTwo", "DayThree", "DayFour", "DayFive", "DaySix", "DaySeven")
train[, (days) := lapply(.SD, as.numeric), .SDcols = days]
train[is.na(train)] <- 0

#Add days column to test
days <- c("1:", "2:", "3:", "4:", "5:", "6:", "7:")

test[, d1 := lapply(dow, function(x) str_extract_all(x, "\\d+\\:"))]
test[, d1 := lapply(d1, unlist, use.names = F)]

toDayColumns(test, days)       #Call the function

for(i in days) {
  
  idx <- which(test[, paste(i), with = F] == TRUE)
  pat <- paste0(i, "[0-9]+")
  test[idx, paste(i) := regmatches(dow, regexpr(pattern = pat, dow))]
  
}

test[, (days) := lapply(.SD, function(x) substring(x, 3)), .SDcols = days]
test[, d1 := NULL]
colnames(test)[86:92] <- c("DayOne", "DayTwo", "DayThree", "DayFour", "DayFive", "DaySix", "DaySeven")
days <- c("DayOne", "DayTwo", "DayThree", "DayFour", "DayFive", "DaySix", "DaySeven")
test[, (days) := lapply(.SD, as.numeric), .SDcols = days]
test[is.na(test)] <- 0


#I will use the same idxTrain and idxVal values from saved variables
train2 <- train[, -c(2:4, 6, 7, 44)]
test2 <- test[, -c(2:6, 43)]

train2$segment <- as.numeric(train2$segment)
train2$segment <- train2$segment - 1

tempTrain <- model.matrix(~.+0, data = train2[, .(hoursCount)])
tempTest <- model.matrix(~.+0, data = test2[, .(hoursCount)])

train2 <- cbind(train2, tempTrain)
test2 <- cbind(test2, tempTest)

train2[, hoursCount := NULL]
test2[, hoursCount := NULL]


xgbTrain <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbVal <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbTest <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))


params2 <- list(booster = "gbtree", 
                objective = "binary:logistic", 
                eval_metric = "auc", 
                eta = 0.1, 
                max_depth = 6, 
                subsample = .8
)

set.seed(1234)
xgbcv2 <- xgb.cv(params = params2, 
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

xgb2 <- xgb.train(params = params2, 
                  data = xgbTrain, 
                  nrounds = xgbcv2$best_iteration, 
                  print_every_n = 10, 
                  early_stopping_rounds = 2,
                  watchlist = list(val = xgbVal, train = xgbTrain), 
                  maximize = T
)

xgb2.preds <- as.data.table(predict(xgb2, xgbTest))
xgb2.preds <- cbind(ID = test2$ID, segment = xgb2.preds$V1)
xgb2.preds <- as.data.table(xgb2.preds)
fwrite(xgb4.preds, "xgb2.csv")

xgbImp <- xgb.importance(colnames(train2)[-c(1,2)], model = xgb4)
xgb.plot.importance (importance_matrix = xgbImp[1:30])

#Prepare for Stacking
trainPred2 <- rbindlist(list(as.data.table(xgbcv2$pred), as.data.table(predict(xgb2, xgbVal))))
targetFull <- rbindlist(list(as.data.table(train2[idxTrain, segment]), as.data.table(train2[idxVal, segment])))

metaTrain <- cbind(trainPred2, targetFull)
colnames(metaTrain) <- c("xgb2", "xgb2.segment")
fwrite(metaTrain, "metaTrain_xgb2.csv")








