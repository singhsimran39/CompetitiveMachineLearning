library(stringr)
library(data.table)
library(xgboost)


#Add hours column to train
hourNumbers <- c("10:", "11:", "12:", "13:", "14:", "15:", "16:", "17:", "18:", "19:", "20:", "21:", "22:", "23:", 
                 "0:", "1:", "2:", "3:", "4:", "5:", "6:", "7:", "8:", "9:")

hourWords <- c("HourTen:", "HourEleven:", "HourTwelve:", "HourThirteen:", "HourFourteen:", "HourFifteen:", 
               "HourSixteen:", "HourSeventeen:", "HourEighteen:", "HourNinteen:", "HourTwenty:", "HourTwentyOne:", 
               "HourTwentyTwo:", "HourTwentyThree:", "HourZero:", "HourOne:", "HourTwo:", "HourThree:", "HourFour:", 
               "HourFive:", "HourSix:", "HourSeven:", "HourEight:", "HourNine:")

hourWords2 <- c("HourZero:", "HourOne:", "HourTwo:", "HourThree:", "HourFour:", "HourFive:", "HourSix:", 
                "HourSeven:", "HourEight:", "HourNine:", "HourTen:", "HourEleven:", "HourTwelve:", "HourThirteen:", 
                "HourFourteen:", "HourFifteen:", "HourSixteen:", "HourSeventeen:", "HourEighteen:", "HourNinteen:", 
                "HourTwenty:", "HourTwentyOne:", "HourTwentyTwo:", "HourTwentyThree:")

trainHourTemp <- train[, .(ID, tod)]
testHourTemp <- test[, .(ID, tod)]

for(i in 1:24) {
  
  pat <- hourNumbers[i]
  rep <- hourWords[i]
  trainHourTemp[, tod := gsub(pattern = pat, replacement = rep, x = tod)]
  testHourTemp[, tod := gsub(pattern = pat, replacement = rep, x = tod)]
  
}

trainHourTemp[, t1 := lapply(tod, function(x) str_extract_all(x, "[[:alpha:]]+\\:"))]
trainHourTemp[, t1 := lapply(t1, unlist, use.names = F)]

testHourTemp[, t1 := lapply(tod, function(x) str_extract_all(x, "[[:alpha:]]+\\:"))]
testHourTemp[, t1 := lapply(t1, unlist, use.names = F)]

toHourColumns <- function(data, variables) {       #this adds hours as columns(Reference - starter script on HackerEarth)
  
  for(i in variables){
    
    data[, paste0(i) := lapply(t1, function(x) any(match(i,x)))]
    
  }
  return (data)
  
}

toHourColumns(trainHourTemp, hourWords2)
toHourColumns(testHourTemp, hourWords2)

for(i in hourWords2) {       #add hour and time watched
  
  pat <- paste0(i, "[0-9]+")
  
  idx <- which(trainHourTemp[, paste(i), with = F] == TRUE)
  trainHourTemp[idx, paste(i) := regmatches(tod, regexpr(pattern = pat, tod))]
  
  idx <- which(testHourTemp[, paste(i), with = F] == TRUE)
  testHourTemp[idx, paste(i) := regmatches(tod, regexpr(pattern = pat, tod))]
  
}

trainHourTemp[, (hourWords2) := lapply(.SD, function(x) str_replace_all(x, "[[:alpha:]]+\\:", "")), .SDcols = hourWords2]
trainHourTemp[, (hourWords2) := lapply(.SD, as.numeric), .SDcols = hourWords2]

testHourTemp[, (hourWords2) := lapply(.SD, function(x) str_replace_all(x, "[[:alpha:]]+\\:", "")), .SDcols = hourWords2]
testHourTemp[, (hourWords2) := lapply(.SD, as.numeric), .SDcols = hourWords2]

trainHourTemp[is.na(trainHourTemp)] <- 0
testHourTemp[is.na(testHourTemp)] <- 0

train <- cbind(train, trainHourTemp[, 4:27])
test <- cbind(test, testHourTemp[, 4:27])

#Model prep
train2 <- train[, -c(2:4, 6, 7)]
test2 <- test[, -c(2:6)]

train2$segment <- as.numeric(train2$segment)
train2$segment <- train2$segment - 1

train2$hoursCount <- as.numeric(train2$hoursCount)
test2$hoursCount <- as.numeric(test2$hoursCount)

train2$daysCount <- as.numeric(train2$daysCount)
test2$daysCount <- as.numeric(test2$daysCount)

xgbMats <- list()

xgbMats[[1]] <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbMats[[2]] <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbMats[[3]] <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))


params5 <- list(booster = "gbtree", 
                objective = "binary:logistic", 
                eval_metric = "auc", 
                eta = 0.1, 
                max_depth = 6, 
                subsample = .8
)

set.seed(1234)
xgbcv5 <- xgb.cv(params = params5, 
                 data = xgbMats[[1]], 
                 nrounds = 800, 
                 nfold = 5, 
                 showsd = T, 
                 stratified = T,
                 print_every_n = 10, 
                 early_stopping_rounds = 5, 
                 maximize = T, 
                 prediction = T
)

xgb5 <- xgb.train(params = params5, 
                  data = xgbMats[[1]], 
                  nrounds = xgbcv5$best_iteration, 
                  print_every_n = 10, 
                  early_stopping_rounds = 2,
                  watchlist = list(val = xgbMats[[2]], train = xgbMats[[1]]), 
                  maximize = T
)

xgb3.preds <- as.data.table(predict(xgb3, xgbMats[[3]]))
xgb3.preds <- cbind(ID = test2$ID, segment = xgb3.preds$V1)
xgb3.preds <- as.data.table(xgb3.preds)
fwrite(xgb5.preds, "xgb3.csv")

xgbImp <- xgb.importance(colnames(train2)[-c(1,2)], model = xgb3)
xgb.plot.importance (importance_matrix = xgbImp[1:30])

#Prepare for stacking
trainPred3 <- rbindlist(list(as.data.table(xgbcv3$pred), as.data.table(predict(xgb3, xgbMats[[2]]))))
targetFull <- rbindlist(list(as.data.table(train2[idxTrain, segment]), as.data.table(train2[idxVal, segment])))

metaTrain <- cbind(trainPred3, targetFull)
colnames(metaTrain) <- c("xgb3", "xgb3.segment")

