library(data.table)
library(corrplot)
library(xgboost)

qmTrain <- fread("train.csv")
qmTrain$timestamp <- as.numeric(qmTrain$timestamp)
qmTrain$Stock_ID <- as.numeric(qmTrain$Stock_ID)

qmTest <- fread("test.csv")
qmTest$timestamp <- as.numeric(qmTest$timestamp)
qmTest$Stock_ID <- as.numeric(qmTest$Stock_ID)

setDT(qmTrain)
setDT(qmTest)

train.X <- qmTrain[1:421650, -c("ID", "Outcome", "True_Range"), with = F]
train.Y <- as.numeric(qmTrain$Outcome[1:421650])
test.X <- qmTrain[421651:702739, -c("ID", "Outcome", "True_Range"), with = F]
test.Y <- as.numeric(qmTrain$Outcome[421651:702739])

testFull.X <- qmTest[, -c("ID", "True_Range"), with = F]

dtrain <- xgb.DMatrix(data = as.matrix(train.X), label = train.Y, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test.X), label = test.Y, missing = NA)
dtestFull <- xgb.DMatrix(data = as.matrix(testFull.X), missing = NA)


params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, gamma = 0, max_depth = 6, 
               min_child_weight = 1, subsample = 1, max_delta_step = 0, eval_metric = "logloss")

xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 300, nfold = 5, showsd = T, stratified = T,
                print_every_n = 10, early_stopping_rounds = 10, maximize = F)


xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 262, print_every_n = 10, early_stopping_rounds = 10,
                  watchlist = list(val = dtest, train = dtrain), maximize = F)



#First
xgb.pred_1 <- predict(xgb1, dtest)
preds <- ifelse(xgb.pred_1 > .5, 1, 0)
xgb_1 <- predict(xgb1, dtestFull)
finalPreds_1 <- data.table(ID = qmTest$ID, Outcome = xgb_1)
fwrite(finalPreds, "final1.csv")

#Second
xgb.pred_2 <- predict(xgb1, dtest)
preds <- ifelse(xgb.pred_2 > .5, 1, 0)
xgb_2 <- predict(xgb1, dtestFull)
a <- (xgb_1 + xgb_2)/2
a <- ifelse(a > .5, 1, 0)
finalPreds_2 <- data.table(ID = qmTest$ID, Outcome = a)
fwrite(finalPreds, "final2.csv")

#Third
xgb.pred_3 <- predict(xgb1, dtest)
xgb_3 <- predict(xgb1, dtestFull)
a <- (xgb_1 + xgb_2 + xgb_3)/3
a <- ifelse(a > .5, 1, 0)
finalPreds_3 <- data.table(ID = qmTest$ID, Outcome = a)
fwrite(finalPreds, "final3.csv")

#Fourth
xgb.pred_4 <- predict(xgb1, dtest)
xgb_4 <- predict(xgb1, dtestFull)
a <- (xgb_1 + xgb_2 + xgb_3 + xgb_4)/4
a <- ifelse(a > .5, 1, 0)
finalPreds_3 <- data.table(ID = qmTest$ID, Outcome = a)
fwrite(finalPreds, "final5.csv")


  