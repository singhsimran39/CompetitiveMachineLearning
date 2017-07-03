library(data.table)
library(stringr)

#400 Non cricket shows, remove bottom 10 cities, hour count, day count

showsTrain <- train[, .(ID, titles)]
showsTest <- test[, .(ID, titles)]

showsTrain[, titles := lapply(titles, function(x) str_replace_all(x, " ", ""))]
showsTrain[, titles := lapply(titles, function(x) str_replace_all(x, "\\|", ""))]
showsTrain[, titles := lapply(titles, function(x) str_replace_all(x, "[^[:alnum:],:]", ""))]
showsTrain[, titles := lapply(titles, function(x) str_replace_all(x, ":[^:\\d+,]", ""))]

showsTest[, titles := lapply(titles, function(x) str_replace_all(x, " ", ""))]
showsTest[, titles := lapply(titles, function(x) str_replace_all(x, "\\|", ""))]
showsTest[, titles := lapply(titles, function(x) str_replace_all(x, "[^[:alnum:],:]", ""))]
showsTest[, titles := lapply(titles, function(x) str_replace_all(as.character(x), ":[^:\\d+,]", ""))]

trainTitles <- lapply(showsTrain$titles, function(x) str_extract_all(x, "[[:alnum:]]+"))
trainTitles <- unlist(trainTitles)
trainTitles <- as.data.table(trainTitles)
trainTitles[, count := .N, by = trainTitles]

testTitles <- lapply(showsTest$titles, function(x) str_extract_all(x, "[[:alnum:]]+"))
testTitles <- unlist(testTitles)
testTitles <- as.data.table(testTitles)
testTitles[, count := .N, by = testTitles]


allTitles <- c(trainTitles$trainTitles, testTitles$testTitles)
allTitles <- as.data.table(allTitles)
allTitles[, count := .N, by = allTitles]
allTitles <- unique(allTitles)
fwrite(allTitles, "allTitles.csv")

#Take all unique titles and then select top 400 non cricket titles. This was done manually by removing titles that looked cricket!!!
nonCricketTitles400 <- fread("Top400NonCricket.csv")

showsTrain[, t1 := lapply(titles, function(k) str_extract_all(string = k, pattern = "[[:alnum:]]+"))]
showsTrain[, t1 := lapply(t1, unlist, use.names=F)]

showsTest[, t1 := lapply(titles, function(k) str_extract_all(string = k, pattern = "[[:alnum:]]+"))]
showsTest[, t1 := lapply(t1, unlist, use.names=F)]


toTitleColumns <- function(data, variables) {
  
  for(i in variables){
    
    data[,paste0(i,"") := lapply(t1, function(x) any(match(i,x)))]
    
  }
  return (data)
  
}

toTitleColumns(showsTrain, nonCricketTitles400$allTitles)
toTitleColumns(showsTest, nonCricketTitles400$allTitles)

for(i in nonCricketTitles400$allTitles) {
  
  pat <- paste0(i, ":[0-9]+")
  
  idx <- which(showsTrain[, paste(i), with = F] != "NA")
  showsTrain[idx, paste(i) := regmatches(as.character(titles), regexpr(pattern = pat, as.character(titles)))]
  
  idx <- which(showsTest[, paste(i), with = F] != "NA")
  showsTest[idx, paste(i) := regmatches(as.character(titles), regexpr(pattern = pat, as.character(titles)))]
  
}

showsTrain <- showsTrain[, -c(1, 2, 3)]
showsTest <- showsTest[, -c(1, 2, 3)]

showsTrain[, colnames(showsTrain) := lapply(.SD, function(x) str_replace_all(x, "[[:alnum:]]+:", "")), 
           .SDcols = colnames(showsTrain)]

showsTest[, colnames(showsTest) := lapply(.SD, function(x) str_replace_all(x, "[[:alnum:]]+:", "")), 
          .SDcols = colnames(showsTest)]

showsTrain[, colnames(showsTrain) := lapply(.SD, as.numeric), .SDcols = colnames(showsTrain)]
showsTest[, colnames(showsTest) := lapply(.SD, as.numeric), .SDcols = colnames(showsTest)]

showsTrain[is.na(showsTrain)] <- 0
showsTest[is.na(showsTest)] <- 0

#Add to train2 and test2
train2 <- cbind(train, showsTrain)
test2 <- cbind(test, showsTest)

#--Saved--
train2 <- train2[, -c(2, 3, 4, 6, 7)]
test2 <- test2[, -c(2, 3, 4, 5, 6)]

#Remove some cities as these were not good features
train2 <- train2[, -c(62:79)]
test2 <- test2[, -c(61:78)]

train2$segment <- as.numeric(train2$segment) - 1

#Prepare XGboost
xgbTrain <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbVal <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbTest <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))


params4 <- list(booster = "gbtree", 
                 objective = "binary:logistic", 
                 eval_metric = "auc", 
                 eta = 0.1, 
                 max_depth = 4, 
                 subsample = .8, 
                 max_delta_step = 5
)

set.seed(1234)
xgbcv4 <- xgb.cv(params = params4, 
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
#0.824365

xgb4 <- xgb.train(params = params4, 
                  data = xgbTrain, 
                  nrounds = xgbcv4$best_iteration, 
                  print_every_n = 10, 
                  early_stopping_rounds = 2,
                  watchlist = list(val = xgbVal, train = xgbTrain), 
                  maximize = T
)
#0.829400

xgb4.preds <- as.data.table(predict(xgb4, xgbTest))
xgb4.preds <- cbind(ID = test2$ID, segment = xgb4.preds$V1)
xgb4.preds <- as.data.table(xgb4.preds)
fwrite(xgb4.preds, "xgb4.csv")

#Prepare for stacking
trainPred4 <- rbindlist(list(as.data.table(xgbcv15$pred), as.data.table(predict(xgb15, xgbVal))))
targetFull <- rbindlist(list(as.data.table(train2[idxTrain, segment]), as.data.table(train2[idxVal, segment])))

metaTrain <- cbind(trainPred4, targetFull)
colnames(metaTrain) <- c("xgb4", "xgb4.segment")
fwrite(metaTrain, "metaTrain_xgb4.csv")

#Importance
xgbImp4 <- xgb.importance(colnames(train2)[-c(1, 2)], model = xgb15)

#Feature Importance
featureList <- names(train2[,-c("ID", "segment")])
featureVector <- c() 
for (i in 1:length(featureList)) { 
  featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
}
write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
xgb.dump(model = xgb15, fname = 'xgb.dump', fmap = "fmap.txt", with_stats = TRUE)

#Bad Features
badFeatures <- setdiff(colnames(train2), xgbImp15$Feature)
badFeatures
badFeatures <- badFeatures[-c(1, 2)]

#Again
train3 <- train2[, -badFeatures, with = F]
test3 <- test2[, -badFeatures, with = F]

train3[, totalTime := log10(totalTime + 10)]
test3[, totalTime := log10(totalTime + 10)]

#Prepare XGboost
xgbTrain2 <- xgb.DMatrix(data = as.matrix(train3[idxTrain, -c("ID", "segment")]), label = train3$segment[idxTrain])
xgbVal2 <- xgb.DMatrix(data = as.matrix(train3[idxVal, -c("ID", "segment")]), label = train3$segment[idxVal])
xgbTest2 <- xgb.DMatrix(data = as.matrix(test3[, -c("ID")]))


params4b <- list(booster = "gbtree", 
                  objective = "binary:logistic", 
                  eval_metric = "auc", 
                  eta = 0.1, 
                  max_depth = 4, 
                  subsample = .8, 
                  max_delta_step = 5
)

set.seed(1234)
xgbcv4b <- xgb.cv(params = params4b, 
                   data = xgbTrain2, 
                   nrounds = 800, 
                   nfold = 5, 
                   showsd = T, 
                   stratified = T,
                   print_every_n = 10, 
                   early_stopping_rounds = 5, 
                   maximize = T, 
                   prediction = T
)
#0.825516

xgb4b <- xgb.train(params = params4b, 
                    data = xgbTrain2, 
                    nrounds = xgbcv4b$best_iteration, 
                    print_every_n = 10, 
                    early_stopping_rounds = 2,
                    watchlist = list(val = xgbVal2, train = xgbTrain2), 
                    maximize = T
)
#0.831084

xgb4b.preds <- as.data.table(predict(xgb4b, xgbTest2))
xgb4b.preds <- cbind(ID = test3$ID, segment = xgb4b.preds$V1)
xgb4b.preds <- as.data.table(xgb4b.preds)
fwrite(xgb4b.preds, "xgb4b.csv")

#Prepare for stacking
trainPred4b <- rbindlist(list(as.data.table(xgbcv4b$pred), as.data.table(predict(xgb4b, xgbVal2))))
targetFullb <- rbindlist(list(as.data.table(train3[idxTrain, segment]), as.data.table(train3[idxVal, segment])))

metaTrainb <- cbind(trainPred4b, targetFullb)
colnames(metaTrainb) <- c("xgb4b", "xgb4b.segment")
fwrite(metaTrainb, "metaTrain_xgb4b.csv")





