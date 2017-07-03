

metaTrain1 <- fread("metaTrain_xgb1.csv")
metaTest1 <- fread("xgb1.csv")

metaTrain2 <- fread("metaTrain_xgb2.csv")
metaTest2 <- fread("xgb2.csv")

metaTrain3 <- fread("metaTrain_xgb3.csv")
metaTest3 <- fread("xgb3.csv")

metaTrain4 <- fread("metaTrain_xgb4.csv")
metaTest4 <- fread("xgb4.csv")

metaTrain5 <- fread("metaTrain_xgb5.csv")
metaTest5 <- fread("xgb5.csv")

metaTrain6 <- fread("metaTrain_xgb6.csv")
metaTest6 <- fread("xgb6.csv")


metaTrainAll <- cbind(metaTrain1, metaTrain2, metaTrain3, metaTrain4, metaTrain5, metaTrain6)
metaTestAll <- cbind(metaTest1, metaTest2, metaTest3, metaTest4, metaTest5, metaTest6)


colnames(metaTrainAll)[11] <- "segment"
colnames(metaTestAll) <- c("ID", "xgb1", "xgb2", "xgb3", "xgb4", "xgb5", "xgb6")

xgbTrain <- xgb.DMatrix(data = as.matrix(metaTrainAll[idxTrain, -c("segment")]), 
                        label = metaTrainAll$segment[idxTrain])
xgbVal <- xgb.DMatrix(data = as.matrix(metaTrainAll[idxVal, -c("segment")]), 
                      label = metaTrainAll$segment[idxVal])
xgbTest <- xgb.DMatrix(data = as.matrix(metaTestAll[, -c("ID")]))


params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eval_metric = "auc", 
               eta = 0.1, 
               max_depth = 3, 
               subsample = .8
)

set.seed(1234)
xgbcv <- xgb.cv(params = params, 
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

xgb <- xgb.train(params = params, 
                 data = xgbTrain, 
                 nrounds = xgbcv$best_iteration, 
                 print_every_n = 10, 
                 early_stopping_rounds = 2,
                 watchlist = list(val = xgbVal, train = xgbTrain), 
                 maximize = T
)

xgb.preds <- as.data.table(predict(xgb, xgbTest))
xgb.preds <- cbind(ID = metaTestAll$ID, segment = xgb.preds$V1)
xgb.preds <- as.data.table(xgb.preds)
fwrite(xgb.preds, "ens.csv")







