#Add 3 way interactions from XGBFi
train2[, va1 := Cricket * KoffeeWithKaran * totalTime]
test2[, va1 := Cricket * KoffeeWithKaran * totalTime]

train2[, var2 := Cricket * totalTime * totalTime]
test2[, var2 := Cricket * totalTime * totalTime]

train2[, var3 := Cricket * KoffeeWithKaran * Romance]
test2[, var3 := Cricket * KoffeeWithKaran * Romance]

train2[, var4 := Cricket * Drama * Ishqbaaaz]
test2[, var4 := Cricket * Drama * Ishqbaaaz]

train2[, var5 := Cricket * Drama * KoffeeWithKaran]
test2[, var5 := Cricket * Drama * KoffeeWithKaran]

train2[, var6 := Cricket * Romance * YehRishtaKyaKehlataHai]
test2[, var6 := Cricket * Romance * YehRishtaKyaKehlataHai]

train2[, var7 := Cricket * Cricket * Ishqbaaaz]
test2[, var7 := Cricket * Cricket * Ishqbaaaz]

train2[, var8 := Cricket * Romance * KhokaBabu]
test2[, var8 := Cricket * Romance * KhokaBabu]

train2[, var9 := Cricket * NachBaliye * Romance]
test2[, var9 := Cricket * NachBaliye * Romance]

train2[, var10 := Cricket * Cricket * Romance]
test2[, var10 := Cricket * Cricket * Romance]

train2[, var11 := Cricket * Cricket * Drama]
test2[, var11 := Cricket * Cricket * Drama]

train2[, var12 := Cricket * Family * KoffeeWithKaran]
test2[, var12 := Cricket * Family * KoffeeWithKaran]

train2[, var13 := KoffeeWithKaran * totalTime * totalTime]
test2[, var13 := KoffeeWithKaran * totalTime * totalTime]

train2[, var14 := totalTime * totalTime * YehRishtaKyaKehlataHai]
test2[, var14 := totalTime * totalTime * YehRishtaKyaKehlataHai]

train2[, var15 := Action * Cricket * totalTime]
test2[, var15 := Action * Cricket * totalTime]

train2[, var16 := Drama * Ishqbaaaz * KoffeeWithKaran]
test2[, var16 := Drama * Ishqbaaaz * KoffeeWithKaran]

train2[, var17 := Drama * KoffeeWithKaran * Romance]
test2[, var17 := Drama * KoffeeWithKaran * Romance]

train2[, var18 := Cricket * Cricket * totalTime]
test2[, var18 := Cricket * Cricket * totalTime]

train2[, var19 := KoffeeWithKaran * NachBaliye * totalTime]
test2[, var19 := KoffeeWithKaran * NachBaliye * totalTime]

train2[, var20 := Cricket * Drama * Romance]
test2[, var20 := Cricket * Drama * Romance]

cols <- colnames(train2)[c(291:312)]
train2[, (cols) := NULL]
test2[, (cols) := NULL]

xgbTrain <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbVal <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbTest <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))

params6 <- list(booster = "gbtree", 
                 objective = "binary:logistic", 
                 eval_metric = "auc", 
                 eta = 0.1, 
                 max_depth = 4, 
                 subsample = .8, 
                 max_delta_step = 5
)

set.seed(1234)
xgbcv6 <- xgb.cv(params = params6, 
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
#0.825299

xgb6 <- xgb.train(params = params6, 
                   data = xgbTrain, 
                   nrounds = xgbcv6$best_iteration, 
                   print_every_n = 10, 
                   early_stopping_rounds = 2,
                   watchlist = list(val = xgbVal, train = xgbTrain), 
                   maximize = T
)
#0.829895

xgb6.preds <- as.data.table(predict(xgb6, xgbTest))
xgb6.preds <- cbind(ID = test2$ID, segment = xgb6.preds$V1)
xgb6.preds <- as.data.table(xgb6.preds)
fwrite(xgb6.preds, "xgb6.csv")






