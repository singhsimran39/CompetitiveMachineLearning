library(data.table)
library(jsonlite)
library(ranger)
library(purrr)
library(stringr)
library(ggplot2)
library(e1071)
library(tm)
library(xgboost)
library(caret)

hotstar <- fromJSON("train_data_HE.json")
train <- data.table(ID = unlist(names(hotstar)))
train[, ":=" (genres = unlist(lapply(hotstar, '[', 1)),
              titles = unlist(lapply(hotstar, '[', 2)),
              cities = unlist(lapply(hotstar, '[', 3)),
              segment = unlist(lapply(hotstar, '[', 4)),
              dow = unlist(lapply(hotstar, '[', 5)),
              tod = unlist(lapply(hotstar, '[', 6))
)]

hotstarTest <- fromJSON("test_data_HE.json")
test <- data.table(ID  = unlist(names(hotstarTest)))
test[, ":=" (genres = unlist(lapply(hotstarTest, '[', 1)),
             titles = unlist(lapply(hotstarTest, '[', 2)),
             tod = unlist(lapply(hotstarTest, '[', 3)),
             cities = unlist(lapply(hotstarTest, '[', 4)),
             dow = unlist(lapply(hotstarTest, '[', 5))
)]

train[, segment := ifelse(segment == 'neg', 0, 1)]

#Make Table Tennis as one word and other inconsistencies
train[, genres := gsub(pattern = "Table Tennis", replacement = "TableTennis", x = genres)]
test[, genres := gsub(pattern = "Table Tennis", replacement = "TableTennis", x = genres)]
train[, genres := gsub(pattern = "Formula1", replacement = "FormulaOne", x = genres)]
test[, genres := gsub(pattern = "Formula1", replacement = "FormulaOne", x = genres)]
train[, genres := gsub(pattern = "NA", replacement = "NotApp", x = genres)]
test[, genres := gsub(pattern = "NA", replacement = "NotApp", x = genres)]
train[, cities := gsub(pattern = "navi mumbai", replacement = "vashi", x = cities)]
test[, cities := gsub(pattern = "navi mumbai", replacement = "vashi", x = cities)]
train[, cities := gsub(pattern = "new delhi", replacement = "naidilli", x = cities)]
test[, cities := gsub(pattern = "new delhi", replacement = "naidilli", x = cities)]

#Take different genres convert to columns (Refrence - Starter script from HackerEarth)
train[, g1 := lapply(genres, function(k) str_extract_all(string = k, pattern = "[[:alpha:]]+"))]
train[, g1 := lapply(g1, unlist, use.names=F)]

uniq_genres <- unique(unlist(lapply(train$genres, function(k) str_extract_all(string = k, pattern = "[[:alpha:]]+"))))
length(uniq_genres)

toColumns <- function(data, variables){      #Reference - Starter script from HackerEarth
  
  for(i in variables){
    
    data[,paste0(i,"") := lapply(g1, function(x) any(match(i,x)))]
    
  }
  return (data)
  
}

toColumns(train, uniq_genres)
train[, g1 := NULL]

#Add minutes to the genres
for(i in uniq_genres) {
  
  idx <- which(train[, i, with = F] == TRUE)
  pat <- paste0(i, ":[0-9]+")
  train[idx, (i) := regmatches(genres, regexpr(pattern = pat, genres))]
  
}

train[, (uniq_genres) := lapply(.SD, gsub, pattern = "NA", replacement = NA), .SDcols = uniq_genres]
train[, (uniq_genres) := lapply(.SD, gsub, pattern = "[[:alpha:]]+\\:", replacement = ""), .SDcols = uniq_genres]
train[, (uniq_genres) := lapply(.SD, as.numeric), .SDcols = uniq_genres]
train[is.na(train)] <- 0

#Same in test
test[, g1 := lapply(genres, function(k) str_extract_all(string = k, pattern = "[[:alpha:]]+"))]
test[, g1 := lapply(g1, unlist, use.names=F)]

toColumns(test, uniq_genres)
test[, g1 := NULL]

for(i in uniq_genres) {
  
  idx <- which(test[, i, with = F] == TRUE)
  pat <- paste0(i, ":[0-9]+")
  test[idx, (i) := regmatches(genres, regexpr(pattern = pat, genres))]
  
}

test[, (uniq_genres) := lapply(.SD, gsub, pattern = "NA", replacement = NA), .SDcols = uniq_genres]
test[, (uniq_genres) := lapply(.SD, gsub, pattern = "[[:alpha:]]+\\:", replacement = ""), .SDcols = uniq_genres]
test[, (uniq_genres) := lapply(.SD, as.numeric), .SDcols = uniq_genres]
test[is.na(test)] <- 0

#Add cities as columns
#Find 4 word cities, remove space between words or case by case
trainCities4 <- str_extract_all(train$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\:")
trainCities4 <- unlist(trainCities4)
trainCities4 <- as.data.table(trainCities4)
trainCities4 <- unique(trainCities4)
trainCities4

testCities4 <- str_extract_all(test$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\:")
testCities4 <- unlist(testCities4)
testCities4 <- as.data.table(testCities4)
testCities4 <- unique(testCities4)
testCities4

train[, cities := gsub(pattern = "indian institute of technology madras", replacement = "chennai", x = cities)]
test[, cities := gsub(pattern = "indian institute of technology madras", replacement = "chennai", x = cities)]
train[, cities := gsub(pattern = "ramaiah institute of technology", replacement = "bangalore", x = cities)]
test[, cities := gsub(pattern = "ramaiah institute of technology", replacement = "bangalore", x = cities)]
train[, cities := gsub(pattern = "university of new south wales", replacement = "sydney", x = cities)]

#Find 3 word cities, remove space between words or case by case
trainCities3 <- str_extract_all(train$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\:")
trainCities3 <- unlist(trainCities3)
trainCities3 <- unique(trainCities3)
trainCities3 <- as.data.table(trainCities3)

testCities3 <- str_extract_all(test$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\s[[:alpha:]]+\\:")
testCities3 <- unlist(testCities3)
testCities3 <- unique(testCities3)
testCities3 <- as.data.table(testCities3)

train[, cities := gsub(pattern = "jawaharlal nehru university", replacement = "delhi", x = cities)]
test[, cities := gsub(pattern = "jawaharlal nehru university", replacement = "delhi", x = cities)]
train[, cities := gsub(pattern = "sangli-miraj and kupwad", replacement = "sangli", x = cities)]
test[, cities := gsub(pattern = "sangli-miraj and kupwad", replacement = "sangli", x = cities)]
train[, cities := gsub(pattern = "university of sydney", replacement = "sydney", x = cities)]
test[, cities := gsub(pattern = "kilpauk medical college", replacement = "chennai", x = cities)]
train[, cities := gsub(pattern = "east kolkata township", replacement = "kolkata", x = cities)]
train[, cities := gsub(pattern = "university of wollongong", replacement = "sydney", x = cities)]
train[, cities := gsub(pattern = "university of rajshahi", replacement = "dhaka", x = cities)]
train[, cities := gsub(pattern = "troy hills township", replacement = "newjersey", x = cities)]
test[, cities := gsub(pattern = "troy hills township", replacement = "newjersey", x = cities)]

#Find 2 word cities, case by case
trainCities2 <- str_extract_all(train$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\:")
trainCities2 <- unlist(trainCities2)
trainCities2 <- unique(trainCities2)
trainCities2 <- as.data.table(trainCities2)

testCities2 <- str_extract_all(test$cities, "[[:alpha:]]+\\s[[:alpha:]]+\\:")
testCities2 <- unlist(testCities2)
testCities2 <- unique(testCities2)
testCities2 <- as.data.table(testCities2)

test[, cities := gsub(pattern = "basking ridge", replacement = "newjersey", x = cities)]
train[, cities := gsub(pattern = "delhi cantonment", replacement = "delhi", x = cities)]
test[, cities := gsub(pattern = "delhi cantonment", replacement = "delhi", x = cities)]
train[, cities := gsub(pattern = "hillsborough township", replacement = "newjersey", x = cities)]
train[, cities := gsub(pattern = "hsr layout", replacement = "bangalore", x = cities)]
test[, cities := gsub(pattern = "hsr layout", replacement = "bangalore", x = cities)]
train[, cities := gsub(pattern = "jakarta barat", replacement = "jakarta", x = cities)]
train[, cities := gsub(pattern = "jakarta selatan", replacement = "jakarta", x = cities)]
train[, cities := gsub(pattern = "jakarta utara", replacement = "jakarta", x = cities)]
train[, cities := gsub(pattern = "kasturibai nagar", replacement = "chennai", x = cities)]
train[, cities := gsub(pattern = "lajpat nagar", replacement = "delhi", x = cities)]
train[, cities := gsub(pattern = "malad west", replacement = "mumbai", x = cities)]
test[, cities := gsub(pattern = "malad west", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "andheri west", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "mulund west", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "north sydney", replacement = "sydney", x = cities)]
test[, cities := gsub(pattern = "north sydney", replacement = "sydney", x = cities)]
train[, cities := gsub(pattern = "pune cantonment", replacement = "pune", x = cities)]
test[, cities := gsub(pattern = "pune cantonment", replacement = "pune", x = cities)]
train[, cities := gsub(pattern = "thiagarajar college", replacement = "madurai", x = cities)]
train[, cities := gsub(pattern = "vasant kunj", replacement = "delhi", x = cities)]
test[, cities := gsub(pattern = "vasant kunj", replacement = "delhi", x = cities)]
train[, cities := gsub(pattern = "bengaluru", replacement = "bangalore", x = cities)]
test[, cities := gsub(pattern = "bengaluru", replacement = "bangalore", x = cities)]
train[, cities := gsub(pattern = "bhandup west", replacement = "mumbai", x = cities)]
test[, cities := gsub(pattern = "bhandup west", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "monmouth junction", replacement = "newjersey", x = cities)]
train[, cities := gsub(pattern = "prabhadevi", replacement = "mumbai", x = cities)]
test[, cities := gsub(pattern = "prabhadevi", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "forest hill", replacement = "london", x = cities)]
test[, cities := gsub(pattern = "forest hill", replacement = "london", x = cities)]
train[, cities := gsub(pattern = "carlsbad springs", replacement = "ontario", x = cities)]
test[, cities := gsub(pattern = "carlsbad springs", replacement = "ontario", x = cities)]
test[, cities := gsub(pattern = "hosur", replacement = "bangalore", x = cities)]
train[, cities := gsub(pattern = "hosur", replacement = "bangalore", x = cities)]
train[, cities := gsub(pattern = "kakinada", replacement = "mumbai", x = cities)]
test[, cities := gsub(pattern = "kakinada", replacement = "mumbai", x = cities)]
train[, cities := gsub(pattern = "bhayandar", replacement = "mumbai", x = cities)]
test[, cities := gsub(pattern = "bhayandar", replacement = "mumbai", x = cities)]


#Remove spaces from 2 word, 3 word and 4 word cities
train[, cities := lapply(cities, function(x) str_replace_all(x, " ", ""))]
test[, cities := lapply(cities, function(x) str_replace_all(x, " ", ""))]

citiesCount <- lapply(train$cities, function(x) str_extract_all(x, "[[:alpha:]]+"))
citiesCount <- unlist(citiesCount)
citiesCount <- as.data.table(citiesCount)
citiesCount[, count := .N, by = citiesCount]
citiesCount <- unique(citiesCount)
citiesCount[, charCount := str_count(citiesCount)]

#Add city count
train[, cityCount := lapply(cities, function(x) str_count(x, ":"))]
test[, cityCount := lapply(cities, function(x) str_count(x, ":"))]
train$cityCount <- as.numeric(train$cityCount)
test$cityCount <- as.numeric(test$cityCount)

#Add title count
train[, titleCount := lapply(titles, function(x) str_count(x, ":[0-9]+"))]
train$titleCount <- as.numeric(train$titleCount)
test[, titleCount := lapply(titles, function(x) str_count(x, ":[0-9]+"))]
test$titleCount <- as.numeric(test$titleCount)

#Add genres count
train[, genresCount := lapply(genres, function(x) str_count(x, "[[:alpha:]]+"))]
train$genresCount <- as.numeric(train$genresCount)
test[, genresCount := lapply(genres, function(x) str_count(x, "[[:alpha:]]+"))]
test$genresCount <- as.numeric(test$genresCount)

#Add total time watched
train[, totalTime := lapply(genres, function(x) str_split(x, ","))]
train[, totalTime := lapply(totalTime, unlist, use.names = F)]
train[, totalTime := lapply(totalTime, function(x) str_replace_all(as.character(x), "[[:alpha:]]+\\:", ""))]
train[, totalTime := lapply(totalTime, function(x) sum(as.numeric(x)))]
train$totalTime <- as.numeric(train$totalTime)

test[, totalTime := lapply(genres, function(x) str_split(x, ","))]
test[, totalTime := lapply(totalTime, unlist, use.names = F)]
test[, totalTime := lapply(totalTime, function(x) str_replace_all(as.character(x), "[[:alpha:]]+\\:", ""))]
test[, totalTime := lapply(totalTime, function(x) sum(as.numeric(x)))]
test$totalTime <- as.numeric(test$totalTime)

#Make segment as factor
train$segment <- as.factor(train$segment)

#Take out top cities
topCities <- citiesCount[order(count, decreasing = T)]
topCities <- topCities[1:40]                  

#Add top cities columns
train[, c1 := lapply(cities, function(k) str_extract_all(string = k, pattern = "[[:alpha:]]+\\:"))]
train[, c1 := lapply(c1, unlist, use.names=F)]
train[, c1 := lapply(c1, function(x) str_replace_all(as.character(x), ":", ""))]
train[, c1 := lapply(c1, function(x) unique(as.character(x)))]


toCityColumns <- function(data, variables) {       #this adds top city columns(Reference - Starter script on HackerEarth)
  
  for(i in variables){
    
    data[,paste0(i,"") := lapply(c1, function(x) any(match(i,x)))]
    
  }
  return (data)
  
}

toCityColumns(train, topCities$citiesCount)

train[, (topCities$citiesCount) := lapply(.SD, gsub, pattern = "NA", replacement = 0), .SDcols = topCities$citiesCount]
train[, (topCities$citiesCount) := lapply(.SD, gsub, pattern = "TRUE", replacement = 1), .SDcols = topCities$citiesCount]
train[, (topCities$citiesCount) := lapply(.SD, as.character), .SDcols = topCities$citiesCount]


for(i in topCities$citiesCount) {      #this takes the time in each city
  
  idx <- which(train[, i, with = F] == "1")
  pat <- paste0(i, ":\\d+")
  train[idx, (i) := ex_default(as.character(cities), pattern = pat)]
  
}

sum_before_m <- function(x) {
  # Grab all numbers 
  matches <- str_match_all(x, "\\d+")
  # Grab the matches column in the list, transform to numeric, then sum
  sapply(matches, function(y) sum(as.numeric(y)))
}

#Same for test
#Add top cities columns
test[, c1 := lapply(cities, function(k) str_extract_all(string = k, pattern = "[[:alpha:]]+\\:"))]
test[, c1 := lapply(c1, unlist, use.names=F)]
test[, c1 := lapply(c1, function(x) str_replace_all(as.character(x), ":", ""))]
test[, c1 := lapply(c1, function(x) unique(as.character(x)))]

toCityColumns(test, topCities$citiesCount)

test[, (topCities$citiesCount) := lapply(.SD, gsub, pattern = "NA", replacement = 0), .SDcols = topCities$citiesCount]
test[, (topCities$citiesCount) := lapply(.SD, gsub, pattern = "TRUE", replacement = 1), .SDcols = topCities$citiesCount]
test[, (topCities$citiesCount) := lapply(.SD, as.character), .SDcols = topCities$citiesCount]

for(i in topCities$citiesCount) {      #this takes the time in each city
  
  idx <- which(test[, i, with = F] == "1")
  pat <- paste0(i, ":\\d+")
  test[idx, (i) := ex_default(as.character(cities), pattern = pat)]
  
}

#Add minutes corresponding to each city
for(i in topCities$citiesCount) {

  idx <- which(train[, paste0(i)] != 0)
  train[, paste0(i) := lapply(.SD, function(x) as.character(sum_before_m(x)))]

  idx <- which(test[, paste0(i)] != 0)
  test[, paste0(i) := lapply(.SD, function(x) as.character(sum_before_m(x)))]

}

#--Saved--
train[, c1 := NULL]
test[, c1 := NULL]

#IndvsSA has just one non zero, move it to Cricket, remove IndVsSA
train[IndiaVsSa != 0, Cricket := Cricket + IndiaVsSa]
train[, IndiaVsSa := NULL]

test[IndiaVsSa != 0, Cricket := Cricket + IndiaVsSa]
test[, IndiaVsSa := NULL]

#--Saved--

sapply(train, class)
sapply(test, class)

train2 <- train[, -c(2, 3, 4, 6, 7)]
test2 <- test[, -c(2:6)]

sapply(train2, class)
sapply(test2, class)

#----Prepare XGBoost-------
train2$segment <- as.numeric(train2$segment)
train2$segment <- train2$segment - 1

tempTrain <- model.matrix(~.+0, data = train2[, .(hoursCount, daysCount)])
tempTest <- model.matrix(~.+0, data = test2[, .(hoursCount, daysCount)])

train2 <- cbind(train2, tempTrain)
train2[, ":="(hoursCount = NULL, daysCount = NULL)]

test2 <- cbind(test2, tempTest)
test2[, ":="(hoursCount = NULL, daysCount = NULL)]

idx <- sample(1:200000, replace = F)
idxTrain <- idx[1:160000]
idxVal <- idx[160001:200000]

xgbMats <- list()

xgbMats[[1]] <- xgb.DMatrix(data = as.matrix(train2[idxTrain, -c("ID", "segment")]), label = train2$segment[idxTrain])
xgbMats[[2]] <- xgb.DMatrix(data = as.matrix(train2[idxVal, -c("ID", "segment")]), label = train2$segment[idxVal])
xgbMats[[3]] <- xgb.DMatrix(data = as.matrix(test2[, -c("ID")]))

params <- list(booster = "gbtree", 
              objective = "binary:logistic", 
              eval_metric = "auc", 
              eta = 0.1, 
              max_depth = 6, 
              subsample = .8
)

set.seed(1234)
xgbcv <- xgb.cv(params = params, 
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

xgb1 <- xgb.train(params = params, 
                  data = xgbMats[[1]], 
                  nrounds = xgbcv$best_iteration, 
                  print_every_n = 10, 
                  early_stopping_rounds = 2,
                  watchlist = list(val = xgbMats[[2]], train = xgbMats[[1]]), 
                  maximize = T
)

xgb1.preds <- as.data.table(predict(xgb1, xgbMats[[3]]))
xgb1.preds <-cbind(ID = test2$ID, segment = xgb1.preds$V1)
xgb1.preds <- as.data.table(xgb1.preds)
fwrite(xgb1.preds, "xgb1.csv")

xgbImp <- xgb.importance(colnames(train2)[-c(1,2)], model = xgb2)
xgb.plot.importance (importance_matrix = xgbImp[1:30])

