library(ggplot2)
library(data.table)
library(caret)
library(xgboost)
library(e1071)
library(tidyr)
library(dplyr)


indTrain <- fread("train_indessa.csv", na.strings = c(" ", "", "n/a", NA))
indTest <- fread("test_indessa.csv", na.strings = c(" ", "", "n/a", NA))

#Exploratory Data Analysis, understading the data
#-----------Find the percentage of loan defaulters-----------
table(indTrain$loan_status)         #23.63% defaulters

#-----------loan amount vs funded amount-----------
a <- indTrain[loan_amnt != funded_amnt, .(.N), by = loan_status]
#about .7% of the data has loan_amount less than fundeed amount and out of that 74% default on loan

#-----------term period of the loan and its relation to loan defaulters-----------
a <- indTrain[, .(.N), by = .(as.factor(term), as.factor(loan_status))]
aa <- spread(a, as.factor.1, N)
aa$percent <- (aa$`1`/(aa$`0`+aa$`1`)) * 100
ggplot(aes(x = as.factor(loan_status), y = funded_amnt), data = indTrain) +
  geom_boxplot() + facet_wrap(~term)
#less people take longer loans but 60 months term has higher median for loan value (seems obvious)

#-----------Employment length vs loan defaulters-----------
a <- as.data.frame(table(indTrain$emp_length, indTrain$loan_status))
ggplot(aes(x = Var1, y = Freq, fill = Var2), data = a) + geom_bar(position = "dodge", stat = "identity")
aa <- a %>% spread(Var2, Freq)
aa$percent <- (aa$`1`/(aa$`0`+aa$`1`)) * 100
ggplot(aes(x = Var1, y = percent), data = aa) + geom_bar(stat = "identity") + 
  scale_y_continuous(breaks = seq(0, 30, 1))
#10+ years has least defaulters

#-----------Annual Income vs loan defaulters-----------
a <- indTrain[, c("annual_inc", "loan_status")]
a$annual_inc <- cut(a$annual_inc, breaks = c(0, 50000, 150000, 500000, 1000000, 10000000), 
                    labels = c("poor", "middle class", "upper middle class", "rich", "filthy rich"), 
                    right = F)
a <- a %>% group_by(annual_inc, loan_status) %>% summarise(cnt = n())
ggplot(aes(annual_inc, cnt, fill = as.factor(loan_status)), data = a) + 
  geom_bar(position = "dodge", stat = "identity")
aa <- a %>% spread(loan_status, cnt)
aa$percent <- (aa$`1`/(aa$`0` + aa$`1`)) * 100
ggplot(aes(x = annual_inc, y = percent), data = aa) + geom_bar(stat = "identity")
#Percentage wise every category is pretty much the same

#-----------Purpose of loan vs defaulting on it-----------
indTrain %>% group_by(purpose, loan_status) %>% summarise(count = n()) %>% 
  ggplot(aes(purpose, count, fill = loan_status)) + geom_bar(position = "dodge", stat = "identity")
a <- as.data.frame(table(indTrain$purpose, indTrain$loan_status))
aa <- a %>% spread(Var2, Freq)
aa$percent <- (aa$`1`/(aa$`0`+aa$`1`)) * 100
ggplot(aes(Var1, percent), data = aa) + geom_bar(stat = "identity")
#Wedding and education have the highest default ratio but they also have the lowest numbers in the dataset

#-----------Do home owners default less-----------
indTrain %>% group_by(home_ownership, loan_status) %>% summarise(count = n()) %>% 
  ggplot(aes(home_ownership, count, fill = loan_status)) + geom_bar(position = "dodge", stat = "identity")

a <- as.data.frame(table(indTrain$home_ownership, indTrain$loan_status))
aa <- spread(a, Var2, Freq)
aa$percent <- (aa$`1`/(aa$`0`+aa$`1`)) * 100
ggplot(aes(Var1, percent), data = aa) + geom_bar(stat = "identity")
#Perfect case of how to lie with graphs. Only consider Mortgage, Own and rent as others have fewer numbers

#-----------Which state has the most defaulters-----------
a <- as.data.frame(table(indTrain$addr_state, indTrain$loan_status))
aa <- spread(a, Var2, Freq)
aa$percent <- (aa$`1`/(aa$`0`+aa$`1`)) * 100
ggplot(aes(Var1, percent), data = aa) + geom_bar(stat = "identity") + 
  scale_y_continuous(breaks = seq(0, 30, 2))
sum(indTrain$loan_status == 1)/nrow(indTrain)      #23.63 national average
#Here IN and TN have low default ratio, again not counting states where the numbers are very low.

table(indTrain$application_type, indTrain$loan_status)
#There are not many joint loan holders, it might be safe to drop this feature

#-----------Annual Income skew-----------
ggplot(aes(annual_inc), data = indTrain) + geom_histogram(binwidth = 25000)
ggplot(aes(log10(annual_inc)), data = indTrain) + geom_histogram(binwidth = .03, fill = "orange", color = "black")

#-----------Total Current balance skew-----------
ggplot(aes(tot_cur_bal), data = indTrain) + 
  geom_histogram(binwidth = 20000, fill = "orange", color = "black")   #again variable is very skewed.

ggplot(aes(log10(tot_cur_bal)), data = indTrain) +
  geom_histogram(binwidth = .05, fill = "orange", color = "black")
#The curve is bimodal if we take the log of the variable

#-----------Total revolving credit skew-----------
ggplot(aes(total_rev_hi_lim), data = indTrain) + 
  geom_histogram(binwidth = 25000, fill = "orange", color = "black")

ggplot(aes(log10(total_rev_hi_lim)), data = indTrain) + 
  geom_histogram(binwidth = .05, fill = "orange", color = "black")
#Perfect normal distribution on a log scale

#==================================================================

#Data Manipulation
#-----------Take the term value out from the string-----------
indTrain[, term := as.integer(str_extract(term, "\\d+"))]
indTest[, term := as.integer(str_extract(term, "\\d+"))]

#-----------Change emp_length to numerical-----------
indTrain[is.na(indTrain$emp_length), emp_length := 0]
indTest[is.na(indTest$emp_length), emp_length := 0]

indTrain[, emp_length := as.integer(str_extract(emp_length, "\\d+"))]
indTest[, emp_length := as.integer(str_extract(emp_length, "\\d+"))]
#I am treating less than 1 year and 1 year as the same

#-----------Change last_week_pay to numerical week value-----------
indTrain[, last_week_pay := as.integer(str_extract(last_week_pay, "\\d+"))]
indTest[, last_week_pay := as.integer(str_extract(last_week_pay, "\\d+"))]

#-----------Change initial_list_status to 0, 1-----------
indTrain[, initial_list_status := as.integer(as.factor(initial_list_status))-1]
indTest[, initial_list_status := as.integer(as.factor(initial_list_status))-1]

#Remove a few features
#-----------Remove description of loan-----------
indTrain[, desc := NULL]
indTest[, desc := NULL]

#-----------verification_status_joint has more than 99% NA, hence removing-----------
indTrain[, verification_status_joint := NULL]
indTest[, verification_status_joint := NULL]

#Title and purpose should be similar, hence removing title-----------
indTrain[, title := NULL]
indTest[, title := NULL]

#-----------Removing batch_enrolled, it does not offer much information-----------
indTrain[, batch_enrolled := NULL]
indTest[, batch_enrolled := NULL]

#-----------Removing application_type because most of the points are of type 1-----------
indTrain[, application_type := NULL]
indTest[, application_type := NULL]

#-----------Remove pymnt_plan because most of the points are of type 1-----------
indTrain[, pymnt_plan := NULL]
indTest[, pymnt_plan := NULL]

#Skewness
#-----------Skew in Annual Incomee-----------
indTrain[is.na(annual_inc), annual_inc := 0]
indTrain[, annual_inc := log(annual_inc + 10)]
ggplot(aes(annual_inc), data = indTrain) + 
  geom_histogram(binwidth = .1, fill = "orange", color = "black")

indTest[is.na(annual_inc), annual_inc := 0]
indTest[, annual_inc := log(annual_inc + 10)]

#Skewness other features
# num_col <- colnames(indTrain)[sapply(indTrain, is.numeric)]
# num_col <- num_col[!(num_col %in% c("member_id", "loan_status"))]
# 
# #in train
# sk <- sapply(indTrain[, num_col, with = F], function(x) skewness(x, na.rm = T))
# sk <- sk[sk > 2]
# 
# indTrain[, names(sk) := lapply(.SD, function(x) log(x + 10)), .SDcols = names(sk)]
# 
# #in test
# sk.t <- sapply(indTest[, num_col, with = F], function(x) skewness(x, na.rm = T))
# sk.t <- sk.t[sk.t > 2]
# 
# indTest[, names(sk.t) := lapply(.SD, function(x) log(x + 10)), .SDcols = names(sk.t)]
#Model performs better without removing skewness.. strange.....

indTrain[, dti:= log10(dti + 10)]
indTest[, dti := log10(dti + 10)]

#-----------for home_ownership there are 3 ANYs move them to OTHER-----------
indTrain[home_ownership == "ANY", home_ownership := "OTHER"]

#-----------Remove member_id-----------
memId_Train <- indTrain[, member_id]
memId_Test <- indTest[, member_id]

indTrain[, member_id := NULL]
indTest[, member_id := NULL]

#-----------OHE some columns-----------
trainTemp <- indTrain[, .(grade, sub_grade, home_ownership, verification_status, purpose)]
testTemp <- indTest[, .(grade, sub_grade, home_ownership, verification_status, purpose)]

train_ex <- model.matrix(~.+0, data = trainTemp)
test_ex <- model.matrix(~.+0, data = testTemp)

newTrain <- cbind(indTrain, train_ex)
newTest <- cbind(indTest, test_ex)

newTrain[, c("grade", "sub_grade", "home_ownership", "verification_status", "purpose") := NULL]
newTest[, c("grade", "sub_grade", "home_ownership", "verification_status", "purpose") := NULL]

char_cols <- colnames(newTrain)[sapply(newTrain, is.character)]
for(i in char_cols) set(x = newTrain, j = i, value = as.integer(as.factor(newTrain[[i]])))

char_cols <- colnames(newTest)[sapply(newTest, is.character)]
for(i in char_cols) set(x = newTest, j = i, value = as.integer(as.factor(newTest[[i]])))

#-----------Remove loan_status from newTrain-----------
newTrain <- newTrain[, -c("loan_status"), with = F]
trainLabel <- as.numeric(indTrain$loan_status)

#-----------Split into train and test-----------
train.X <- newTrain[1:320000, ]
train.Y <- trainLabel[1:320000]
test.X <- newTrain[320001:nrow(newTrain), ]
test.Y <- trainLabel[320001:length(trainLabel)]

#-----------Convert to Matrix, data table to martrix-----------
dtrain <- xgb.DMatrix(data = as.matrix(train.X), label = train.Y, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test.X), label = test.Y, missing = NA)
dtestFull <- xgb.DMatrix(data = as.matrix(newTest), missing = NA)

#-------------XGBoost-------------

#scale_pos_weight = sum(negative)/sum(positive)
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, gamma = 0, max_depth = 6, 
                min_child_weight = 1, subsample = .5, colsample_bytree = .2, max_delta_step = 1, 
                scale_pos_weight = 3.24)

#default parameters
#params <- list(booster = "gbtree", objective = "binary:logistic", eta = .3, gamma = 0, max_depth = 6,
#               min_child_weight = 6, subsample = 1, colsample_bytree = 1)


xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 600, nfold = 5, showsd = T, stratified = T,
                print_every_n = 10, early_stopping_rounds = 20, maximize = F, eval_metric = "auc")


xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 314, print_every_n = 10, early_stopping_rounds = 10,
                  watchlist = list(val = dtest, train = dtrain), maximize = F)

xgb.pred <- predict(xgb1, dtest)
preds <- ifelse(xgb.pred > .5, 1, 0)
cm <- confusionMatrix(preds, test.Y)            #Accuracy ~91.8, 
cm

#-------------Check on the test data provided by HackerEarth-------------
xgb2 <- predict(xgb1, dtestFull)
finalPreds <- data.table(member_id = memId_Test, loan_status = xgb2)
fwrite(finalPreds, "final2.csv")

