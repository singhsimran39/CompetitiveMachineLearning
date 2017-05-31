library(data.table)
library(lubridate)
library(geosphere)
library(h2o)
options(na.action = "na.pass")
library(e1071)

cabbie <- fread("train.csv", na.strings = "")
train <- cabbie
test <- fread("test.csv", na.strings = "")

#Factor columns
fac_cols <- c("vendor_id", "new_user", "rate_code", "store_and_fwd_flag", "payment_type")
train[, (fac_cols) := lapply(.SD, as.factor), .SDcols = fac_cols]
test[, (fac_cols) := lapply(.SD, as.factor), .SDcols = fac_cols]


#Replace new_user "" with "NO" (convert to char and then change)
train$new_user <- as.character(train$new_user)
train$new_user <- ifelse(is.na(train$new_user), "NO", train$new_user)
train$new_user <- as.factor(train$new_user)
sapply(train, class)

#Remove negative and zero fare amount
train <- train[fare_amount > 0]

#Remove 0 passenger count
train <- train[passenger_count > 0]

#Remove invalid coordinates
train <- train[pickup_longitude > -80 | is.na(pickup_longitude)]
train <- train[TID != "AIX000471323"]

#Remove rows with 0 coordinates (keep NAs)
train <- train[pickup_longitude != 0 | is.na(pickup_longitude)]
train <- train[pickup_latitude != 0 | is.na(pickup_latitude)]
train <- train[dropoff_longitude != 0 | is.na(dropoff_longitude)]
train <- train[dropoff_latitude != 0 | is.na(dropoff_latitude)]

#Add trip distance
dist <- distHaversine(train[, 10:11], train[, 14:15])
train[, trip_distance := dist]

dist2 <- distHaversine(test[, 10:11], test[, 14:15])
dist2 <- format(dist2, scientific = F)
test[, trip_distance := dist2]
test$trip_distance <- as.numeric(test$trip_distance)

#Remove 0 distance
train <- train[trip_distance > 0 | is.na(trip_distance)]

#Only keep less than 5 trip distance if the fare amount is less than 5
train <- train[trip_distance > 5 | (trip_distance <= 5 & fare_amount <= 5) | is.na(trip_distance)]

#Remove distance more than 250 kms
train <- train[trip_distance <= 250000 | is.na(trip_distance)]

#List of invalid coordinates AIX0001906507
train <- train[TID != "AIX0001906507"]

#new user has now become a contant column, rmeove it
train[, new_user := NULL]
test[, new_user := NULL]

#Change to date time format
train$pickup_datetime <- ymd_hms(train$pickup_datetime, tz = "America/New_York")
train$dropoff_datetime <- ymd_hms(train$dropoff_datetime, tz = "America/New_York")

test$pickup_datetime <- ymd_hms(test$pickup_datetime, tz = "America/New_York")
test$dropoff_datetime <- ymd_hms(test$dropoff_datetime, tz = "America/New_York")

#Add duration
train[, trip_duration := (dropoff_datetime - pickup_datetime)]
test[, trip_duration := (dropoff_datetime - pickup_datetime)]
train$trip_duration <- as.numeric(train$trip_duration)
test$trip_duration <- as.numeric(test$trip_duration)

#rather remove 0 trip durations
train <- train[trip_duration > 0 | is.na(trip_duration)]

#Add speed
train[, trip_speed := trip_distance/trip_duration]
test[, trip_speed := trip_distance/trip_duration]

#Remove speeds greater than 40m/sec
train <- train[trip_speed < 40 | is.na(trip_speed)]

#Make passenger factor
train$passenger_count <- as.factor(train$passenger_count)
test$passenger_count <- as.factor(test$passenger_count)

#Add day of week
train[, DoW := as.factor(wday(pickup_datetime))]
test[, DoW := as.factor(wday(pickup_datetime))]

#Add hour of day
train[, hourOfDay := hour(pickup_datetime)]
test[, hourOfDay := hour(pickup_datetime)]
train$hourOfDay <- as.factor(train$hourOfDay)
test$hourOfDay <- as.factor(test$hourOfDay)

#Make mta_tax as fator
train$mta_tax <- as.factor(train$mta_tax)
test$mta_tax <- as.factor(test$mta_tax)

#Make surcharge either 0, 0.5 or 1
train[surcharge == 0.01, surcharge := 0]
train[surcharge > 1, surcharge := 1]

#====================KMeans===================================
h2o.init(nthreads = -1, max_mem_size = "6G")

pickups <- as.h2o(train[, 9:10])
pickup_KM <- h2o.kmeans(pickups, k = 30, max_iterations = 2000)

pickup_region <- h2o.predict(pickup_KM, pickups)
pickup_region <- as.data.table(pickup_region)
pickup_region$predict <- as.factor(pickup_region$predict)
colnames(pickup_region) <- "pickup_region"
table(pickup_region$pickup_region)

dropoffs <- as.h2o(train[, 13:14])
dropoff_KM <- h2o.kmeans(dropoffs, k = 30, max_iterations = 2000)

dropoff_region <- h2o.predict(dropoff_KM, dropoffs)
dropoff_region <- as.data.table(dropoff_region)
dropoff_region$predict <- as.factor(dropoff_region$predict)
colnames(dropoff_region) <- "dropoff_region"
table(dropoff_region$dropoff_region)

train <- cbind(train, pickup_region)
train <- cbind(train, dropoff_region)


#For test data
pickups2 <- as.h2o(test[, 9:10])
pickup2_KM <- h2o.kmeans(pickups2, k = 30, max_iterations = 2000)

pickup2_region <- h2o.predict(pickup2_KM, pickups2)
pickup2_region <- as.data.table(pickup2_region)
pickup2_region$predict <- as.factor(pickup2_region$predict)
colnames(pickup2_region) <- "pickup_region"

dropoffs2 <- as.h2o(test[, 13:14])
dropoff2_KM <- h2o.kmeans(dropoffs2, k = 30, max_iterations = 2000)

dropoff2_region <- h2o.predict(dropoff2_KM, dropoffs2)
dropoff2_region <- as.data.table(dropoff2_region)
dropoff2_region$predict <- as.factor(dropoff2_region$predict)
colnames(dropoff2_region) <- "dropoff_region"

test <- cbind(test, pickup2_region)
test <- cbind(test, dropoff2_region)

#take TID
tidTrain <- train$TID
train[, TID := NULL]

tidTest <- test$TID
test[, TID := NULL]

#Remove dates
train[, ":="(pickup_datetime = NULL, dropoff_datetime = NULL)]
test[, ":="(pickup_datetime = NULL, dropoff_datetime = NULL)]
sapply(train, class)
sapply(test, class)

#change fare amount to log(fare_amount)
train[, fare_amount := log10(fare_amount)]


#=====================H2O setup======================================
h2o.train <- as.h2o(train, "train.hex")
h2o.test <- as.h2o(test, "test.hex")

xd <- h2o.splitFrame(h2o.train, ratios = c(0.8, 0.1), seed = 1234)

split_train <- h2o.assign(xd[[1]], "split_train.hex")
split_val <- h2o.assign(xd[[2]], "split_val.hex")
split_test <- h2o.assign(xd[[3]], "split_test.hex")

y <- "fare_amount"
x <- colnames(h2o.train)[-14]

gbm_2805_lucky_1 <- h2o.gbm(x = x,
                            y = y,
                            training_frame = split_train, 
                            validation_frame = split_val, 
                            stopping_metric = "MAE", 
                            model_id = "gbm_2805_lucky_1", 
                            stopping_rounds = 5, 
                            stopping_tolerance = 0.0001, 
                            ntrees = 300,
                            max_depth = 18,
                            learn_rate = 0.05,
                            col_sample_rate = 0.9, 
                            sample_rate = 0.9, 
                            score_tree_interval = 10,
                            seed = 1234
)

summary(gbm_2805_lucky_1)

gbm_2805_lucky_1.preds <- as.data.table(h2o.predict(gbm_2805_lucky_1, h2o.test))
gbm_2805_lucky_1.preds$predict <- 10^(gbm_2805_lucky_1.preds$predict)
gbm_2805_lucky_1.preds$predict <- gbm_2805_lucky_1.preds$predict - 10
gbm_2805_lucky_1.preds <- cbind(tidTest, gbm_2805_lucky_1.preds)
colnames(gbm_2805_lucky_1.preds) <- c("TID", "fare_amount")
fwrite(gbm_2805_lucky_1.preds, "gbm_2805_lucky_1.csv")


#========Grid search for max depth==========
maxDepth_params <- list(max_depth = c(16, 18, 20, 22, 24))

maxDepth_grid <- h2o.grid(hyper_params = maxDepth_params, 
                          search_criteria = list(strategy = "Cartesian"), 
                          algorithm = "gbm", 
                          grid_id = "maxDepth_grid", 
                          x = x, y = y, 
                          training_frame = split_train, 
                          validation_frame = split_val, 
                          ntrees = 1000, 
                          learn_rate = 0.05, 
                          sample_rate = 0.8, 
                          col_sample_rate = 0.8, 
                          stopping_rounds = 5, 
                          stopping_tolerance = 0.00001, 
                          stopping_metric = "MAE", 
                          score_tree_interval = 10
)

sorted.maxDepthGrid <- h2o.getGrid("maxDepth_grid", sort_by = "MAE")
sorted.maxDepthGrid

#========Full grid search==========
hyper_params <- list(max_depth = c(16, 18, 19, 20), 
                     sample_rate = seq(0.6,1,0.05), 
                     col_sample_rate = seq(0.6,1,0.05), 
                     col_sample_rate_per_tree = seq(0.6,1,0.05), 
                     col_sample_rate_change_per_level = seq(0.9,1.1,0.05)
)

search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 50, 
                        stopping_rounds = 5,                
                        stopping_metric = "MAE",
                        stopping_tolerance = 0.00001, 
                        seed = 1234
)


full_grid <- h2o.grid(hyper_params = hyper_params, 
                      search_criteria = search_criteria, 
                      algorithm = "gbm", 
                      grid_id = "full_grid", 
                      x = x, 
                      y = y, 
                      training_frame = split_train, 
                      validation_frame = split_val, 
                      ntrees = 1000, 
                      learn_rate = 0.02, 
                      max_runtime_secs = 3600, 
                      stopping_rounds = 5, 
                      stopping_tolerance = 0.00001, 
                      stopping_metric = "MAE", 
                      score_tree_interval = 10, 
                      seed = 1234
)

summary(full_grid)


#=====taking the best models from the grid.
gbm_2905_1 <- h2o.gbm(x = x,
                      y = y,
                      training_frame = split_train, 
                      validation_frame = split_val, 
                      stopping_metric = "MAE", 
                      model_id = "gbm_2905_1", 
                      stopping_rounds = 5, 
                      stopping_tolerance = 0.00001, 
                      ntrees = 1000,
                      max_depth = 18,
                      learn_rate = 0.02,
                      sample_rate = 0.7,
                      col_sample_rate = 0.95, 
                      col_sample_rate_change_per_level = 1.05, 
                      col_sample_rate_per_tree = 0.95, 
                      score_tree_interval = 10,
                      seed = 1234
)

summary(gbm_2905_1)

#On split_test
s_t <- as.data.table(split_test[14])
a <- as.data.table(h2o.predict(gbm_2905_1, split_test))
head(a)
sum(abs(a$predict - s_t$fare_amount))/161580



gbm_2905_2 <- h2o.gbm(x = x,
                      y = y,
                      training_frame = split_train, 
                      validation_frame = split_val, 
                      stopping_metric = "MAE", 
                      model_id = "gbm_2905_2", 
                      stopping_rounds = 5, 
                      stopping_tolerance = 0.00001, 
                      ntrees = 1000,
                      max_depth = 20,
                      learn_rate = 0.02,
                      sample_rate = 0.6,
                      col_sample_rate = 0.9, 
                      col_sample_rate_per_tree = 0.95, 
                      score_tree_interval = 10,
                      seed = 1234
)

summary(gbm_2905_2)

#On split_test 
b <- as.data.table(h2o.predict(gbm_2905_2, split_test))
head(b)
sum(abs(b$predict - s_t$fare_amount))/161580
#this is better

gbm_2905_2.preds <- as.data.table(h2o.predict(gbm_2905_2, h2o.test))
head(gbm_2905_2.preds)
gbm_2905_2.preds$predict <- 10^(gbm_2905_2.preds$predict)
gbm_2905_2.preds <- cbind(tidTest, gbm_2905_2.preds$predict)
gbm_2905_2.preds <- as.data.table(gbm_2905_2.preds)
colnames(gbm_2905_2.preds) <- c("TID", "fare_amount")
fwrite(gbm_2905_2.preds, "finalSubmission.csv")



