stock_close
download.file("http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Research_Data_Factors_daily_CSV.zip")
unzip("F-F_Research_Data_Factors_daily_CSV.zip")
file.remove("F-F_Research_Data_Factors_daily_CSV.zip")
ff <- read.csv("F-F_Research_Data_Factors_daily.CSV", header=T, skip=3)
ff = ff[-nrow(ff),]
ff =  as.data.frame(ff)
ff[,1]= as.Date(ff[,1],"%Y%m%d")
print(ff)
install.packages("tseries")
library(tseries)
library(zoo)
install.packages("forecast")
library(forecast)
install.packages("FinTS")
library(FinTS)
install.packages("rugarch")
library(rugarch)

stock_close <- get.hist.quote(instrument= "^DJI",
                              start = "2016-11-10", 
                              end= "2020-09-30",
                              quote="Close", provider = "yahoo",
                              retclass="zoo")
stock_close<-as.data.frame(stock_close)
stock_date<-rownames(stock_close)
class(stock_date)
install.packages("lubridate")
library(lubridate)
stock_date= ymd(stock_date)
stock_close<-data.frame(cbind(stock_date,stock_close))
str(stock_close)
head(stock_close)
adjdiff =  data.frame(diff(stock_close[,2]))
head(adjdiff)
adjreturn = adjdiff/stock_close[1:nrow(stock_close)-1,2]
adjreturn = data.frame(cbind(stock_close[2:nrow(stock_close),1],adjreturn))
selff = subset(ff,ff$X >= adjreturn[1,1] & ff$X <= adjreturn[nrow(adjreturn),1])

adjreturnmrf = as.data.frame(cbind(selff, adjreturn[,2]-selff$RF/100))
colnames(adjreturnmrf) <- c("X","MKTRF","SMB","HML","RF","RETURNRF")
adjreturnmrf

install.packages("caret")
library(caret)

inTrain <- createDataPartition( adjreturnmrf$RETURNRF  # your target (Y) variable
                               , p=0.70   # % of records you want for training
                               , list=F
)

train <- adjreturnmrf[inTrain,]
test <- adjreturnmrf[-inTrain,]

ctrl <- trainControl(method="cv"          # cv is for k-fold
                     ,number=100           # the k in k-fold
                      # =TRUE for classification-type problems
                     ,summaryFunction = defaultSummary      # use for regression-type problems
                     
)

returnrf <- as.data.frame(adjreturnmrf$RETURNRF)
rm(returnrf)

lmFit <- train(RETURNRF ~ .,   # Target ~ features
               data = train,       # data used to train model
               method = "glmboost",      # model/method type you want to use
               trControl = ctrl,   # incorporates your specified design
               metric = "RMSE")    # statistical performance measure
summary(lmFit)
adjreturnmrf <- as.data.frame(adjreturnmrf)
lm <- defaultSummary(data=data.frame(obs=test$RETURNRF,
                                     pred = predict(lmFit, newdata = test))
                     , model = lmFit)


summary(lm)
preds <- (predict(lmFit))
lm
summary(preds)
acc <- sum(adjreturnmrf$RETURNRF)/sum(lm)
RMSE(test$RETURNRF)
digits.yhat3 <- predict(lmFit)
summary(digits.yhat3)
defaultSummary(digits.yhat3)
barplot(table(digits.yhat3)) 
summary(lm)
RMSE(lm)
defaul
(adjreturnmrf$RETURNRF)
summary(adjreturnmrf)
training <- createDataPartition(adjreturnmrf$RETURNRF , p=0.80, list=FALSE)
test <- adjreturnmrf[-training]


control<- trainControl(method = "cv", number = 10 )
metric <- "RMSE"
x<- adjreturnmrf$RETURNRF
is.na(x)
install.packages("RWeka")
library(RWeka)
install.packages("keras")
library(keras)

set.seed(1234)
fit.lda <- train(RETURNRF~., data=adjreturnmrf, method="pcaNNet", metric=metric, trContol=control)
fit.ld <- train(RETURNRF~., data=adjreturnmrf, method="rf", metric=metric, trContol=control)
fit.l <- train(RETURNRF~., data=adjreturnmrf, method="nnet",tuneGrid = expand.grid(.size = c(5), .decay = 0.1), metric=metric, trContol=control)
fit.ldaa <- train(RETURNRF~., data=adjreturnmrf, method="lda", metric=metric, trContol=control)
fit.ldaaa <- train(RETURNRF~., data=adjreturnmrf, method="glm", metric=metric, trContol=control)


tart_time <- Sys.time()  # capture time before you train your model
digits.m1 <- train(RETURNRF~.,
                   data=adjreturnmrf,
                   method = "rnn",
                   tuneGrid = expand.grid(
                     .size = c(5),
                     .decay = 0.1),
                   trControl = trainControl(method = "none"),
                   MaxNWts = 10000,
                   maxit = 100)
end_time <- Sys.time()
digits.yhat1 <- predict(digits.m1)
digits.ml <- as.data.frame(digits.m1)
str(x)
str(digits.m1)
summary(digits.m1)
warning()
results<- resamples(list(rf=fit.ld, df=fit.lda, fr=fit.l ))

e <- predict_rnn(digits.m1, x, hidden = FALSE, real_output = T)

summary(results)

dotplot(results)
install.packages("tribble") 


autoplot(adjreturnmrf)
adjreturnmrf



adjreturnmrf1 <- as.h2o(adjreturnmrf)
rm(adjreturnmrf)
head(adjreturnmrf1)
install.packages("h2o")



library(h2o)
h2o.init(nthreads=1, max_mem_size="4g")
# prepare the data
y <- "RETURNRF"                                # target variable to learn
x <- setdiff(names(adjreturnmrf1), y)                # feature variables are all other columns
parts <- h2o.splitFrame(adjreturnmrf1, 0.8, seed=99) # randomly partition data into 80/20
train <- parts[[1]]                         # random set of training obs
test <- parts[[2]] 
print(train)
print(test)
m <- h2o.randomForest(x, y, train)

h2o.rmse(m)  
m
h2o.glm(x, y, train)

summary(m)

h2o.hit_ratio_table(m,valid = T)[1,2]
h2o.stackedEnsemble(x, y, train)
p <- h2o.predict(m, test)
as.data.frame(p)
print(y)
as.data.frame(h2o.cbind(p$predict, test$class))


aml <- h2o.automl(x, y, test
                  , max_runtime_secs = 180     # max time to run in seconds
                  , max_models = 4            # max num of models
                  , seed = 123                # for reproducibility.
)
amlt <- h2o.automl(x, y, adjreturnmrf1
                  , max_runtime_secs = 180     # max time to run in seconds
                  , max_models = 4            # max num of models
                  , seed = 123                # for reproducibility.
)
lb <- aml@leaderboard
lbt <- amlt@leaderboard
print(lb, n = nrow(lb))
pred  <- (h2o.predict(aml, test)
predt <- h2o.predict(aml, adjreturnmrf1) 
as.data.frame(pred)
summary(pred)
summary(predt)
as.data.frame(predt)
accu <- h2o.performance(m)
as.data.frame(accu)
lb <- h2o.get_leaderboard(object = aml, extra_columns = 'ALL')
lbt <- h2o.get_leaderboard(object = amlt, extra_columns = 'ALL')
lb <- as.data.frame(lb)
lbt <- as.data.frame(lbt)
summary(test)
print(test)

RMSE(test)

accur <- sum(predt)/(sum(adjreturnmrf1$RETURNRF))

)
amlf <- h2o.automl(x, y, adjreturnmrf1
                   , max_runtime_secs = 180     # max time to run in seconds
                   , max_models = 4            # max num of models
                   , seed = 123                # for reproducibility.
)
 
lbf <- amlf@leaderboard
predf <- h2o.predict(amfl, adjreturnmrf1) 
summary(predf)
as.data.frame(predf)
lbf <- h2o.get_leaderboard(object = amlf, extra_columns = 'ALL')
lbf <- as.data.frame(lbf)
accur <- sum(predf)/(sum(adjreturnmrf1$RETURNRF))



install.packages("forecast")

library(forecast)
ggplot(lbf)
