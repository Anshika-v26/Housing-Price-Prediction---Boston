rm(list=ls())#clear all data from your current Global Environment
#load and examine the data

?BostonHousing

install.packages("mlbench")
install.packages("xgboost")
install.packages("Metrics")
install.packages("magrittr")
install.packages("caret")
install.packages("ggplot2")
install.packages("lattice")

library(mlbench)
library(xgboost)
library(Metrics)
library(magrittr)
library(dplyr)
library(caret)
library(ggplot2)
library(lattice)


data(BostonHousing)
head(BostonHousing)
summary(BostonHousing)
str(BostonHousing)

#a.describe()
#BostonHousing.hist()
#plt.show()

cbind(lapply(lapply(BostonHousing, is.na), sum))#no 0 values


#spliting data into training and testing sets
set.seed(10)# for reproducibility
#createDataPartition function- to create training and testing subsets
train.index = createDataPartition(BostonHousing$medv, p = 0.8, list = F)
data.train = BostonHousing[train.index,]
data.test = BostonHousing[-train.index,]

#The distributions of the target variable are similar across the 2 splits:
summary(data.train$medv)
summary(data.test$medv)

#The predictor variables should be rescaled in each subset:
data.train.z = data.train %>% 
  mutate_if(is.numeric ,scale) %>% 
  data.frame()
data.test.z = data.test %>% 
  mutate_if(is.numeric,scale) %>% data.frame()
# add unscaled Y variable back
data.train.z$medv = data.train$medv
data.test.z$medv = data.test$medv


#trainControl() function used to define the type of ‘in­model’ 
#sampling and evaluation undertakento iteratively refine the model. 
#It generates a list of parameters that are passed to the train function 
#thatcreates the model. Here a simple 10 fold cross validation will suffice:
trainControl <- trainControl(method="cv", number=10)

#Then the model can be run over the grid, setting a seed for reproducibility:
## run the model over the grid
set.seed(1)
m.caret <- train(medv ~ ., data=data.train.z, method="xgbLinear",
                 trControl=trainControl, verbose=FALSE, metric="MAE")

## Examine the results
print(m.caret)
# explore the results
names(m.caret)
# see best tune
m.caret[6]


## Find the best parameter combination
# put into a data.frame
grid_df = data.frame(m.caret[4])
# confirm best model
grid_df[which.min(grid_df$results.MAE), ]


## Prediction and Model evaluation
# generate predictions
pred = predict(m.caret, newdata = data.test.z)
# plot these against observed
data.frame(Predicted = pred, Observed = data.test.z$medv) %>%
  ggplot(aes(x = Observed, y = Predicted))+ geom_point(size = 1, alpha = 0.5)+geom_smooth(method = "lm")

# generate some prediction accuracy measures
postResample(pred = pred, obs = data.test.z$medv)

# examine variable importance

varImp(m.caret, scale = FALSE)
       





xgbgrid<- expand.grid(nrounds=c(1,1000),
                      lambda=c(.01,.5),
                      alpha=c(.01, .5),
                      eta= 0.5)

# Tune the model
xgb.tune = train(
  medv ~ .,
  data = data.train.z,
  method = "xgbLinear",
  trControl = trainControl,
  verbose = FALSE,
  metric = "MAE",
  tuneGrid = xgbgrid
)

# evaluate the tuned model using the testing subset
xgb.tuned.pred = predict(xgb.tune, newdata = data.test.z)
postResample(pred = xgb.tuned.pred, obs = data.test.z$medv)


# plot predicted vs actual prices
data.frame(Predicted = xgb.tuned.pred, Observed = data.test.z$medv) %>%
  ggplot(aes(x = Observed, y = Predicted)) + 
  geom_point(size = 1, alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(x = "Actual Prices", y = "Predicted Prices") +
  ggtitle("Predicted vs Actual House Prices FOR TUNED XGBLINER")
postResample(pred = pred, obs = data.test.z$medv)

# examine variable importance

varImp(xgb.tune, scale = FALSE)
