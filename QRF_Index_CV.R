#Set word dictionary
rm(list = ls()) # clear environment
cat("\014") # clear console
setwd("/Users/bin/Dropbox/Bin - Yaote/Eco Data/Used")

#load packages
library(caret)
library(mlbench)
library(quantregForest)
library(ggplot2)

#Load Dataset
data_econ_factors <- read.csv("./Eco_Step_1_Data.csv")
date_seq <- seq(as.Date("1992/01/01"),as.Date("2015/12/01"),"months")
data_econ_factors[1] <- date_seq
data_econ_factors <- cbind(data_econ_factors[,-1], row.names=(data_econ_factors[,1]))
data_econ_factors <- lapply(data_econ_factors,as.numeric)
data_econ_factors <- as.data.frame(data_econ_factors)
row.names(data_econ_factors) <- date_seq

#Seperate the dependent variables and independent variables
var_ind <- data_econ_factors[,1:23]
var_dep <- data_econ_factors[,24:27]
target <-4
Target <- var_dep[,target]
Target.list <- c("$DJI","$NYA","$IXIC","$GSPC")

fs_data <- as.data.frame(cbind(var_ind,Target))

control <- trainControl(method="timeslice",initialWindow = 230, horizon = 12,
                        fixedWindow = TRUE, allowParallel = FALSE,
                        savePredictions = TRUE)

# train the model "qrf","avNNet","bagEarth"
model <- train(Target~., data=fs_data, method="qrf", preProcess="scale",trControl=control)

# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
#plot(importance)
plot(importance, main = "Important factors")
