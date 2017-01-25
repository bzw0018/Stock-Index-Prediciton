#_____________________________Initilization________________________________________________________
rm(list = ls()) # clear environment
cat("\014") # clear console
setwd("/Users/bin/Dropbox/Bin - Yaote/Eco Data/Used")

#Load require package
library(caret)
library(mlbench)
library(quantregForest)

#______________________ Load Dataset_____________________________
data_econ_factors <- read.csv("./Eco_Step_2_Data.csv")
date_seq <- seq(as.Date("1999/01/01"),as.Date("2015/12/01"),"months")
data_econ_factors[1] <- date_seq
data_econ_factors <- cbind(data_econ_factors[,-1], row.names=(data_econ_factors[,1]))
data_econ_factors <- lapply(data_econ_factors,as.numeric)
data_econ_factors <- as.data.frame(data_econ_factors)
row.names(data_econ_factors) <- date_seq

#Seperate the dependent variables and independent variables
var_ind <- data_econ_factors[,1:23]
var_dep <- data_econ_factors[,24:32]
target <-2 # From 1 to 9
Target <- var_dep[,target]
Target.list <- c("Materials","Energy","Financial","Industrials",
                 "Technology","Consumer Staples","Utilities",
                 "Healthcare","Consumer Discretionary")

#____________________________________________________________________________
#Create the functions for evaluation 
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

fun_measure <- function(actual, predict){
  error = actual - predict
  rmse = sqrt(mean(error^2))
  mae  = mean(abs(error))
  error_percent = abs((actual - predict))/actual
  mape = mean(abs(error_percent))*100
  accuracy = 1-error_percent
  rsq = mean(qrf$rsq)
  result = matrix(NA, nrow = 4,ncol = 2)
  result[1,1] = "RMSE"
  result[1,2] = rmse
  result[2,1] = "MAE"
  result[2,2] = mae
  result[3,1] = "MAPE"
  result[3,2] = mape
  result[4,1] = "RSQ"
  result[4,2] = percent(rsq)
  result <- as.data.frame(result)
  return(result)
}
#____________________________________________________________________________
# Feature selection
# Perpare dataset for feature selection
fs_data <- as.data.frame(cbind(var_ind,Target))
# prepare training scheme
control <- trainControl(method="timeslice",initialWindow = 160, horizon = 12,
                        fixedWindow = TRUE, allowParallel = FALSE,
                        savePredictions = TRUE) # Change initialWindow
# train the model "qrf","avNNet","bagEarth"
model <- train(Target~., data=fs_data, method="qrf", preProcess="scale",trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
#plot(importance)
plot(importance, main = "Important factors")

#___________________________________________________________________________
#Define the important factors
# Need run step 2 before runing the following code
# Get the importance table
imp_temp = as.data.frame(importance$importance)
var_temp = as.data.frame(row.names(imp_temp))
importance_new <- cbind(var_temp,imp_temp)
row.names(importance_new) <- NULL

#Select the most important variables
imp_level <- 0.6
imp_var_table <- subset(importance_new,importance_new$Overall >= imp_level)
imp_var <- as.character(imp_var_table[,1])

#Get the dateset based on the important vars
used_data_ind <- subset(data_econ_factors,select = imp_var)
used_data <- as.data.frame(cbind(used_data_ind, Target))

#Create validation date use last 1 year data
validate_months <- 12 # Use months here

validateDF <- tail(used_data,validate_months)
modelDF <- head(used_data,-validate_months)

# Create the training and testing data sets
splitIndex <- createDataPartition(modelDF$Target, p = .9, list = FALSE, times = 1)
trainDF <- modelDF[splitIndex,]
testDF <- modelDF[-splitIndex,]

# Use quantile regression forest to predict
quantiles <- c(0.1,0.5,0.9)
# qrf <- quantregForest(x=trainDF[,1:length(trainDF)-1],y=trainDF[,length(trainDF)],
#                       importance = TRUE,quantiles=quantiles, keep.inbag = TRUE)
## predict test data using all obervations per node for prediction
# quant.newdata <- predict(qrf, quantiles=quantiles,
#                          newdata= validateDF[,1:length(validateDF)-1],all=TRUE)

qrf <- quantregForest(x=trainDF[,1:length(trainDF)-1],y=trainDF[,length(trainDF)],
                      importance = TRUE,quantiles=quantiles, keep.inbag = TRUE,nodesize = 10)
quant.newdata <- predict(qrf,newdata= validateDF[,1:length(validateDF)-1], what=mean)

result_qrf <- data.frame(cbind(quant.newdata,validateDF$Target))

# fit the model

quant.fit <- predict(qrf,newdata= modelDF[,1:length(modelDF)-1], what = 0.5)
result.fit <- data.frame(cbind(quant.fit,modelDF$Target))
date_seq_fit <- head(date_seq,-validate_months)
plot(date_seq_fit, result.fit$quant.fit,type = "l",col="red")
lines(date_seq_fit,result.fit$V2)
legend('topleft', c("Actual","Predict"), lty=1, col=1:2,cex=1)

# Data for plot

a <- result_qrf$quant.newdata
b <- result_qrf$V2

plot_data <- data.frame(cbind(a,b))

colnames(plot_data) <- c("Predict", "Actual")

date_seq_new <- tail(date_seq,validate_months)

plot(date_seq_new,plot_data$Predict, col="red",main = paste("Prediction for Sector",Target.list[target])
     ,xlab = "Date", ylab = "Price",type="l", lwd = 3)
lines(date_seq_new,plot_data$Actual,lwd=2)
legend('topright', c("Actual","Predict"), col=1:ncol(plot_data), lty=1, cex=.65)

# Measure the result
measurement <- fun_measure(b,a)

