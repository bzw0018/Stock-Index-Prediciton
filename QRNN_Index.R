#_____________________________Initilization________________________________________________________
rm(list = ls()) # clear environment
cat("\014") # clear console
setwd("/Users/bin/Dropbox/Bin - Yaote/Eco Data/Used")
set.seed(123)

#Load require package
library(caret)
library(mlbench)
library(qrnn)

#______________________ Load Dataset_____________________________
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

#____________________________________________________________________________
#Create the functions for evaluation 
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

fun_measure_qrnn <- function(actual, predict){
  error = actual - predict
  rmse = sqrt(mean(error^2))
  mae  = mean(abs(error))
  error_percent = abs((actual - predict))/actual
  mape = mean(abs(error_percent))*100
  accuracy = 1-error_percent
  result = matrix(NA, nrow = 3,ncol = 2)
  result[1,1] = "RMSE"
  result[1,2] = rmse
  result[2,1] = "MAE"
  result[2,2] = mae
  result[3,1] = "MAPE"
  result[3,2] = mape
  result <- as.data.frame(result)
  return(result)
}
#____________________________________________________________________________
# Feature selection
# Perpare dataset for feature selection
fs_data <- as.data.frame(cbind(var_ind,Target))
# prepare training scheme
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

# Use qrnn  model for predicting

qrnn.model <- qrnn.fit(x=as.matrix(trainDF[,1:length(trainDF)-1]),
                       y=as.matrix(trainDF[,length(trainDF)]),
n.hidden = 5, tau = 0.5,lower = -0.1, iter.max = 500, n.trials = 1)

qrnn.predict <- qrnn.predict(x=as.matrix(validateDF[,1:length(validateDF)-1]), parms = qrnn.model)

a <- qrnn.predict
b <- validateDF[,length(validateDF)]

#Plot the data
  
plot_data <- data.frame(cbind(a,b))

colnames(plot_data) <- c("Predict", "Actual")

date_seq_new <- tail(date_seq,validate_months)

minY <- min(plot_data$Predict,plot_data$Actual)
maxY <- max(plot_data$Predict,plot_data$Actual)

setwd("/Users/bin/Dropbox/My Research/Paper 2/Paper2 Fig/Revise")

jpeg(filename = paste("QRNN_",Target.list[target],".jpeg"))

par(mai = c(0.8,1,0.5,0.2))

plot(date_seq_new,plot_data$Predict,main = paste("Experiment result of",Target.list[target],
                                                 "Using QRNN")
     ,xlab = "Date", ylab = "Price",type="l", lwd = 2
     ,ylim = c(minY,maxY),
     cex.lab = 1.5, cex.main = 1.5, cex.axis = 1.5, cex.sub = 1.5)
lines(date_seq_new,plot_data$Actual,lty=3,lwd=2)
legend('topright', c("Predict","Actual"), lty=1:ncol(plot_data), cex=1.5)

dev.off()
#Measure the result
fun_measure_qrnn(b,a)
Target.list[target]
