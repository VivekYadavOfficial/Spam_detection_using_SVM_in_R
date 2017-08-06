#vivekyadavofficial

#load required libraries
library(data.table)
library(caret)
library(kernlab)
library(e1701)
library(doParallel)

#load data
data <- fread("data.csv", sep=";")

#load column names
names <- fread("names.csv", sep=";")

#assign names to data
names(data) <- sapply((1:nrow(names)), function(i) toString(names[i,1]))

#factorize the target variable
data$y <- factor(data$y)

#sample dataset from complete data
sample <- data[sample(1:nrow(data),1000),]

#partition data using caret package
trainIndex <- createDataPartition(sample$y, p=.8, list=FALSE, times=1)
train <- sample[trainIndex]
test <- sample[-trainIndex]

#setting the multicore environment using doParallel package
registerDoParallel()

#set seed
seed(89)

#train svm model
sigDist <- sigest(y~.,data=train,frac=1)

#svm tune grid
svmTuneGrid <- data.frame(.sigma=sigDist[1], .C=2^(-2:7))

#fit model
svm.fit <- train(y~., data=train, preProc=c("center", "scale"), tuneGrid=svmTuneGrid, trControl= trainControl(method="repeatedcv", repeats=5))

#predict
predicted <- predict(svm.fit, test[,1:57])

#confusion matrix to compare
acc <- confusionMatrix(predicted, test$y)