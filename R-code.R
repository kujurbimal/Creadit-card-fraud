library(readr)
creditcard <- read_csv("D:/nielit/pro00/credit card fraud detection with ML/creditcard.csv")
View(creditcard)
library(data.table)
install.packages("ranger")
install.packages("caret")
head(creditcard)
dim(creditcard)
tail(creditcard)
table(creditcard$Class)
summary(creditcard$Amount)
names(creditcard)
var(creditcard$Amount)
sd(creditcard$Amount)
creditcard$Amount=scale(creditcard$Amount)
NewData=creditcard[,-c(1)]
head(NewData)
install.packages("caTools")
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)
Logistic_Model=glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)
install.packages("pROC")
lr.predict <- predict(Logistic_Model,train_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")
Logistic_Model=glm(Class~.,train_data,family=binomial())
summary(Logistic_Model)
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , creditcard, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard, type = 'class')
probability <- predict(decisionTree_model, creditcard, type = 'prob')
rpart.plot(decisionTree_model)
library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)
predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)
library(gbm, quietly=TRUE)
# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
#Plot the gbm model
plot(model_gbm)
# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)

#recursive partitioning
party_tree <- ctree(formula=Amount~. , data = creditcard)



