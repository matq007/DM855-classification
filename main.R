# Assignment 3

# Decission tree
# Regression
# Neural network

#install.packages("neuralnet")

require(neuralnet)
require(nnet)
require(1e1071)
require(caret)
require(ggplot2)
setwd("~/School/Github/DM855-classification")

# Preparation
d1 = read.table("student-mat.csv",sep=";",header=TRUE)

ggplot(data = d1, mapping = aes(x = G3, y = romantic)) + geom_jitter(aes(colour = G3))
ggplot(data = d1, mapping = aes(x = G3, y = Mjob)) + geom_jitter(aes(colour = G3))

dataset <- NULL
dataset <- cbind(dataset, school = as.factor(d1$school))
dataset <- cbind(dataset, address = as.factor(d1$address))
dataset <- cbind(dataset, famsize = as.factor(d1$famsize))
dataset <- cbind(dataset, pstatus = as.factor(d1$Pstatus))
dataset <- cbind(dataset, medu = as.factor(d1$Medu))
dataset <- cbind(dataset, fedu = as.factor(d1$Fedu))
dataset <- cbind(dataset, mjob = as.factor(d1$Mjob))
dataset <- cbind(dataset, guardian = as.factor(d1$guardian))
dataset <- cbind(dataset, reason = as.factor(d1$reason))
dataset <- cbind(dataset, traveltime = as.factor(d1$traveltime))
dataset <- cbind(dataset, studytime = as.factor(d1$studytime))
dataset <- cbind(dataset, failures = as.factor(d1$failures))
dataset <- cbind(dataset, schoolsup = as.factor(d1$schoolsup))
dataset <- cbind(dataset, famsup = as.factor(d1$famsup))
dataset <- cbind(dataset, extraclass = as.factor(d1$paid))
dataset <- cbind(dataset, activities = as.factor(d1$activities))
dataset <- cbind(dataset, heigher = as.factor(d1$higher))
dataset <- cbind(dataset, internet = as.factor(d1$internet))
dataset <- cbind(dataset, romantic = as.factor(d1$romantic))
dataset <- cbind(dataset, familiyrep = as.factor(d1$famrel))
dataset <- cbind(dataset, freetime = as.factor(d1$freetime))
dataset <- cbind(dataset, goout = as.factor(d1$goout))
dataset <- cbind(dataset, dalc = as.factor(d1$Dalc))
dataset <- cbind(dataset, walc = as.factor(d1$Walc))
dataset <- cbind(dataset, health = as.factor(d1$health))
dataset <- cbind(dataset, absence = as.factor(d1$absences))
dataset <- cbind(dataset, G1 = as.factor(d1$G1))
dataset <- cbind(dataset, G2 = as.factor(d1$G2))
dataset <- cbind(dataset, G3 = as.factor(d1$G3))
dataset <- as.data.frame(dataset)
dataset_end <- length(colnames(dataset)) - 1

# Correlation calculation
# tmp <- cor(dataset)
# highlyCorrelated <- findCorrelation(tmp, cutoff=0.5)
# print(highlyCorrelated)

# Visualization 
pca <- prcomp(dataset[1:dataset_end])
plot(pca$x[,1:2], col = dataset$G3)
hist(dataset$G3)

# Preprocessing
train_split = floor(0.70 * nrow(dataset))
index = sample(seq_len(nrow(dataset)), size = train_split)
train_data <- dataset[index,]
test_data <- dataset[-index,]

# KNN
# model <- train(G3 ~ ., data=subset(train_data, select = -c(G1,G2)), method = "knn")
model <- train(G3 ~ ., data=train_data, method = "knn")
pred <- predict(model, subset(test_data, select = -G3))
pred <- sapply(pred,round,digits=0)
res <- data.frame(actual = test_data$G3, predicted = pred)
matrix <- confusionMatrix(
  factor(res$actual, levels=1:20),
  factor(res$predicted, levels=1:20)
)
matrix$overall['Accuracy']
plot(res$predicted, res$actual, xlab = "predicted", ylab = "actual", main = "")
abline(0,1)

# SVM
studentModel <- train(G3 ~ ., data=train_data, method = "svmLinear")
studentTestPred <- predict(studentModel, subset(test_data, select = -G3))
studentTestPred <- sapply(studentTestPred,round,digits=0)
matrix <- confusionMatrix(
  factor(studentTestPred, levels=1:20),
  factor(test_data$G3, levels=1:20)
)
matrix$overall['Accuracy']
plot(res$predicted, res$actual, xlab = "predicted", ylab = "actual", main = "SVM")
abline(0,1)
importance <- varImp(model, scale=FALSE)
plot(importance)

# Linear regression
model <- lm(G3 ~ ., data = train_data)
pred <- predict(model, subset(test_data, select = -G3))
pred <- sapply(pred,round,digits=0)
res <- data.frame(actual = test_data$G3, predicted = pred)
matrix <- confusionMatrix(
  factor(res$predicted, levels=1:20),
  factor(res$actual, levels=1:20)
)
matrix$overall['Accuracy']
mse <- mean((test_data$G3 - pred)^2)
plot(res$predicted, res$actual, xlab = "predicted", ylab = "actual", main = "Linear regression")
abline(0,1)

# GLM
model <- glm(G3 ~ ., data = train_data)
pred <- predict(model, subset(test_data, select = -G3))
pred <- sapply(pred,round,digits=0)
res <- data.frame(actual = test_data$G3, predicted = pred)
matrix <- confusionMatrix(
  factor(res$predicted, levels=1:20),
  factor(res$actual, levels=1:20)
)
matrix$overall['Accuracy']
mse <- mean((pred - test_data$G3)^2)
plot(res$predicted, res$actual, xlab = "predicted", ylab = "actual", main = "GLM")
abline(0,1)


# Neural network
dataset$reason <- NULL
dataset$reason <- class.ind(as.factor(d1$reason))
formula <- as.formula(
  paste("G3 ~ ",
        paste(colnames(dataset)[1:dataset_end], collapse = "+"),
        sep = "")
)

maxs <- apply(dataset, 2, max) 
mins <- apply(dataset, 2, min)
scaled <- as.data.frame(scale(dataset, center = mins, scale = maxs - mins))
set.seed(13)

accuracy <- NULL
for (i in 1:30) {
  train_split = floor(0.75 * nrow(scaled))
  index = sample(seq_len(nrow(scaled)), size = train_split)
  train_data <- scaled[index,]
  test_data <- scaled[-index,]
  
  nn <- neuralnet(formula, data = train_data, 
                  hidden = c(31),
                  act.fct = "logistic",
                  linear.output = F,
                  lifesign = "minimal",
                  threshold = 0.01)
  pred <- compute(nn, test_data[1:dataset_end])
  pred_ <- pred$net.result*(max(dataset$G3)-min(dataset$G3))+min(dataset$G3)
  test_ <- (test_data$G3)*(max(dataset$G3)-min(dataset$G3))+min(dataset$G3)
  res <- data.frame(actual = test_, predicted = pred_)
  res$predicted <- sapply(res$predicted,round,digits=0)
  
  matrix <- confusionMatrix(
    factor(res$predicted, levels=1:20),
    factor(res$actual, levels=1:20)
  )
  accuracy[i] <- matrix$overall['Accuracy']
}
mean(accuracy)
