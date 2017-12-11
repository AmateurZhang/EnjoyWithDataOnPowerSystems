library(randomForest)
mydata<-read.csv("file:///C:/Users/thuzhang/Documents/MyFiles/00Seminar/07PROJECT7/DataBase.csv")

attach(mydata)
index <- sample(2,nrow(mydata),replace = TRUE,prob=c(0.8,0.2))
traindata <- mydata[index==1,]
testdata <- mydata[index==2,]

names(mydata)
set.seed(123)
model<-c()
for(i in range(1:5))
{
  rf_ntree <- randomForest(data$LOAD~.,data=traindata,ntree=50)
  model[i]<-rf_ntree
}
model
rf_ntree <- randomForest(LOAD~.,data=traindata,ntree=50)
plot(rf_ntree)
importance(rf_ntree)
varImpPlot(rf_ntree)
iris_rf<-rf_ntree
iris_pred <- predict(model[1],newdata=testdata)

error<-(iris_pred-testdata$LOAD)
hist(error)
sqrt(var(error))/mean(iris_pred)
write.csv(error,"C:/Users/thuzhang/Documents/MyFiles/00Seminar/07PROJECT7/errror.csv")

plot(iris_pred[200:300],type = "l",col="blue")
points(testdata$LOAD[200:300],type = "l",col="red")