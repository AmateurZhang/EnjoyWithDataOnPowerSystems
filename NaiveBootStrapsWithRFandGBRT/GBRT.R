mydata<-read.csv("file:///C:/Users/thuzhang/Documents/MyFiles/00Seminar/07PROJECT7/DataBase.csv")
attach(mydata)
library(gbm)

index <- sample(2,nrow(mydata),replace = TRUE,prob=c(0.8,0.2))
traindata <- mydata[index==1,]
testdata <- mydata[index==2,]

gbrtmodel<-gbm(LOAD~.,distribution = "gaussian",data = traindata,n.trees = 10000)
best.iter <- gbm.perf(gbrtmodel)
summary.gbm(gbrtmodel,best.iter)

gbrtfit<-predict(gbrtmodel,testdata,best.iter)
error<-(gbrtfit-testdata$LOAD)/testdata$LOAD
mean(error)
mean(abs(error))
hist(error)
sqrt(var(error))

plot(gbrtfit[200:300],type = "l",col="blue")
points(testdata$LOAD[200:300],type = "l",col="red")