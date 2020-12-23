bank<-read.csv(file.choose(),header=T)
names(bank)
attach(bank)
summary(bank)
library(tree)
library(randomForest)

head(bank)

## Check missing values ##
sum(complete.cases(bank))
sum(!complete.cases(bank))

## Split data into training and testing set ##
set.seed(1)
train=sample(nrow(bank),nrow(bank)*0.8)
bank.test=bank[-train,]
deposit.test=deposit[-train]


## Logistic Regression ##
mod1=glm(as.factor(deposit)~., family="binomial",data=bank) # Fit all variables
summary(mod1)

mod2=glm(as.factor(deposit)~job+marital+education+balance+housing+loan+contact+
           month+duration+campaign+poutcome, family="binomial",data=bank) # Fit only significant predictors
summary(mod2)

mod3=glm(as.factor(deposit)~job+marital+education+balance+housing+loan+contact+
           month+duration+campaign+poutcome, family="binomial",data=bank, subset=train) # For prediction
summary(mod3)

glm.probs=predict(mod3,bank.test, type="response")
glm.pred=rep("No",2233)
glm.pred[glm.probs>.5]="Yes"

table(glm.pred,deposit.test) # Confusion Matrix 


## Bagging ##
bag.bank=randomForest(as.factor(deposit)~.,data=bank,subset=train,mtry=16,importance=TRUE) # Fit all 16 predictors in each tree
bag.bank

yhat.bag = predict(bag.bank,newdata=bank[-train,])
table(yhat.bag,deposit.test) # Confusion Matrix 
mean(yhat.bag==deposit.test) # Accuracy of bagging
mean(yhat.bag!=deposit.test)

## Random Forest ##
rf.bank=randomForest(as.factor(deposit)~.,data=bank,subset=train,mtry=5,importance=TRUE) # Fit 4 predictors in each tree
yhat.rf = predict(rf.bank,newdata=bank[-train,])
table(yhat.rf,deposit.test) # Confusion Matrix 
mean(yhat.rf==deposit.test) # Accuracy of random forest
mean(yhat.rf!=deposit.test)

importance(rf.bank)
varImpPlot(rf.bank)

## Classification tree ##
tree.model=tree(as.factor(deposit)~age+as.factor(job)+as.factor(marital)+
                  as.factor(education)+as.factor(default)+balance+as.factor(housing)+
                  as.factor(loan)+as.factor(contact)+day+as.factor(month)+duration+
                  campaign+as.factor(poutcome),bank,subset=train)

cv.model=cv.tree(tree.model,K=10, FUN=prune.misclass) # 10-fold cross validation
cv.model

prune.model=prune.tree(tree.model,best=9)
plot(prune.model)
text(prune.model,pretty=0)

prunetree.pred=predict(prune.model,bank.test,type="class")
table(prunetree.pred,deposit.test) # Confusion Matrix 
