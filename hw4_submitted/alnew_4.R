#--hw4 TO567--


#--Import Data--
seat_sales = read.csv('data/SeatSales1.csv')
seat_high = read.csv('data/SeatHigh.csv')

#--Install Packages--
install.packages("tidyverse") 
install.packages("class")
install.packages("caret")
install.packages("e1071")
install.pacakges("rpart")

install.packages("gmodels")
install.packages("vcd")
install.packages("ROCR")

library(class)
library(caret)

#--Decision Tree libraries--
library(C50)
library(rpart)
library(partykit)


# GOAL: We want to develop a model to predict whether sales will be high (Yes) or low (No)
# Part 1. kNN algorithm
#a) Run kNN with k=5 (using SeatsSales1 as training data and SeatHigh as validation data).
head(seat_sales)

nrow(train_x) #--400 rows training data
nrow(val_x) #--100 rows validation data

#--To run knn, your dependent variable should be a factor: check using str()--
train_x = seat_sales[,1:10]
train_y = seat_sales[,11]
dim(train_y)

str(train_y) #--is a factor
table(train_y) #--No:236 Yes:164

val_x = seat_high[,1:10]
val_y = seat_high[,11]
str(val_y) #--is a factor
table(val_y) #--No:63 Yes:37

knn_5_pred = knn(train = train_x, test=val_x,
                 cl=train_y, k=5, prob = TRUE)
knn_5_pred

# Print confusionMatrix with positive = “Yes”
cf_knn5 = confusionMatrix(knn_5_pred, val_y, positive = "Yes")
cm_table5 = cf_knn5$table


# Report Accuracy, Type 1 error rate (False positive) and Type 2 error rate (False Negative)
accuracy = cf_knn5$overall[1]
type1error = 1 - cf_knn5$byClass[2]
type2error = 1 - cf_knn5$byClass[1]

# Run kNN with k=10 (using SeatsSales1 as training data and SeatHigh as validation data).
# Print confusionMatrix with positive = “Yes”
knn_10_pred = knn(train = train_x, test=val_x,
                 cl=train_y, k=10, prob = TRUE)
cf_knn10 = confusionMatrix(knn_10_pred, val_y, positive = "Yes")
cm_table10 = cf_knn10$table
cm_table10

# Report Accuracy, Type 1 error rate (False positive) and Type 2 error rate (False Negative)
accuracy10 = cf_knn10$overall[1]
type1error10 = 1 - cf_knn10$byClass[2]
type2error10 = 1 - cf_knn10$byClass[1]
type1error10
type2error10

# Part 2. CART (use rpart in caret package)
# a) Build a tree to predict whether the sales will be high or not

#--Model--
sales_model_rpart <- rpart(High~., data=seat_sales, method = "class") #--have too many x-axes, so do ~. which says everything else that hasn't been used (or everything besides default)
sales_model_rpart 

#Drawing the tree plot and text  
plot(sales_model_rpart)
text(sales_model_rpart)

printcp(sales_model_rpart)

summary(sales_model_rpart)


#--Making New Prediction-- 
sales_predict_class <- predict(sales_model_rpart, newdata=seat_high, type = "class") 
sales_predict_prob <- predict(sales_model_rpart, newdata=seat_high, type = "prob")

# Confusion Table using gmodels and  caret 
library(gmodels)
library(caret)

CrossTable(sales_predict_class, seat_high$High) #--Create table
confusionMatrix(sales_predict_class, seat_high$High, positive="Yes") #--get results 


#--Build a CART/tree with minsplit=100--
sales_model_rpart_100 = rpart(High~., data=seat_sales, method = "class", control = rpart.control(minsplit=100))
plot(sales_model_rpart_100, uniform=TRUE,
     main="Parameter Tuning with minsplit = 100")
text(sales_model_rpart_100, use.n=TRUE, all=TRUE, cex=1.0)


# Predict testdata 
sales_predict_class <- predict(sales_model_rpart_100, newdata=seat_high, type = "class")
sales_predict_prob <- predict(sales_model_rpart_100, newdata=seat_high, type = "prob")

# Confusion Table. 
CrossTable(sales_predict_class, seat_high$High)
confusionMatrix(sales_predict_class, seat_high$High, positive="Yes")
