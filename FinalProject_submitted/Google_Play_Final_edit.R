convb <- function(x){
  ptn <- "(\\d*(.\\d+)*)(.*)"
  num  <- as.numeric(sub(ptn, "\\1", x))
  unit <- sub(ptn, "\\3", x)             
  unit[unit==""] <- "1" 
  
  mult <- c("1"=1, "k"=1024, "M"=1024^2, "G"=1024^3)
  num * unname(mult[unit])
}


###############
#--Import Data--
###############
data = read.csv('googleplaystore.csv')
rawdata <- data
#####################
#--Import Libraries--
#####################
library(tidyverse)
library(ggplot2)
library(GGally)

library(class)
library(rpart)
library(caret)
library(e1071)

library(gmodels)
library(vcd)
library(ROCR)
library(C50)
library(partykit)
library(RWeka)
library(kernlab)
library(caretEnsemble)
library(mlbench)
library(dplyr)


##############
#--Clean Data--
##############

#--determine data types--
sapply(data, class)

head(data)

#--determine number of null rows by column --
colSums(is.na(data))

#--Look for any incorrect data --
table(data$Rating) # Has one '19'
table(data$Price) # Has a row with 'Everyone'
table(data$Android.Ver) # Too many "Varies with device"
table(data$Type) # a couple of rows with messy data (0s, hard-coded "NaN")
table(data$Installs) # "Free"

#--Remove row with 'Everyone', remove '19', remove 'Free', remove N/As--
levels(data$Price)
data <- data[data$Price!="Everyone",]
data <- data[data$Rating!="19",]
data <- data[data$Installs!="Free",]
length(data$Price)
data <- na.omit(data)
str(data)
summary(data)

#--Convert data types from Categorical to Numeric, store in new columns--
data$Reviews = as.numeric(as.factor(data$Reviews)) 
data$Categ_num = as.numeric(as.factor(data$Category))  
data$Type_num = as.numeric(as.factor(data$Type))
data$Genre_num = as.numeric(as.factor(data$Genre))
data$ContentRating_num = as.numeric(as.factor(data$Content.Rating))
data$Android.Ver = as.numeric(data$Android.Ver)

#--Regex on Installs and Price column, convert to Numeric, then store in new column--
data$Installs_num = gsub("[[:punct:]]", "", data$Installs)
data$Installs_num = as.numeric(data$Installs_num)

data$Price_num = gsub("\\$", "", as.character(data$Price))
data$Price_num = as.numeric(data$Price_num)

str(data)
data <- unique(data,by="App")
length(unique(data$App))
length(data$App)

#View(data)

#--Add column for above/below 1M installs, this will be our target (1=Blockbuster, 0=Not Blockbuster)--
data$above1M = 'NotBlockbuster'  #--not blockbuster
data$above1M[data$Installs_num >= 1000000] = 'Blockbuster' #--blockbuster
data$above1M = as.factor(data$above1M)

#--Split looks semi-balanced
#   0    1 
#6429 4411 

#--Dataset for models--
data2 = subset(data, select =c(3,4,14,15,16,17,19,20))
head(data2)
sapply(data2, class)
summary(data2)

#--Split Data into Train and Validation--
set.seed(567)
split_tr <- 0.75*nrow(data) #--10840
train_rows = sample(nrow(data),split_tr) #75/25 split for train and validation data
train = data2[train_rows, ]
val = data2[-train_rows, ]


########################
#-- Models
########################

# setting up table rows
acc <- NULL
fp <- NULL
fn <- NULL
k <- NULL

# Model 1: Naive Bayes
library(e1071)
google_nb = naiveBayes(above1M~. , data=train)
#What does the model say? Print the model summary
pred_nb <- predict(google_nb, newdata=val)
cm_nb <- confusionMatrix(pred_nb, val$above1M, positive = 'Blockbuster')
cm_nb

acc <- cbind(acc, NB = cm_nb$overall["Accuracy"])
fp <- cbind(fp, NB = 1 - cm_nb$byClass["Specificity"])
fn <- cbind(fn, NB = 1 - cm_nb$byClass["Sensitivity"])
k <- cbind(k, NB = cm_nb$overall["Kappa"])


# Model 2: rpart
library(rpart)
google_rpart1 <- rpart(above1M~., data=train, method = "class", control = rpart.control(minsplit=100))
google_rpart1
plot(google_rpart1)
text(google_rpart1, pretty= FALSE, cex =1.3)
pred_rpart1c <- predict(google_rpart1, newdata=val, type = "class") 

library(caret)
cm_rp <- confusionMatrix(pred_rpart1c, val$above1M, positive = "Blockbuster")
cm_rp

acc <- cbind(acc, RPart = cm_rp$overall["Accuracy"])
fp <- cbind(fp, RPart = 1 - cm_rp$byClass["Specificity"])
fn <- cbind(fn, RPart = 1 - cm_rp$byClass["Sensitivity"])
k <- cbind(k, RPart = cm_rp$overall["Kappa"])

# Model 3: C50
#Tree Review with 5 tree boosting  
library(C50)
google_c5 <- C5.0(above1M~., data= train, trials=5) # Training. 
# plot(google_c5,0)
plot(google_c5, 4)  #--this tree is simpler than the one above

summary(google_c5)
pred_c5 <- predict(google_c5, newdata=val, type = "class") 
cm_c50 <- confusionMatrix(pred_c5, val$above1M, positive = "Blockbuster")

acc <- cbind(acc, C50 = cm_c50$overall["Accuracy"])
fp <- cbind(fp, C50 = 1 - cm_c50$byClass["Specificity"])
fn <- cbind(fn, C50 = 1 - cm_c50$byClass["Sensitivity"])
k <- cbind(k, C50 = cm_c50$overall["Kappa"])


#google_1r
google_1r <- OneR(above1M~., data=train)
pred_1r <- predict(google_1r, val)
cm_1r <- confusionMatrix(pred_1r, val$above1M, positive = "Blockbuster")

acc <- cbind(acc, OneR = cm_1r$overall["Accuracy"])
fp <- cbind(fp, OneR = 1 - cm_1r$byClass["Specificity"])
fn <- cbind(fn, OneR = 1 - cm_1r$byClass["Sensitivity"])
k <- cbind(k, OneR = cm_1r$overall["Kappa"])


#google_jr
google_jr <- JRip(above1M~., data=train)
google_jr
pred_jr <- predict(google_jr, val)
cm_jr <- confusionMatrix(pred_jr, val$above1M, positive = "Blockbuster")
cm_jr

acc <- cbind(acc, JRip = cm_jr$overall["Accuracy"])
fp <- cbind(fp, JRip = 1 - cm_jr$byClass["Specificity"])
fn <- cbind(fn, JRip = 1 - cm_jr$byClass["Sensitivity"])
k <- cbind(k, JRip = cm_jr$overall["Kappa"])

#ksvm
google_ksvm <- ksvm(above1M~., data=train, type = "C-svc", kernel = "vanilladot")
pred_ksvm <- predict(google_ksvm, val)
cm_ksvm <- confusionMatrix(pred_ksvm, val$above1M, positive = "Blockbuster")

acc <- cbind(acc, KSVM = cm_ksvm$overall["Accuracy"])
fp <- cbind(fp, KSVM = 1 - cm_ksvm$byClass["Specificity"])
fn <- cbind(fn, KSVM = 1 - cm_ksvm$byClass["Sensitivity"])
k <- cbind(k, KSVM = cm_ksvm$overall["Kappa"])

#svm
google_svm <- svm(above1M~., data=train, method = "C-classificaion", kernel = "linear")
pred_svm <- predict(google_svm, val)
cm_svm <- confusionMatrix(pred_svm, val$above1M, positive = "Blockbuster")

acc <- cbind(acc, SVM = cm_svm$overall["Accuracy"])
fp <- cbind(fp, SVM = 1 - cm_svm$byClass["Specificity"])
fn <- cbind(fn, SVM = 1 - cm_svm$byClass["Sensitivity"])
k <- cbind(k, SVM = cm_svm$overall["Kappa"])

#####
# renaming row names for FPR and FNR
rownames(fp) <- c("False Positive Rate")
rownames(fn) <- c("False Negative Rate")

# combining results into a table
ans <- rbind(acc, fp, fn, k)
ans

#--Ensemble Model Test --

library(caret)
library(caretEnsemble)
library(mlbench)

models <- caretList(train[,-8], train[,8], methodList=c("rpart", "M5"))
ens <- caretEnsemble(models)
pred_ens <- predict(ens, val)
confusionMatrix(pred_ens, val$above1M, positive = "Blockbuster")

# this did not work for some reason. Github was posting an error.

