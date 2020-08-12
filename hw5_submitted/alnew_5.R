#--Install Packages--
install.packages("C50")
install.packages("e1071")
install.packages("kernlab")
library("C50")

#--Import Data--
cancer = read.csv('cancer.csv')

head(cancer)

#--convert your dependent variable as a factor with two levels (“No” and “Yes”). 
cancer$Cancer = factor(cancer$Cancer, levels=c(0,1),
                       labels=c('No', 'Yes'))

nrow(cancer)
#--Split the data (rows 1-581 in train set, row 582-681 in validation set).
set.seed(567)
train_rows = sample(681, 581)
train = cancer[train_rows,]
val = cancer[-train_rows,]

train[,-1]


###################
#--Q1: Run Algorithms--
###################

##############
#--a.	MODEL 1 C5.0 tree with trials = 1 (single tree)
#Use C5.0 with trials=1 
cancer_finder1 = C5.0(train[,-1], train$Cancer, trials=1)

#Alternatively, the code below will do the same:
cancer_finder1 = C5.0(Cancer~., data=train, trials=1)

#See the result: summary and plot 
summary(cancer_finder1)
plot(cancer_finder1)


#--Report Accuracy, FP (where positive=Yes), FN--
# predict newdata set 
cancer_finder1_pred = predict(cancer_finder1, val[,-1])

# Create two confusion matrix table
# Write down Type 1 and Type 2 Error in your note
CrossTable(cancer_finder1_pred, val$Cancer)
confusionMatrix(cancer_finder1_pred, val$Cancer, positive = "Yes")

#################
#--b.	MODEL 2: C5.0 tree with trials = 5 (5 tree boosting) 
#  
library(C50)
cancer_c5 = C5.0(train[,-1], train$Cancer, trials=5) #Training 
cancer_c5 = C5.0(Cancer~., data=train, trials=5) # This works too. 
plot(cancer_c5,0)
plot(cancer_c5, 4)  #--this tree is simpler than the one above

summary(cancer_c5)
pred_c5 = predict(cancer_c5, newdata=val, type = "class")  
confusionMatrix(pred_c5, val$Cancer, positive = "Yes")


#################
#--c. MODEL 3 Naive Bayes
#
install.packages("e1071")
library(e1071)
cancer_nb = naiveBayes(Cancer~. , data=train)
summary(cancer_nb)

pred_nb = predict(cancer_nb, newdata=val[,-1]) #--do I need the [,-1] here?

confusionMatrix(pred_nb, val$Cancer, positive = "Yes")


#################
#--d. MODEL 4 OneR
#
#1R predictor 
install.packages("RWeka")
library(RWeka)
install.packages("caret")
library("caret")

cancer_1r = OneR(Cancer~., data=train)
pred_1r = predict(cancer_1r, val[,-1])

cancer_1r
summary(cancer_1r)

confusionMatrix(pred_1r, val$Cancer, positive = "Yes")


# > cancer_1r
# UShap:
#   < 3.5	-> Yes
# >= 3.5	-> No
# (541/581 instances correct)
# 
# > summary(cancer_1r)
# 
# === Summary ===
#   
#   Correctly Classified Instances         541               93.1153 %
# Incorrectly Classified Instances        40                6.8847 %
# Kappa statistic                          0.8476
# Mean absolute error                      0.0688
# Root mean squared error                  0.2624
# Relative absolute error                 14.9728 %
# Root relative squared error             54.7308 %
# Total Number of Instances              581     
# 
# === Confusion Matrix ===
#   
#   a   b   <-- classified as
# 180  28 |   a = No
# 12 361 |   b = Yes
# > 
#   > confusionMatrix(pred_1r, val$Cancer, positive = "Yes")
# Confusion Matrix and Statistics
# 
# Reference
# Prediction No Yes
# No  26   6
# Yes  4  64
# 
# Accuracy : 0.9            
# 95% CI : (0.8238, 0.951)
# No Information Rate : 0.7            
# P-Value [Acc > NIR] : 1.556e-06      
# 
# Kappa : 0.7664         
# Mcnemar's Test P-Value : 0.7518         
#                                          
#             Sensitivity : 0.9143         
#             Specificity : 0.8667         
#          Pos Pred Value : 0.9412         
#          Neg Pred Value : 0.8125         
#              Prevalence : 0.7000         
#          Detection Rate : 0.6400         
#    Detection Prevalence : 0.6800         
#       Balanced Accuracy : 0.8905         
#                                          
#        'Positive' Class : Yes   
# 
# #=== Summary ===
# 
# Correctly Classified Instances         541               93.1153 %
# Incorrectly Classified Instances        40                6.8847 %
# Kappa statistic                          0.8476
# Mean absolute error                      0.0688
# Root mean squared error                  0.2624
# Relative absolute error                 14.9728 %
# Root relative squared error             54.7308 %
# Total Number of Instances              581     
# 
# === Confusion Matrix ===
#   
#   a   b   <-- classified as
# 180  28 |   a = No
# 12 361 |   b = Yes'


#################
#--e. MODEL 5 JRip
#
#JRip Rule Generator 
cancer_jr = JRip(Cancer~., data=train)
cancer_jr

summary(cancer_jr)

pred_jr = predict(cancer_jr, val[,-1])
confusionMatrix(pred_jr, val$Cancer, positive = "Yes")

# > cancer_jr
# JRIP rules:
#   ===========
#   
#   (UShap >= 3) and (BNucl >= 4) => Cancer=No (176.0/6.0)
# (Thick >= 7) => Cancer=No (27.0/3.0)
# (USize >= 5) => Cancer=No (7.0/0.0)
# (BNucl >= 3) and (Thick >= 5) => Cancer=No (7.0/2.0)
# => Cancer=Yes (364.0/2.0)
# 
# Number of Rules : 5
# 
# > summary(cancer_jr)
# 
# === Summary ===
#   
#   Correctly Classified Instances         568               97.7625 %
# Incorrectly Classified Instances        13                2.2375 %
# Kappa statistic                          0.9518
# Mean absolute error                      0.0409
# Root mean squared error                  0.143 
# Relative absolute error                  8.8936 %
# Root relative squared error             29.8267 %
# Total Number of Instances              581     
# 
# === Confusion Matrix ===
#   
#   a   b   <-- classified as
# 206   2 |   a = No
# 11 362 |   b = Yes
# > 
#   > pred_jr = predict(cancer_jr, val)
# > confusionMatrix(pred_jr, val$Cancer, positive = "Yes")
# Confusion Matrix and Statistics
# 
# Reference
# Prediction No Yes
# No  28   3
# Yes  2  67
# 
# Accuracy : 0.95            
# 95% CI : (0.8872, 0.9836)
# No Information Rate : 0.7             
# P-Value [Acc > NIR] : 3.993e-10       
# 
# Kappa : 0.8821          
# Mcnemar's Test P-Value : 1               
# 
# Sensitivity : 0.9571          
# Specificity : 0.9333          
# Pos Pred Value : 0.9710          
# Neg Pred Value : 0.9032          
# Prevalence : 0.7000          
# Detection Rate : 0.6700          
# Detection Prevalence : 0.6900          
# Balanced Accuracy : 0.9452          
# 
# 'Positive' Class : Yes     



#################
#--f. MODEL 6 Svm (Support Vector Machine)
#
library(e1071)

# radial kernel 
cancer_svm = svm(Cancer~., data=train, method = "C-classificaion", kernel = "radial")
cancer_svm 
pred_svm = predict(cancer_svm, val[,-1])
summary(cancer_svm)

confusionMatrix(pred_svm, val$Cancer, positive = "Yes")


#################
#--g. MODEL 7 Ksvm
#
library(kernlab) 

# Types of kernels:
# rbfdot Radial Basis kernel "Gaussian"
# polydot Polynomial kernel
# vanilladot Linear kernel
# splinedot Spline kernel


# splinedot: Spline kernel
cancer_ksvm = ksvm(Cancer~., data=train, type="C-svc", kernel="splinedot")

pred_ksvm = predict(cancer_ksvm, val[,-1])
summary(cancer_ksvm)

confusionMatrix(pred_ksvm, val$Cancer, positive = "Yes")


###################
#--Q2: Create a table showing the prediction results of all 7 algorithms for the first 10 rows in the validation set. --
###################

# a. 
pred_c5single = cancer_finder1_pred[c(1:10)]
table(pred_c5single)

# b.
pred_c5boost = pred_c5[c(1:10)]
table(pred_c5boost)

# c.
pred_naive_b = pred_nb[c(1:10)]
table(pred_naive_b)

# d.
pred_oneR = pred_1r[c(1:10)]
table(pred_oneR)

# pred_oneR
# No Yes 
# 4   6 

# e.
pred_JRip = pred_jr[c(1:10)]
table(pred_JRip)

# pred_JRip
# No Yes 
# 5   5 


# f.
pred_SVM = pred_svm[c(1:10)]
table(pred_SVM)

# g.
pred_KSVM = pred_ksvm[c(1:10)]
table(pred_KSVM)

###################
# Q3


###################
# Q4 Run a logistic regression with Y= Cancer and X = USize. Show the summary result and write down the regression equation. 

# Logistic Regression -- CHANGE THIS TO CANCER DATA --
#  
head(train)
cancer_logit = glm(Cancer~ USize, data=train, family="binomial")
summary(cancer_logit) # display results

round(exp(coef(cancer_logit)),3) # exponentiated coefficients

pred_logitp = predict(cancer_logit, newdata=val, type="response") # predicted values


