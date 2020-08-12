#--Import Data--
cont_colors = read.csv('data/continent_colors.csv')
country_cc = read.csv('data/country_cc.csv')
country_codes = read.csv('data/country_codes.csv')
country_info = read.csv('data/country_info.csv')

Titanic = read.csv('data/TitanicforKNN.csv')

#--Import Libraries--
install.packages("tidyverse")
install.packages('dplyr')
library(dplyr)

library(class)
library(gmodels)
library(caret)

############
#---Q1
# A. Use join to add two columns: the hex color code of a country (country_cc) and the hex color code of continent (cont_cc) that a country belongs.
class(country_info)
class(country_codes)
class(country_cc)
class(cont_colors)

country_join = inner_join(country_info, country_cc, by='country')
country_join

cont_country_join = inner_join(country_join, cont_colors, by='continent')
cont_country_join

#--Rename columns for country and continent colors--
colors_join = rename(cont_country_join, cont_color = cc.y,
       country_color = cc.x)
head(colors_join)

#Then use Unique( data_name[, columns]) to create a new table contains the following information only: country, continent, country_color, cont_color

colors_df = unique(colors_join[,c('country', 'continent', 'country_color', 'cont_color')])
colors_df

#Then use head() and copy the first 6 entries of that table.
head(colors_df)


# B. Focus on year=2007 and add two extra columns – the first one showing population rank (largest = 1, 2nd largest =2, etc.) and the second on showing the gdpPercap rank (largest =1 , 2nd largest =2).

#--filter out any year that is not 2007
world_2007 = filter(colors_join, year==2007)
colnames(world_2007)
head(world_2007)

#--order data frame by population from smallest to largest--
pop_rank_2007 = world_2007[order(world_2007$pop),]
head(pop_rank_2007)

#--add population rank column--
pop_rank_2007$pop_rank = 1:nrow(pop_rank_2007)
pop_rank_2007

#--order by gdpPercap--
gdp_rank = pop_rank_2007[order(pop_rank_2007$gdpPercap),]

#--add gdpPercap rank column--
gdp_rank$gdpPercap_rank = 1:nrow(gdp_rank)

#1. Compute the number of countries, the average GDP, average population ranking, and average gdpPercap ranking for each continent.

#--Number of countries per continent--
countries_per_continent = summarise(group_by(gdp_rank, continent), count=n())
countries_per_continent

#--Average GDP per continent---
avg_gdp_per_continent = summarise(group_by(gdp_rank, continent), avg_gdp_per_cont =mean(gdpPercap, na.rm = TRUE))
avg_gdp_per_continent

#--Average population ranking per continent--
avg_pop_rank_per_continent = summarise(group_by(gdp_rank, continent), avg_pop_rank=mean(pop_rank, na.rm=TRUE))
avg_pop_rank_per_continent

#--Average gdpPercap ranking per continent--
avg_gdp_rank_per_continent = summarise(group_by(gdp_rank, continent), avg_gdp_rank=mean(gdpPercap_rank, na.rm=TRUE))
avg_gdp_rank_per_continent

#2. Remove a continent with fewer than 10 countries and arrange the table by the descending order of average GDP and paste the table here.

#--Remove continent that has fewer than 10 countries--
cont_10_plus_countries = gdp_rank[gdp_rank$continent != 'Oceania',]
cont_10_plus_countries


#--Arrange table by descending order of average GDP--Use Pipe to chain the output of one function and put it into another--
avg_gdp_per_cont = summarise(group_by(cont_10_plus_countries, continent), avg_gdp_per_cont =mean(gdpPercap, na.rm = TRUE))
avg_gdp_per_cont

avg_gdp_per_cont = cont_10_plus_countries %>%
  arrange(desc(gdpPercap)) %>%
  slice(1:10)
avg_gdp_per_cont

############
#--Q2. Titanic Survivor using KNN.
#In this question, you will use the K-Nearest Neighbors (KNN) algorithm to predict whether a passenger will survive or not (Survived==1, Not survived==0). To begin your work on this question, first read the data from the file "TitanicforKNN.csv" to a data frame named Titanic. **Note: Please review the data before proceeding. You will notice that I already converted all the categorical variables (Gender, Fare, Class) into 0-1 columns. I did so, because KNN does not work well with non-numeric variables.** 

#Split the data into training data and test data with a split of 691-200 and seed 567. (**Remember to include set.seed(567) before sampling in your code, so we all end up making the same split.**). Closely examine the data format. In this context, should the dependent variable be represented as a factor or a numerical type (such as integer)? Make a correction if necessary.
summary(Titanic)

#--Split the data into train, test (691-200 split, seed=567)
set.seed(567)
train_rows = sample(891, 691)

#--Create character class for kNN y--
Titanic$Survived = factor(Titanic$Survived, levels=c(0,1),
                           labels = c('Deceased', 'Survived'))
Titanic

titanic_train = Titanic[train_rows,]    #--training set
titanic_valid = Titanic[-train_rows,]   #--validation set

nrow(titanic_train)
nrow(Titanic)
head(titanic_train)

#--Separate x and y variables-- (Survived==1, Not survived==0)
#--separate y
titanic_train_y <- titanic_train[, 1] #--change back to 1 if character doesn't work
titanic_valid_y <- titanic_valid[,1]

#--separate x
titanic_train_x <- titanic_train[, c(2:8)]
titanic_valid_x <- titanic_valid[, c(2:8)]


# (a) Run the KNN algorithm to predict the response variable Survived==1 for each passenger in the test data. Do this for K = 5, 10, 25, 50, 100. According to these predictions for K = 5, 10, 25, 50, 100, what is the proportion of passengers in the test data who will be classified as “survive”?

#--Example from class--
# knn_5_pred <- knn(train = titanic_train_x[,1:7], test = titanic_train_x[,1:7], 
#                   cl = titanic_train_y, k = 5, prob = FALSE) #--prob=FALSE makes it so Deceased/Survived comes up rather than a probability
knn_5_pred
# summary(knn_5_pred)
table(knn_5_pred)

#--Function to run KNN values--
knn_alg = function(num) {
  result=knn(train = titanic_train_x[,1:7], test = titanic_train_x[,1:7], 
      cl = titanic_train_y, k = num, prob = FALSE)
  return(result)
}

#--For loop that gets Survived proportion result for each of the K values--
lst = c(5, 10, 25, 50, 100)
for (elem in lst) {
  cat('    Survival for:', '\n', 
      'K =', elem, ': ', 
      round(summary(knn_alg(elem))[2]/691, 4), '\n\n')  #--rounds to 4 decimals the summary of the survived/total num passengers for each k
}


# (b) For each K, create a confusion matrix using confusionMatrix in caret package. Make a table and report the following statistics for each choice of K.

knn_5_pred <- knn(train = titanic_train_x[,1:7], test = titanic_valid_x[,1:7],
                  cl = titanic_train_y, k = 5, prob = TRUE)
length(knn_5_pred)
length(titanic_valid_y)

knn_10_pred <- knn(train = titanic_train_x[,1:7], test = titanic_valid_x[,1:7],
                  cl = titanic_train_y, k = 10, prob = TRUE)

knn_25_pred <- knn(train = titanic_train_x[,1:7], test = titanic_valid_x[,1:7],
                  cl = titanic_train_y, k = 25, prob = TRUE)

knn_50_pred <- knn(train = titanic_train_x[,1:7], test = titanic_valid_x[,1:7],
                  cl = titanic_train_y, k = 50, prob = TRUE)

knn_100_pred <- knn(train = titanic_train_x[,1:7], test = titanic_valid_x[,1:7],
                  cl = titanic_train_y, k = 100, prob = TRUE)


#--Confusion matrix for each k--
confusion5 = confusionMatrix(knn_5_pred, titanic_valid_y, positive = "Survived")
confusion10 = confusionMatrix(knn_10_pred, titanic_valid_y, positive = "Survived")
confusion25 = confusionMatrix(knn_25_pred, titanic_valid_y, positive = "Survived")
confusion50 = confusionMatrix(knn_50_pred, titanic_valid_y, positive = "Survived")
confusion100 = confusionMatrix(knn_100_pred, titanic_valid_y, positive = "Survived")

#--Create Data frames for each of the results and add the prob--
table5 = data.frame(knn_5_pred)
table5 = data.frame(table5, attr(knn_5_pred, "prob"))

table10 = data.frame(knn_10_pred)
table10 = data.frame(table10, attr(knn_10_pred, "prob"))

table25 = data.frame(knn_25_pred)
table25 <- data.frame(table25, attr(knn_25_pred, "prob"))

table50 = data.frame(knn_50_pred)
table50 = data.frame(table50, attr(knn_50_pred, "prob"))

table100 = data.frame(knn_100_pred)
table100 = data.frame(table100, attr(knn_100_pred, "prob"))

#--combine all accuracy-related scores into 1 matrix
stats_matrix = do.call("rbind", list(confusion5$overall, confusion10$overall,
                                    confusion25$overall, confusion50$overall,
                                    confusion100$overall))
stats_matrix

#--convert matrix to data frame
stats_table = as.data.frame(stats_matrix)
class(stats_table)

#--add naming column
stats_table$k_value = c('k=5', 'k=10', 'k=25', 'k=50', 'k=100')
stats_table = stats_table[c(8,1,2,7)] #--reordering columns
colnames(stats_table)
stats_table
                    
library(data.table)

#--caluculate and create FP and FN dataframe--
c5 = c(1- confusion5$byClass[1], 1- confusion5$byClass[2]) 
c10 = c(1- confusion10$byClass[1], 1- confusion10$byClass[2]) 
c25 = c(1- confusion100$byClass[1], 1- confusion25$byClass[2]) 
c50 = c(1- confusion50$byClass[1], 1- confusion50$byClass[2]) 
c100 = c(1- confusion100$byClass[1], 1- confusion100$byClass[2]) 

#--turn FP and FN scores into a data frame--
df = rbind(c5,c10,c25,c50,c100)
df = as.data.frame(df)
class(df)
df = round(df, 4)
df

#--merge two data frames--
total = cbind(stats_table, df)
total

total = rename(total, c("Sensitivity"='False Neg', "Specificity"="False Pos"))
total

#--remove row names that came from merging data frames--
rownames(total) = c()
total


# i) Accuracy
# ii) False positive error rate --> means you get a positive result from a test, but you should have gotten a negative result == 1-Specificity

# iii) False negative error rate --> A false negative is a test result that indicates a person does not have a disease or condition when the person actually does have it, or you get a negative when you should have gotten a positive == 1-Sensitivity

# iv) P-value [Acc > NIR] --> McNemar's Test P-Value
# v) Kappa statistics


