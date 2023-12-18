setwd("/Users/leonardo/Desktop/swarm")
rm(list=ls())

# 1) PACKAGES INSTALLATION AND LIBRARIES LOADING 

# Install required package
if (!require(dplyr)) {
  install.packages("dplyr")
}

if (!require(stringr)) {
  install.packages("stringr")
}

if (!require(ggplot2)) {
  install.packages("ggplot2")
}

if (!require(maps)) {
  install.packages("maps")
}

if (!require(dplyr)) {
  install.packages("dplyr")
}

if (!require(viridis)) {
  install.packages("viridis")
}

if (!require(caret)) {
  install.packages("caret") 
}

if (!require(bestglm)) {
  install.packages("bestglm")
}

if (!require(glmnet)) {
  install.packages("glmnet")
}

if (!require(GGally)) {
  install.packages("GGally")
}

if (!require(gglasso)) {
  install.packages("gglasso")
}

# Load required libraries
library(ggplot2)
library(maps)
library(dplyr)
library(viridis)
library(stringr)
library(caret)
library(lattice)
library(bestglm)
library(glmnet)
library(Matrix)
library(GGally)
library(gglasso)

# 2) DATASET LOADING AND EXPLORATION

# Loading the dataset
data <- read.csv("swarm_behaviour.csv")

# Clearing the data: in the train set there are some NAs 
data <- na.omit(data)

# Summary of the train set
summary(data)

# 2.1) Balance of classes
# Balanced classes?
table(data$Swarm_Behaviour)

# Setting seed for reproducibility
set.seed(123)  

# Rebalance the classes deleting from class 0 until the classes are balanced
indices_class_0 <- which(data$Swarm_Behaviour == 0)
sampled_indices <- sample(indices_class_0, 7954)
new_class_0 <- data[sampled_indices, ]
data_class_1 <- data[data$Swarm_Behaviour == 1, ]
data <- rbind(data_class_1, new_class_0)

# We can see now that the classes are balanced
table(data$Swarm_Behaviour)

# 2.2) Train test split
# Create an index for splitting the data
index <- createDataPartition(data$Swarm_Behaviour, p = 0.1, list = FALSE)

# Create training and testing sets
train <- data[index, ]
test <- data[-index, ]

# Define X_train and y_train
X_train <- train[, !names(train) %in% "Swarm_Behaviour"]  # Exclude the column "Swarm_Behaviour"
y_train <- train$Swarm_Behaviour

# Define X_test and y_test
X_test <- test[, !names(test) %in% "Swarm_Behaviour"]
y_test <- test$Swarm_Behaviour

# Print the dimensions of the resulting sets
cat("Dimensions of Training Set:", dim(train), "\n")
cat("Dimensions of Testing Set:", dim(test), "\n")

# 3) CORRELATIONS
correlations <- cor(train)

# Let's see the correlations between each feature and the class
# Extract the last column (correlation with the variable 'Swarm_behaviour')
cor_with_class <- correlations[, ncol(correlations)]

# Order the correlation values
ordered_cor <- sort(cor_with_class)

# Exclude the last element
ordered_cor <- ordered_cor[-length(ordered_cor)]

# Plot the ordered correlation values
barplot(ordered_cor, main = "Correlation with 'class'", xlab = "Features", ylab = "Correlation")

# Print the tail
tail(ordered_cor)

# How correlated are to each other the top 10 most correlated (wrt Swarm_behavour) variables? 
# Get the names of the top 10 most correlated variables
top_10_vars <- names(tail(ordered_cor, 10))

# Extract the relevant columns from the training set
top_10_correlation_data <- train[, c(top_10_vars, "Swarm_Behaviour")]

# Calculate the correlation matrix for the top 10 variables
top_10_correlations <- cor(top_10_correlation_data)

# Print the correlation matrix
print(top_10_correlations)
# We can see that the correlations are between 0.60 and 0.75, so they are correlated

ggplot(train, aes(x = yVel172 , y = yVel69 , color = as.factor(Swarm_Behaviour))) +
  geom_point() +
  labs(title = "yVel172 vs yVel69",
       x = "yVel172",
       y = "yVel69",
       color = "Swarm_Behaviour")

# Create a pair plot using ggpairs
ggpairs(top_10_correlation_data)
# It does not seem good though

# 4) LASSO
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.vector(y_train)
y_test <- as.vector(y_test)

# Lasso 
model <- glmnet(x = X_train, y = y_train)
lasso_cv <- cv.glmnet(x = X_train, y = y_train, alpha = 1)

# Plot lasso model
plot(lasso_cv)

# Choose the best lambda value based on cross-validation results
best_lambda <- lasso_cv$lambda.min
se_lambda <- lasso_cv$lambda.1se
selected_lambda <- se_lambda # CHANGE IT TO CHANGE ALL THE FOLLOWING LAMBDAs 

plot(model, xvar = "lambda")
abline(v = log(best_lambda), col = "red", lty = 2)
abline(v = log(se_lambda), col = "blue", lty = 2)

# Fit the final Lasso model with the selected lambda
final_model <- glmnet(x = X_train, y = y_train, alpha = 1, lambda = selected_lambda)

# Extract coefficients for selected variables
coefficients <- coef(final_model, s = selected_lambda)

# Count and print selected variables
selected_variables <- sum(coefficients != 0)
print(paste("Number of selected variables:", selected_variables))

# Get the names and coefficients of selected variables
nonzero_rows <- coefficients[rowSums(coefficients != 0, na.rm = TRUE) > 0, ]
print(nonzero_rows)

# Make predictions (replace X_test with your test set)
predictions <- predict(final_model, newx = X_test, s = selected_lambda)

# Mean Squared Error
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)

# Confusion Matrix
conf_matrix <- table(binary_predictions, y_test)
print("Confusion Matrix:")
print(conf_matrix)

# Precision
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
print(paste("Precision:", precision))

# Recall
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Recall:", recall))

### RIDGE ###

# Ridge
model <- glmnet(x = X_train, y = y_train, alpha = 0)  # Use alpha = 0 for Ridge regression
ridge_cv <- cv.glmnet(x = X_train, y = y_train, alpha = 0)

# Plot Ridge model
plot(ridge_cv)

# Choose the best lambda value based on cross-validation results
best_lambda <- ridge_cv$lambda.min
se_lambda <- ridge_cv$lambda.1se
selected_lambda <- se_lambda  # CHANGE IT TO CHANGE ALL THE FOLLOWING LAMBDAs 

plot(model, xvar = "lambda")
abline(v = log(best_lambda), col = "red", lty = 2)
abline(v = log(se_lambda), col = "blue", lty = 2)

# Fit the final Ridge model with the selected lambda
final_model <- glmnet(x = X_train, y = y_train, alpha = 0, lambda = selected_lambda)  # Use alpha = 0 for Ridge regression

# Extract coefficients for selected variables
coefficients <- coef(final_model, s = selected_lambda)

# Count and print selected variables
selected_variables <- sum(coefficients != 0)
print(paste("Number of selected variables:", selected_variables))

# Get the names and coefficients of selected variables
nonzero_rows <- coefficients[rowSums(coefficients != 0, na.rm = TRUE) > 0, ]
print(nonzero_rows)

# Make predictions (replace X_test with your test set)
predictions <- predict(final_model, newx = X_test, s = selected_lambda)

# Mean Squared Error
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)

# Confusion Matrix
conf_matrix <- table(binary_predictions, y_test)
print("Confusion Matrix:")
print(conf_matrix)

# Precision
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
print(paste("Precision:", precision))

# Recall
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Recall:", recall))

### GROUP LASSO ### 
library(survival)
library(glmnet)
#install.packages("ncvreg")
library(ncvreg)
library(gglasso)
#install.packages("zoo")
library(zoo) 

# Create grouping information
# grouping feature relative to the same boid 
group_indicator <- rep(1:ceiling(ncol(X_train)/12), each = 12, length.out = ncol(X_train))

#grouping feature of the same type 
group_indicator <- rep(1:12, length.out = ncol(X_train)) %% 12 
group_indicator <- ifelse(group_indicator == 0, 12, group_indicator)

tail(group_indicator, 13)
head(group_indicator, 13)

fit=gglasso(x=X_train, y=y_train, group=group_indicator, loss='ls')

coef.mat=fit$beta
#
plot(fit)
#
# Group1 enters the equation - first model where group 1 is present
# (I look at variable 1 but it was the same if I was looking at variable 2 or 3, 
# since the three variables enter in the model together)
g1=max(which(coef.mat[1,]==0))
#
#Group2 enters the equation - first model where group 2 is present
g2=max(which(coef.mat[4,]==0))
#
#Group3 enters the equation - first model where group 3 is present
g3=max(which(coef.mat[8,]==0))
#
#Group4 enters the equation - first model where group 4 is present
g4=max(which(coef.mat[12,]==0))
#
#Coefficient Plot
#
plot(fit$b0,main="Coefficient vs Step",
     ylab="Intercept",xlab="Step (decreasing Lambda =>)",
     xlim=c(-1,100),
     ylim=c(5,max(fit$b0)),
     type="l",lwd=4)
grid()
#
x=c(g1,g2,g3,g4)
y=c(fit$b0[g1],fit$b0[g2],fit$b0[g3],fit$b0[g4])
#
points(x=x, y=y, pch=13, lwd=2, cex=2)
#
lmda=round(fit$lambda[c(g1,g2,g3,g4)],2)
text(x=x-0.5, y=y+0.1, labels=c("Group1", "Group2",
                                "Group3", "Group4"), pos=3, cex=0.9)
text(x=x-0.5, y=y-0.1, labels=paste("Lambda\n=",lmda), pos=1, cex=0.8)
#
#The intercept is not penalized and hence is always present 
#in the regression equation. But as the plot above shows, 
#each group enters the regression equation at a particular 
#value of lambda.


