setwd("/Users/leonardo/Desktop/SM4HDD")
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

if (!require(lattice)) {
  install.packages("lattice") 
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

if (!requireNamespace("sparseSVM", quietly = TRUE)) {
  install.packages("sparseSVM")
}

if (!require(genlasso)) {
  install.packages("genlasso")
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
library(sparseSVM)
library(genlasso)

# 1) LOADING DATA
data <- read.csv("Data/Swarm_Behaviour.csv")
# Clearing the data: in the train set there are some NAs
data <- na.omit(data)
dim(data)

# change some data types
sapply(data, class)
data <- as.data.frame(apply(data, 2, as.numeric))  # Convert all variable types to numeric
sapply(data, class)                                # Print classes of all colums

# 2.1) Balanced classes?
table(data$Swarm_Behaviour)

# Setting seed for reproducibility
set.seed(123)

# Rebalance the classes deleting from class 0 until the classes are balanced
indices_class_0 <- which(data$Swarm_Behaviour == 0)
sampled_indices <- sample(indices_class_0, 7954)
new_class_0 <- data[sampled_indices, ]

indices_class_1 <- which(data$Swarm_Behaviour == 1)
sampled_indices1 <- sample(indices_class_1, 7954)
new_class_1 <- data[sampled_indices1, ]

data <- rbind(new_class_1, new_class_0)

# We can see now that the classes are balanced
table(data$Swarm_Behaviour)

# 2.2) Train test split
# Create an index for splitting the data
index <- createDataPartition(data$Swarm_Behaviour, p = 0.85, list = FALSE)

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

# Set the desired number of observations for the subsets (extract a subset of train for the cv)
desired_size_train <- 1000  

# Randomly sample the specified number of observations for training set
sample_indices_train <- sample(1:nrow(X_train), size = desired_size_train, replace = FALSE)
X_train_cv <- X_train[sample_indices_train, ]
y_train_cv <- y_train[sample_indices_train]

# Print the dimensions of the resulting subsets
cat("Dimensions of X_train_cv:", dim(X_train_cv), "\n")
cat("Dimensions of y_train_cv:", length(y_train_cv), "\n")

# Count of 1s and 0s in the training set
table_y_train <- table(y_train)
cat("Training Set - Count of 1s:", table_y_train[["1"]], "\n")
cat("Training Set - Count of 0s:", table_y_train[["0"]], "\n")

# Count of 1s and 0s in the cross-validation training set
table_y_train_cv <- table(y_train_cv)
cat("CV Training Set - Count of 1s:", table_y_train_cv[["1"]], "\n")
cat("CV Training Set - Count of 0s:", table_y_train_cv[["0"]], "\n")

# 3) Correlations 
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

## 4) Lasso Regularization - Linear Model
# Rename the datasets
X_cv <- X_train_cv
y_cv <- y_train_cv

# Adjust type
X_cv <- as.matrix(X_cv)
y_cv <- as.vector(y_cv)

model <- glmnet(x = X_cv, y = y_cv)
cv_results <- cv.glmnet(X_cv, y_cv, type.measure = "mse", nfolds = 10)

# Plot the graph with lambdas
plot(cv_results)
plot(model, xvar = "lambda")

# Extract the best lambda
best_lambda <- cv_results$lambda.min

# print the best lambda
cat("The lambda min is: ", best_lambda, "\n")

# Fit the final model with the best lambda
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)

# Extract coefficients for selected variables
coefficients <- coef(lasso_model, s = best_lambda)

# Count and print selected variables
selected_variables <- sum(coefficients != 0)
print(paste("Number of selected variables:", selected_variables))

# Make predictions (replace X_test with your test set)
predictions <- predict(lasso_model, newx = as.matrix(X_test), s = best_lambda)

# Mean Squared Error
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)

conf_matrix <- table(binary_predictions, y_test)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

# Precision
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
print(paste("Precision:", precision))

# Recall
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Recall:", recall))

# Adding headers to the confusion matrix
colnames(conf_matrix) <- c("Actual 0", "Actual 1")
rownames(conf_matrix) <- c("Predicted 0", "Predicted 1")

# Print the modified confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Get the names and coefficients of selected variables
nonzero_rows <- as.matrix(coefficients[apply(coefficients != 0, 1, any), , drop = FALSE])
variable_name <- names(which(nonzero_rows[, 1] != 0))
print(variable_name)
variable_list <- as.list(variable_name)
variable_list <- variable_list[variable_list != "(Intercept)"]

# Deleting the number at the end of the strings 
for (i in seq_along(variable_list)) {
  # Extract the current variable name
  current_variable <- variable_list[[i]]
  
  # Remove numbers at the end of the string
  cleaned_variable <- gsub("\\d+$", "", current_variable)
  
  # Update the variable name in the list
  variable_list[[i]] <- cleaned_variable
}

# Assuming variable_list is a list of variable names
# Convert the list to a vector
variable_vector <- unlist(variable_list)

# Create a frequency table
variable_frequency <- table(variable_vector)

# Define the desired order for the x-axis
desired_order <- c("x", "y", "xVel", "yVel", "xA", "yA", "xS", "yS", "xC", "yC", "nAC", "nS")

# Plot the frequency histogram with specific order on the x-axis
barplot(variable_frequency[desired_order], main = "Variable Frequency Histogram", xlab = "Variable Names", ylab = "Frequency", names.arg = desired_order)

## 5) Ridge Regression - Linear Model
# Rename the datasets
X_cv <- X_train_cv
y_cv <- y_train_cv

# Adjust type
X_cv <- as.matrix(X_cv)
y_cv <- as.vector(y_cv)

# Fit Ridge Regression Model
library(glmnet)
model <- glmnet(x = X_cv, y = y_cv, alpha = 0)  # Set alpha to 0 for ridge regression

# Cross-Validation
cv_results <- cv.glmnet(X_cv, y_cv, type.measure = "mse", nfolds = 10)

# Plot Cross-Validation Results
plot(cv_results)

# Plot Coefficient Paths
plot(model, xvar = "lambda")

# Extract the optimal lambda value
optimal_lambda <- cv_results$lambda.min
cat("Optimal Lambda:", optimal_lambda, "\n")

# Refit the model with the optimal lambda
final_model <- glmnet(x = X_cv, y = y_cv, alpha = 0, lambda = optimal_lambda)

# Extract the best lambda
best_lambda <- cv_results$lambda.min

# Print the best lambda
cat("The lambda min is:", best_lambda, "\n")

# Fit the final model with ridge regression and the best lambda
ridge_model <- glmnet(x = X_train, y = y_train, alpha = 0, lambda = best_lambda)

# Extract coefficients for selected variables
coefficients <- coef(ridge_model, s = best_lambda)

# Make predictions (replace X_test with your test set)
predictions <- predict(ridge_model, newx = as.matrix(X_test), s = best_lambda)

# Mean Squared Error
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)

# Calculate confusion matrix
conf_matrix <- table(binary_predictions, y_test)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

# Precision
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
print(paste("Precision:", precision))

# Recall
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Recall:", recall))

# 6) Elastic Net

# Convert data to matrix
X_train_cv <- as.matrix(X_train_cv)
X_test <- as.matrix(X_test)
y_train_cv <- as.vector(y_train_cv)
y_test <- as.vector(y_test)
X_train <- as.matrix(X_train)
y_train <- as.vector(y_train)

# Define alpha and lambda sequences for grid search
alphas <- seq(0, 1, by = 0.1)  # Set your alpha sequence
lambdas <- seq(0.001, 1, by = 0.1)  # Set your lambda sequence
# Generate a grid of all combinations of alpha and lambda
param_grid <- expand.grid(alpha = alphas, lambda = lambdas)
# Train using grid search and save the MSE values
mse_values <- matrix(NA, nrow = length(alphas), ncol = length(lambdas))

custom <- trainControl(method = "repeatedcv", number = 10, repeats = 5, verboseIter = TRUE)
set.seed(1234)
train_data_cv <- cbind.data.frame(y_train_cv, X_train_cv)
for (i in seq_len(nrow(param_grid))) {
  current_alpha <- param_grid$alpha[i]
  current_lambda <- param_grid$lambda[i]
  
  model <- train(
    y_train_cv ~ .,
    data = train_data_cv,
    method = 'glmnet',
    tuneGrid = data.frame(alpha = current_alpha, lambda = current_lambda),
    trControl = custom
  )
  
  mse_values[which(alphas == current_alpha), which(lambdas == current_lambda)] <- min(model$results$RMSE)
}

# Convertinf MSE values to a data frame for plotting
mse_data <- expand.grid(alpha = alphas, lambda = lambdas)
mse_data$MSE <- as.vector(mse_values)

# Plotting
ggplot(mse_data, aes(x = alpha, y = lambda, fill = MSE)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "MSE values across Alphas/Lambdas", x = "Alphas", y = "Lambdas") +
  theme_minimal()

# Find the best alpha and lambda based on minimum MSE
best_index <- which(mse_data$MSE == min(mse_data$MSE))
best_alpha <- mse_data$alpha[best_index]
best_lambda <- mse_data$lambda[best_index]

# print best alpha and best lambda
cat("Best Alpha:", best_alpha, "\n")
cat("Best Lambda:", best_lambda, "\n")

# Set up the model with the best alpha and lambda
final_model <- glmnet(x = X_train, y = y_train, alpha = best_alpha, lambda = best_lambda)

# Make predictions
predictions <- predict(final_model, newx = X_test, s = best_lambda)

# Mean Squared Error
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", mse))

# Convert predictions to binary
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

## 7) Lasso regularization - Sparse SVM
# Rename the datasets
X_cv <- X_train_cv
y_cv <- y_train_cv

# Adjust type
X_cv <- as.matrix(X_cv)
y_cv <- as.vector(y_cv)

lasso.svm <- sparseSVM(X_cv, y_cv)
plot(lasso.svm, xvar="norm")
plot(lasso.svm, xvar="lambda")

# Fit Lasso Sparse SVM Model with Cross-Validation
cv.svm <- cv.sparseSVM(X_cv, y_cv)

# Extract the best lamda
plot(cv.svm)
cat("Lambda min:", cv.svm$lambda.min, "\n")

# Refit the model with the optimal cost parameter
final_svm_model <- sparseSVM(as.matrix(X_train),as.vector(y_train),lambda = cv.svm$lambda.min)

# Predict on the test data
svm_predictions <- as.vector(predict(final_svm_model, X = as.matrix(X_test)))

# Mean Squared Error
svm_mse <- mean((svm_predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", svm_mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
svm_binary_predictions <- ifelse(svm_predictions > threshold, 1, 0)

# Calculate confusion matrix
svm_conf_matrix <- table(svm_binary_predictions, y_test)
print("Confusion Matrix:")
print(svm_conf_matrix)

# Calculate accuracy
svm_accuracy <- sum(diag(svm_conf_matrix)) / sum(svm_conf_matrix)
print(paste("Accuracy:", svm_accuracy))

# Precision
svm_precision <- svm_conf_matrix[2, 2] / sum(svm_conf_matrix[, 2])
print(paste("Precision:", svm_precision))

# Recall
svm_recall <- svm_conf_matrix[2, 2] / sum(svm_conf_matrix[2, ])
print(paste("Recall:", svm_recall))

# 8) Generalized Lasso
# adjust type of X_train, ...
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.vector(y_train)
y_test <- as.vector(y_train)
X_train_cv <- as.matrix(X_train_cv)
y_train_cv <- as.vector(y_train_cv)

# define the matrix of weights
D <- matrix(0, nrow = 2400-12, ncol = 2400)
for (i in 1:(2400-12)) {
  idx <- seq(i, 2400, by = 12)
  D[i, idx] <- 1
}

# defining a first model in order to retain the lambdas used in the build in cross validation
model_gen<-genlasso(y_train_cv, X_train_cv, D)
summary(model_gen)

# plot the model
plot(model_gen)

# Get the lambda values used during model training (we are going to use them for our cross validation by hand)
lambda_values <- model_gen$lambda
plot(lambda_values)

# Perform a cross validation in order to find the best lambda
set.seed(123)  # Set seed for reproducibility
n.train <- nrow(X_train_cv)  # Number of training points
n.folds <- 5  # Number of CV folds
foldid <- sample(rep_len(seq.int(n.folds), n.train))  # Fold number for each training point

# Initialize variables to store results
best_lambda <- NULL
best_mean_cv_loss <- Inf

# Evaluate each lambda on each fold:
for (lambda_index in seq_along(lambda_values)) {
  mean_cv_loss <- 0
  
  # Iterate over folds for cross-validation
  for (fold in 1:n.folds) {
    fold_indices <- which(foldid == fold)
    
    # Fit the model on the training fold with the current lambda
    fold_genlasso_fit <- genlasso(y = y_train_cv[-fold_indices],
                                  X = X_train_cv[-fold_indices, ],
                                  D = D)
    
    # Predict on the validation fold
    fold_genlasso_preds <- predict(fold_genlasso_fit,
                                   lambda = lambda_values[lambda_index],
                                   Xnew = X_train_cv[fold_indices, ])$fit
    
    # Calculate mean squared error for this fold
    fold_loss <- mean((fold_genlasso_preds - y_train_cv[fold_indices])^2)
    mean_cv_loss <- mean_cv_loss + fold_loss
  }
  
  mean_cv_loss <- mean_cv_loss / n.folds  # Calculate mean cross-validation loss for this lambda
  
  
  # Check if this lambda has the smallest cross-validation loss
  if (mean_cv_loss < best_mean_cv_loss) {
    best_mean_cv_loss <- mean_cv_loss
    best_lambda <- lambda_values[lambda_index]
  }
}

# Print the best lambda and its mean cross-validation loss
print(paste("Best Lambda:", best_lambda))
print(paste("Mean Cross-Validation Loss:", best_mean_cv_loss))

# Now we retrain the best model 
model_gen_final <- genlasso(y_train, X_train, D, lambda = best_lambda)

# plot the model
plot(model_gen_final)

# Predict on the test data
genlasso_predictions <- as.vector(predict(model_gen_final, newdata = as.matrix(X_test)))

# Mean Squared Error
genlasso_mse <- mean((genlasso_predictions - y_test)^2)
print(paste("Mean Squared Error on Test Set:", genlasso_mse))

# Convert predictions to binary (e.g., using a threshold)
threshold <- 0.5
genlasso_binary_predictions <- ifelse(genlasso_predictions > threshold, 1, 0)

# Calculate confusion matrix
genlasso_conf_matrix <- table(genlasso_binary_predictions, y_test)
print("Confusion Matrix:")
print(genlasso_conf_matrix)

# Calculate accuracy
genlasso_accuracy <- sum(diag(genlasso_conf_matrix)) / sum(genlasso_conf_matrix)
print(paste("Accuracy:", genlasso_accuracy))

# Precision
genlasso_precision <- genlasso_conf_matrix[2, 2] / sum(genlasso_conf_matrix[, 2])
print(paste("Precision:", genlasso_precision))

# Recall
genlasso_recall <- genlasso_conf_matrix[2, 2] / sum(genlasso_conf_matrix[2, ])
print(paste("Recall:", genlasso_recall))
