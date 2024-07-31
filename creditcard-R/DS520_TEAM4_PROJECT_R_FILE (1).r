# Install and load required libraries
install.packages(c("rpart", "rpart.plot", "caret", "ROSE", "gplots"))
library(rpart)
library(rpart.plot)
library(caret)
library(ROSE)
library(gplots)
install.packages(c("pROC", "ROCR"))
library(pROC)
library(ROCR)
# LOAD DATASET
creditcard_data <- read.csv("creditcard.csv")

# Explore Dataset
str(creditcard_data)

table(creditcard_data$Class)

# Converting non-numeric variables to factors
non_numeric_cols <- sapply(creditcard_data, function(x) !is.numeric(x))
creditcard_data[, non_numeric_cols] <- lapply(creditcard_data[, non_numeric_cols], as.factor) # nolint: line_length_linter.

# Scaling - Scale the Amount variable
creditcard_data$Amount <- scale(creditcard_data$Amount)

str(creditcard_data)

# Create a binary target variable 'Fraud' based on the Class column
creditcard_data$Fraud <- ifelse(creditcard_data$Class == 1, "Yes", "No")
creditcard_data$Fraud <- as.factor(creditcard_data$Fraud)

# Split data into training and testing sets
set.seed(123)
split_index <- createDataPartition(creditcard_data$Fraud, p = 0.98, list = FALSE) # nolint: line_length_linter.
train_data <- creditcard_data[split_index, ]
test_data <- creditcard_data[-split_index, ]
#BEFORE UNDERSAMPLING
table(train_data$Class)

# Undersample the majority class (non-fraud)
nrow_fraud <- nrow(train_data[train_data$Class == 1, ])
undersample_frac <- 0.1
undersample_size <- nrow_fraud / undersample_frac
undersample_data <- ovun.sample(Class ~ ., data = train_data, method = "under", N = undersample_size, seed = 250) # nolint: line_length_linter.
train_undersampled <- undersample_data$data
#AFTER UNDERSAMPLING
table(train_undersampled$Class)
ggplot(train_undersampled, aes(x = Class, fill = factor(Class))) +
  geom_bar() +
  labs(title = "Histogram of Train Undersampled- fraud(1) and non-fraud(0)",
       x = "Class",
       y = "Frequency") +
  theme_minimal()

cat("Number of samples in the training set:", nrow(train_undersampled), "\n")
cat("Number of samples in the test set:", nrow(test_data), "\n")
# Build decision tree model
tree_model <- rpart(Fraud ~ . - Class, data = train_undersampled, method = "class") # nolint: line_length_linter.
print(tree_model)

# Visualize the decision tree
prp(tree_model, type = 2, extra = 1)

# Evaluate the model on the test set
predictions <- predict(tree_model, newdata = test_data, type = "class")
conf_matrix <- table(predictions, test_data$Fraud)
print(conf_matrix)

# Calculate accuracy, precision, recall, and F1-score
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall) # nolint: line_length_linter.

print(paste("Accuracy :", accuracy))
print(paste("Precision :", precision))
print(paste("Recall :", recall))
print(paste("F1 Score :", f1_score))

# Cross-validation for tree pruning
cv_prune <- prune.rpart(tree_model, cp = 0.01)

# Evaluate the pruned model on the test set
predictions_pruned <- predict(cv_prune, newdata = test_data, type = "class")
conf_matrix_prune <- table(predictions_pruned, test_data$Fraud)
print(conf_matrix_prune)

# Calculate accuracy, precision, recall, and F1-score for the pruned model
accuracy_prune <- sum(diag(conf_matrix_prune)) / sum(conf_matrix_prune)
precision_prune <- conf_matrix_prune[2, 2] / sum(conf_matrix_prune[, 2])
recall_prune <- conf_matrix_prune[2, 2] / sum(conf_matrix_prune[2, ])
f1_score_prune <- 2 * (precision_prune * recall_prune) / (precision_prune + recall_prune) # nolint: line_length_linter.

print(paste("Accuracy (Pruned):", accuracy_prune))
print(paste("Precision (Pruned):", precision_prune))
print(paste("Recall (Pruned):", recall_prune))
print(paste("F1 Score (Pruned):", f1_score_prune))

plot(conf_matrix_prune, main = "Confusion Matrix (Pruned)", col = c("red", "purple")) # nolint: line_length_linter.


# Calculate ROC curve
roc_curve_prune <- roc(test_data$Fraud, as.numeric(predictions_pruned))

pdf("ROC_Curve_Pruned.pdf", width = 4, height = 4) 
# Plot ROC curve
plot(roc_curve_prune, main = "ROC Curve (Pruned)", col = "red", lwd = 2)
abline(a = 0, b = 1, col = "green", lty = 2)
legend("bottomright", legend = "ROC Curve", col = "red", lwd = 2)

auc_value_prune <- auc(roc_curve_prune)
cat("Area Under the Curve (AUC - Pruned):", auc_value_prune, "\n")
dev.off()
