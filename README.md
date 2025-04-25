# Fraud-Detection
This project explores fraud detection using various machine learning models. It aims to classify transactions as fraudulent or legitimate based on a preprocessed, anonymized dataset. Along the way, I built a model evaluation pipeline, tested different algorithms, and examined the value of feature reduction through Principal Component Analysis (PCA).

## Dataset Overview
Source: Kaggle Credit Card Fraud Dataset

Records: 568,630 transactions

Features: 30 (including anonymized PCA components + Amount, Class, ID)

Target variable: Class (0 for legitimate, 1 for fraud)

## Preprocessing Steps
Checked for missing values (none found).

Checked for class imbalance (none found).

Conducted assumption checks for logistic regression:

Linearity with logit using Box-Tidwell test.

Multicollinearity via VIF (Variance Inflation Factor).

Applied Ridge regularization for logistic regression (Model 3).

Scaled features as needed.

## Models Tried
Logistic Regression

Random Forest

Support Vector Machine

XGBoost

Neural Network

The Random Forest model performed the best in terms of accuracy and F1-score. Interestingly, it achieved nearly the same performance using only the top 13 PCA components, suggesting the possibility of reducing model complexity without sacrificing accuracy.

## Evaluation Metrics
All models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion matrix

## Key Findings
The Random Forest model was the most effective overall.

Feature reduction showed that the top 13 components were enough for strong performance.

Surprisingly, the Transaction Amount feature — the only raw variable — was among the least predictive.

The hyperparameter tuning pipeline developed for Random Forest proved too slow to run on Kaggle, and will be tested on a smaller dataset next.

## Future Work
Run the hyperparameter tuning pipeline on a smaller project to validate speed and efficiency.

Try interpretability tools (e.g., SHAP or LIME) to explain model predictions and understand PCA component influences.

Consider more advanced fraud-specific models like Isolation Forests or Autoencoders.
