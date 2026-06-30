# Risk Alert Classifier

A classification project built for a digital banking platform to flag high-risk customers (likely to default or commit fraud) using their demographic, behavioral and transaction data.

## Objective

Banks want an early-warning system that can catch risky customers before they default or commit fraud. The catch is that risky customers are a small minority in the data, so the goal of this project isn't just to build a model, but to build one that actually handles this imbalance properly and gets evaluated using the right metrics instead of plain accuracy.

## Problem Statement

Working as a Data Scientist on this dataset, the task was to:
- Identify high-risk customers accurately
- Handle the class imbalance in the data
- Evaluate the model using metrics that actually matter for this problem (Recall, F1-Score, AUC-ROC, not just accuracy)
- Improve performance through hyperparameter tuning

## Dataset

The dataset (`Risk_Alert_Classifier_Dataset.csv`) has 4600 customer records and 19 columns covering:
- Demographic info (age, gender, region, employment type)
- Financial details (annual income, credit score, debt balance)
- Behavioral/transaction indicators (missed payments, late payment days, monthly spend, failed login attempts, complaints)
- Target column: `risk_status` (0 = Low Risk, 1 = High Risk)

Only about 12% of the customers are labeled High Risk, so the dataset is clearly imbalanced.

## Project Structure

```
├── Risk_Alert_Classifier.ipynb     # Main notebook (EDA, preprocessing, modeling, evaluation)
├── PART_A_Theory.pdf               # Theory answers (Part A - conceptual questions)
├── Risk_Alert_Classifier_Dataset.csv
└── README.md
```

## What's Inside the Notebook

1. **Data Understanding** - shape, column info, data types, missing values, duplicates, target distribution
2. **EDA** - distribution plots for key numerical columns, boxplots to check outliers, correlation heatmap
3. **Preprocessing** - date column converted to year/month, categorical columns label encoded, missing values filled using KNN Imputer
4. **Baseline Model** - Logistic Regression with confusion matrix, accuracy, precision/recall/F1, and Type-I/Type-II error breakdown
5. **Handling Imbalance** - Under-Sampling, Over-Sampling, SMOTE and ADASYN compared side by side on Recall, F1-Score and AUC-ROC
6. **Tree-Based Models** - Decision Tree and Random Forest, including a check for overfitting (train vs test accuracy)
7. **Hyperparameter Tuning** - Randomized Search CV followed by a focused Grid Search CV
8. **Model Evaluation** - ROC curves and AUC-ROC scores for the final model
9. **Final Report** - best model selection, impact of balancing, and business interpretation of false positives/negatives

## Results Summary

| Step | Result |
|---|---|
| Baseline Logistic Regression | Struggled with Recall before balancing the data |
| After SMOTE | Recall and F1-Score for the High Risk class improved noticeably |
| Random Forest (tuned) | ~99.7% accuracy, 0.99 F1-Score on High Risk class |
| Final AUC-ROC Score | ~0.9999 |

**Best model:** Random Forest (after hyperparameter tuning), chosen mainly for its strong Recall on the High Risk class, since missing an actual risky customer (a False Negative) is more costly for a bank than flagging a safe customer for extra review.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Decision Tree, Random Forest, RandomizedSearchCV, GridSearchCV)
- imbalanced-learn (Under-Sampling, Over-Sampling, SMOTE, ADASYN)
- matplotlib, seaborn for visualization

## How to Run

1. Clone this repository
2. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
3. Open `Risk_Alert_Classifier.ipynb` in Jupyter Notebook / JupyterLab
4. Run all cells from top to bottom

## Note

The dataset used here is fairly cleanly separable between the two classes, which is why some of the models score very high (close to 1.0 AUC-ROC). This is a property of this particular dataset rather than a guarantee of real-world performance, and would be worth keeping in mind if this approach were ever applied to live banking data.
