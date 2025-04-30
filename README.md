# Diabetes Prediction Using Multiple ML Algorithms

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blue)
![Python](https://img.shields.io/badge/Python-3.7+-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📋 Overview

This project aims to predict diabetes occurrence using various machine learning classification algorithms. The implementation compares the performance of Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, and K-Nearest Neighbors (KNN) algorithms on the Pima Indians Diabetes dataset.

## 🔍 Dataset

The project uses the Pima Indians Diabetes Database, which contains diagnostic measurements for females of Pima Indian heritage. The dataset includes features such as:

- Pregnancies
- Glucose level
- Blood Pressure
- Skin Thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (Target variable: 1 indicates diabetes, 0 indicates no diabetes)

## 🛠️ Features

- Data preprocessing and exploration
- Correlation matrix visualization with heatmap
- Feature selection based on correlation analysis
- Implementation of multiple machine learning algorithms:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
- Performance comparison between algorithms
- Prediction accuracy assessment

## 🔧 Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📥 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Usage

1. Make sure you have the dataset file in the project directory.
2. Run the script:
```bash
python diabetes_prediction.py
```

3. The script will:
   - Load and preprocess the dataset
   - Display correlation heatmap
   - Train different machine learning models
   - Display prediction results and accuracy scores

## 📝 Code Structure

```python
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors

# Load and explore data
df = pd.read_csv('pima-indians-diabetes.csv')  # Note: Replace with actual filename
mat = df.corr()
sns.heatmap(mat)
plt.show()
print(df.columns)

# Feature selection
x = df[['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# Split dataset
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.95)

# Train and evaluate multiple models
# 1. Logistic Regression
# 2. Support Vector Machine (SVM)
# 3. Decision Tree
# 4. Random Forest
# 5. K-Nearest Neighbors (KNN)
```

## 📈 Results

The project evaluates multiple machine learning algorithms and compares their accuracy scores. Based on initial runs, the models show different performance characteristics:

- Logistic Regression: Good baseline performance
- SVM: Handles non-linear decision boundaries well
- Decision Tree: Provides interpretable rules
- Random Forest: Strong performance with ensemble learning
- KNN: Effective for clustered data patterns

## 🔮 Future Improvements

- Implement cross-validation for more robust model evaluation
- Add hyperparameter tuning for each algorithm
- Incorporate more advanced preprocessing techniques
- Add visualization for model decision boundaries
- Implement imbalanced learning techniques if needed
