import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load datasets
X_train = pd.read_csv('weatherAUS_X_train.csv')
y_train = pd.read_csv('weatherAUS_y_train.csv')

# Function to handle missing values
def impute_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

# Function to handle outliers using IQR
def handle_outliers(df, numerical_cols):
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[numerical_cols] = df[numerical_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)
    return df

# Preprocessing
X_train = impute_missing_values(X_train)
y_train.dropna(inplace=True)  # Remove missing target values
X_train = X_train.loc[y_train.index]

numerical_cols = X_train.select_dtypes(exclude=['object']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LinearSVC': LinearSVC(random_state=42, dual=False),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
}

# Feature selection using SelectFromModel with LinearSVC
feature_selector = SelectFromModel(LinearSVC(random_state=42, dual=False), max_features=10)

# Train-test split
X_train_processed = preprocessor.fit_transform(X_train)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_processed, y_train.values.ravel(), test_size=0.2, random_state=42)

# Fit feature selector
feature_selector.fit(X_train_split, y_train_split)
X_train_selected = feature_selector.transform(X_train_split)
X_val_selected = feature_selector.transform(X_val)

# Train models and evaluate
results = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train_split)
    y_pred = model.predict(X_val_selected)
    
    accuracy = accuracy_score(y_val, y_pred)
    class_report = classification_report(y_val, y_pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{class_report}\n")

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'classification_report': class_report
    }

# Function to plot ROC curve
def plot_roc_curves(models, X_val, y_val):
    plt.figure(figsize=(10, 7))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_val)[:, 1]
        else:
            probabilities = model.decision_function(X_val)
        
        fpr, tpr, _ = roc_curve(y_val, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curves
plot_roc_curves(models, X_val_selected, y_val)
