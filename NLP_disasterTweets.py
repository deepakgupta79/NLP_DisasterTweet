# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, clasettsification_report

# Load the synthetic dataset from the CSV file
df = pd.read_csv('synthetic_dataset.csv')

# Display column names and information about the DataFrame
print("Column Names:", df.columns)
print("DataFrame Information:")
print(df.info())

try:
    # Assuming 'text' and 'target' columns exist in your dataset
    X = df['text'].astype(str)  # Ensure 'text' column is treated as string
    y = df['target']

    # Data Preprocessing
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(X)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Hyperparameter Tuning
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=min(5, y_train.value_counts().min()),
                               scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Visualize Hyperparameter Tuning Results
    results = pd.DataFrame(grid_search.cv_results_)

    # Plot the mean test score for each hyperparameter
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(x='param_C', y='mean_test_score', data=results, marker='o')
    plt.xscale('log')
    plt.xlabel('C (Inverse of Regularization Strength)')
    plt.ylabel('Mean Test Score (Accuracy)')
    plt.title('Hyperparameter Tuning Results')

    # Model Training with Best Hyperparameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Model Evaluation on Validation Set
    y_val_pred = best_model.predict(X_val)

    # Print evaluation metrics
    print("\nValidation Set Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Classification Report on Validation Set:\n", classification_report(y_val, y_val_pred))

    # Model Evaluation on Test Set
    y_test_pred = best_model.predict(X_test)

    # Print evaluation metrics
    print("\nTest Set Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report on Test Set:\n", classification_report(y_test, y_test_pred))

    # Visualize Box Plot for Model Performance on Validation Set
    plt.subplot(1, 2, 2)
    sns.boxplot(x=y_val, y=best_model.predict_proba(X_val)[:, 1])
    plt.xlabel('True Class')
    plt.ylabel('Predicted Probability for Class 1 (Disaster)')
    plt.title('Box Plot of Model Performance on Validation Set')

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Additional Plots for Data Analysis
    plt.figure(figsize=(16, 6))

    # Violin Plot
    plt.subplot(1, 3, 1)
    sns.violinplot(x=y, y=best_model.predict_proba(X)[:, 1])
    plt.xlabel('True Class')
    plt.ylabel('Predicted Probability for Class 1 (Disaster)')
    plt.title('Violin Plot of Predicted Probabilities')

    # Density Plot
    plt.subplot(1, 3, 2)
    sns.kdeplot(data=df, x='target', y=best_model.predict_proba(X)[:, 1], fill=True, cmap='viridis')
    plt.xlabel('True Class')
    plt.ylabel('Predicted Probability for Class 1 (Disaster)')
    plt.title('Density Plot of Predicted Probabilities')

    # Scatter Plot
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=best_model.predict_proba(X)[:, 1], y=y)
    plt.xlabel('Predicted Probability for Class 1 (Disaster)')
    plt.ylabel('True Class')
    plt.title('Scatter Plot of True Class vs Predicted Probability')

    # Display the additional plots
    plt.tight_layout()
    plt.show()

except KeyError as e:
    print(f"KeyError: {e}. Check if 'text' and 'target' columns exist in your dataset.")
except Exception as e:
    print(f"An error occurred: {e}")
