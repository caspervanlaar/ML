import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

# Create output log file
output_log = open('output_log.txt', 'w')

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    df = pd.read_csv('traumaset.csv', sep=';', decimal=',')
    
    # Fix typo in column name
    df = df.drop('Patient_ID', axis=1)  #
    
    # Label encode categorical variables
    categorical_cols = ['Hair_Phenotype', 'heart_rate', 'skin_conductance', 
                       'skin_temperature', 'cortisol_level', 'Systolic_BP', 
                       'Diastolic_BP', 'Trauma_Severity']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # Store trauma severity mapping
    trauma_severity_mapping = dict(zip(le_dict['Trauma_Severity'].transform(['Low_Severity', 'Medium_Severity', 'High_Severity']), 
                                     ['Low_Severity', 'Medium_Severity', 'High_Severity']))
        
    # Split features and target
    X = df.drop('Trauma_Severity', axis=1)
    y = df['Trauma_Severity']
    
    # Split data with shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_learning_curves(estimator, title, X, y):
    cv_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv_split,
        n_jobs=-1,
        train_sizes=train_sizes,
        scoring='accuracy'
    )
    
    # Print detailed scores
    print(f"\nDetailed scores for {title}:")
    print("Training scores:", np.mean(train_scores, axis=1))
    print("CV scores:", np.mean(val_scores, axis=1))
    print(f"Score variance: {np.var(val_scores):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
    plt.fill_between(train_sizes, 
                    np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                    alpha=0.1)
    plt.title(f'Learning Curves - {title}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'learning_curves_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'{title}\nAccuracy: {accuracy_score(y_true, y_pred):.2f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load and preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data()

# Train models
print("\nTraining models...")

# Define cross-validation split
cv_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# KNN
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1)
knn_grid.fit(X_train_scaled, y_train)

# Decision Tree
dt_params = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train_scaled, y_train)

# Dummy Classifier
dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
dummy_clf.fit(X_train_scaled, y_train)

# Get predictions
results = {
    'knn_preds': knn_grid.predict(X_test_scaled),
    'dt_preds': dt_grid.predict(X_test_scaled),
    'dummy_preds': dummy_clf.predict(X_test_scaled),
    'y_test': y_test,
    'models': {
        'knn': knn_grid.best_estimator_,
        'dt': dt_grid.best_estimator_,
        'dummy': dummy_clf
    }
}

# Save results
joblib.dump(results, 'classifier_results.joblib')

# Print model performances
print("\nModel Performances:")
for model_name in ['knn', 'dt', 'dummy']:
    print(f"\n{model_name.upper()} Classification Report:")
    print(classification_report(results['y_test'], results[f'{model_name}_preds']))
    plot_confusion_matrix(
        results['y_test'], 
        results[f'{model_name}_preds'],
        f'{model_name.upper()} Confusion Matrix',
        f'{model_name}_confusion_matrix.png'
    )
    plot_learning_curves(
        results['models'][model_name],
        f'{model_name.upper()}',
        X_train_scaled,
        y_train
    )
