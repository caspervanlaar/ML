import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import randint, uniform

# Load and prepare data
print("Loading dataset...")
df = pd.read_csv('kc_house_data.csv')

# Handle outliers
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]

# Feature engineering
df['age'] = 2023 - df['yr_built']
df['renovated'] = (df['yr_renovated'] != 0).astype(int)
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Calculate correlation with price
correlations = df.corr()['price'].abs().sort_values(ascending=False)
print("\nFeature correlations with price:")
print(correlations)

# Get top features (excluding price itself)
top_features = correlations[1:5].index.tolist()  # Increased to top 5 features
print(f"\nTop features: {top_features}")

# Separate features (X) and target variable (y)
X = df.drop(['price', 'id', 'date'], axis=1)  # Remove irrelevant features
y = df['price']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use RobustScaler instead of StandardScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# First do RandomizedSearchCV with broader parameter space
random_param_grid = {
    'n_estimators': randint(100, 1000),
    'max_depth': [None] + list(range(10, 100, 10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'max_samples': uniform(0.5, 0.5)
}

print("\nPerforming Randomized Search for initial parameter space exploration...")
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=random_param_grid,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
print("\nBest parameters from random search:")
print(random_search.best_params_)

# Use the results from RandomizedSearchCV to define a focused parameter grid
best_random = random_search.best_params_
param_grid = {
    'n_estimators': [max(100, best_random['n_estimators'] - 100), 
                     best_random['n_estimators'], 
                     min(1000, best_random['n_estimators'] + 100)],
    'max_depth': [best_random['max_depth'] - 10 if best_random['max_depth'] is not None else None,
                  best_random['max_depth'],
                  best_random['max_depth'] + 10 if best_random['max_depth'] is not None else None],
    'min_samples_split': [max(2, best_random['min_samples_split'] - 2),
                         best_random['min_samples_split'],
                         best_random['min_samples_split'] + 2],
    'min_samples_leaf': [max(1, best_random['min_samples_leaf'] - 1),
                        best_random['min_samples_leaf'],
                        best_random['min_samples_leaf'] + 1],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [best_random['bootstrap']],
    'max_samples': [best_random['max_samples']]
}

# Use GridSearchCV with focused parameter grid
print("\nPerforming Grid Search with focused parameters...")
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print("\nBest parameters after focused grid search:")
print(grid_search.best_params_)

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\nModel Performance Metrics:')
print(f'Mean Squared Error: {mse:,.2f}')
print(f'Root Mean Squared Error: {rmse:,.2f}')
print(f'R-squared Score: {r2:.4f}')

# Create enhanced visualizations
plt.style.use('seaborn')
fig = plt.figure(figsize=(20, 15))

# Plot 1: Actual vs Predicted with confidence intervals
ax1 = plt.subplot(2, 2, 1)
scatter = ax1.scatter(y_test, y_pred, alpha=0.5, c='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price')
ax1.set_ylabel('Predicted Price')
ax1.set_title('Actual vs Predicted Prices')
ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, fontsize=12)

# Plot 2: Enhanced Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

ax2 = plt.subplot(2, 2, 2)
sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax2)
ax2.set_title('Top 10 Feature Importance')

# Plot 3: Residuals Plot
ax3 = plt.subplot(2, 2, 3)
residuals = y_test - y_pred
ax3.scatter(y_pred, residuals, alpha=0.5)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Predicted Price')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs Predicted Values')

# Plot 4: Error Distribution
ax4 = plt.subplot(2, 2, 4)
sns.histplot(residuals, kde=True, ax=ax4)
ax4.set_title('Distribution of Residuals')
ax4.set_xlabel('Residual Value')

plt.tight_layout()
plt.show()

# Save the optimized model
import joblib
joblib.dump(best_model, 'optimized_random_forest_model.joblib')
joblib.dump(scaler, 'optimized_scaler.joblib')
print("\nOptimized model and scaler saved to disk.")
