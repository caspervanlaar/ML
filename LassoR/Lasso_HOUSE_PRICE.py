import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import seaborn as sns
import joblib
import os

# Try to load saved model first
if os.path.exists('lasso_model.joblib'):
    print("Loading saved model...")
    saved_data = joblib.load('lasso_model.joblib')  # Load as saved_data instead of model_data
    final_model = saved_data['model']
    scaler = saved_data['scaler']
    best_alpha = saved_data['best_alpha']
    best_score = saved_data['best_score']
    best_features = saved_data['best_features']
    features = saved_data['features']
    alphas = saved_data['alphas']
    train_scores = saved_data['train_scores']
    test_scores = saved_data['test_scores']
    coef_counts = saved_data['coef_counts']

    
else:
    print("No saved model found. Training new model...")
    # Load the data
    print("Loading dataset...")
    df = pd.read_csv('kc_house_data.csv')

    # Select specified features
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
               'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
               'lat', 'long', 'sqft_living15', 'sqft_lot15', 'condition']

    X = df[features]
    y = df['price']

    # Scale y values as well (this can help with convergence)
    y = np.log1p(y)  # Log transform prices to reduce scale

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a range of alphas to test
    alphas = np.logspace(-4, 4, 100)  # Test 100 alpha values from 10^-4 to 10^4

    # Lists to store scores
    train_scores = []
    test_scores = []
    coef_counts = []  # To track number of features used

    print("\nTesting different alpha values...")
    for alpha in tqdm(alphas):
        # Create and train model
        lasso = Lasso(alpha=alpha, max_iter=100000, tol=1e-4)
        
        # Get cross-validation score
        cv_scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Fit model to get coefficients
        lasso.fit(X_train_scaled, y_train)
        
        # Store results
        train_scores.append(np.mean(cv_scores))
        test_scores.append(r2_score(y_test, lasso.predict(X_test_scaled)))
        coef_counts.append(np.sum(lasso.coef_ != 0))  # Count non-zero coefficients

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot 1: R² scores vs alpha
    plt.subplot(2, 1, 1)
    plt.semilogx(alphas, train_scores, label='Train R²', marker='o')
    plt.semilogx(alphas, test_scores, label='Test R²', marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Alpha Parameter')
    plt.legend()
    plt.grid(True)

    # Plot 2: Number of features vs alpha
    plt.subplot(2, 1, 2)
    plt.semilogx(alphas, coef_counts, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Number of Features Used')
    plt.title('Feature Count vs Alpha Parameter')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Find best alpha
    best_idx = np.argmax(test_scores)
    best_alpha = alphas[best_idx]
    best_score = test_scores[best_idx]
    best_features = coef_counts[best_idx]

    print(f"\nBest alpha: {best_alpha:.6f}")
    print(f"Best R² score: {best_score:.4f}")
    print(f"Number of features used: {best_features}")

    # Train final model with best alpha
    final_model = Lasso(alpha=best_alpha, max_iter=100000, tol=1e-4)
    final_model.fit(X_train_scaled, y_train)

    # Save the model and related data
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'best_score': best_score,
        'best_features': best_features,
        'features': features,
        'alphas': alphas,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'coef_counts': coef_counts
    }
    joblib.dump(model_data, 'lasso_model.joblib')

# Show coefficients for best model
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': final_model.coef_
})
coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values('Coefficient', key=abs, ascending=False)

print("\nNon-zero coefficients:")
print(coef_df)

# Save results
with open('lasso_alpha_analysis.txt', 'w') as f:
    f.write("Lasso Alpha Analysis Results\n")
    f.write("===========================\n\n")
    f.write(f"Best alpha: {best_alpha:.6f}\n")
    f.write(f"Best R² score: {best_score:.4f}\n")
    f.write(f"Number of features used: {best_features}\n\n")
    f.write("Non-zero coefficients:\n")
    f.write(coef_df.to_string())

# Load test data for predictions
df = pd.read_csv('kc_house_data.csv')
X = df[features]
y = df['price']
y = np.log1p(y)  # Log transform prices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = final_model.predict(X_test_scaled)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

# After model training and predictions, create the visualization plots

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted Plot
plt.subplot(3, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)

# 2. Residuals vs Predicted Values
plt.subplot(3, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

# 3. Q-Q Plot
plt.subplot(3, 2, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

# 4. Residuals Distribution
plt.subplot(3, 2, 4)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Distribution of Residuals')

# 5. Lasso Coefficients
plt.subplot(3, 2, 5)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': final_model.coef_
})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)
plt.barh(y=coef_df['Feature'], width=coef_df['Coefficient'])
plt.title('Lasso Coefficients (Feature Importance)')

# 6. Regularization Path
plt.subplot(3, 2, 6)
plt.semilogx(alphas, train_scores, label='Train R²', alpha=0.7)
plt.semilogx(alphas, test_scores, label='Test R²', alpha=0.7)
plt.axvline(best_alpha, color='r', linestyle='--', 
            label=f'Best Alpha = {best_alpha:.6f}')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R² Score')
plt.title('Regularization Path')
plt.legend()

plt.tight_layout()
plt.show()

# Additional diagnostic plots
plt.figure(figsize=(20, 5))

# 7. Residuals vs Feature Values
for i, feature in enumerate(features[:3]):  # Plot first 3 most important features
    plt.subplot(1, 3, i+1)
    plt.scatter(X_test[feature], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(feature)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs {feature}')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nModel Summary:")
print("=" * 50)
print(f"Best Alpha: {best_alpha:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print("\nFeature Importance:")
print("-" * 50)
importance_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
print(importance_df)
print("\nResiduals Statistics:")
print("-" * 50)
print(f"Mean of Residuals: {np.mean(residuals):,.2f}")
print(f"Std of Residuals: {np.std(residuals):,.2f}")
print(f"Skewness: {stats.skew(residuals):.3f}")
print(f"Kurtosis: {stats.kurtosis(residuals):.3f}")
