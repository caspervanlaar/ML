import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import joblib
import os

class CoordinateDescentSVR:
    def __init__(self, learning_rate=0.1, epsilon=0.1, C=1.0, max_iterations=1000):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.C = C
        self.max_iterations = max_iterations
        self.losses = []
        self.r2_scores = []
        
    def rbf_kernel(self, x1, x2):
        gamma = 1.0 / x1.shape[0]  # Automatic gamma selection
        return np.exp(-gamma * np.sum((x1 - x2) ** 2))
    
    def compute_kernel_matrix(self, X):
        # Check if kernel matrix exists
        if os.path.exists('kernel_matrix.joblib'):
            print("\nLoading existing kernel matrix...")
            K = joblib.load('kernel_matrix.joblib')
            print("Kernel matrix loaded successfully!")
            print(f"Matrix shape: {K.shape}")
            print(f"Matrix statistics:")
            print(f"Mean value: {np.mean(K):.4f}")
            print(f"Std deviation: {np.std(K):.4f}")
            print(f"Min value: {np.min(K):.4f}")
            print(f"Max value: {np.max(K):.4f}")
            return K

        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        print(f"\nComputing {n_samples}x{n_samples} kernel matrix...")
        
        # Create progress bar for outer loop
        for i in tqdm(range(n_samples), desc="Computing kernel rows"):
            # Print percentage complete every 10%
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"Progress: {((i + 1) / n_samples * 100):.1f}% complete")
                print(f"Computing kernel values for sample {i + 1}/{n_samples}")
                
            for j in range(n_samples):
                K[i,j] = self.rbf_kernel(X[i], X[j])
                
            # Print some kernel values for monitoring
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"Sample kernel values for row {i + 1}: {K[i, :5]}")
        
        print("\nKernel matrix computation completed!")
        print(f"Matrix shape: {K.shape}")
        print(f"Matrix statistics:")
        print(f"Mean value: {np.mean(K):.4f}")
        print(f"Std deviation: {np.std(K):.4f}")
        print(f"Min value: {np.min(K):.4f}")
        print(f"Max value: {np.max(K):.4f}")
        
        # Save kernel matrix
        joblib.dump(K, 'kernel_matrix.joblib')
        print("Kernel matrix saved to kernel_matrix.joblib")
        
        return K
    
    def compute_loss(self, predictions, y):
        # SVR loss function with epsilon-insensitive loss
        errors = np.abs(predictions - y)
        return np.mean(np.maximum(0, errors - self.epsilon) ** 2)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        
        # Initialize parameters
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        print("Computing kernel matrix...")
        self.K = self.compute_kernel_matrix(X)
        
        # Setup live plotting
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        loss_line, = ax1.plot([], [], 'b-', label='Loss')
        r2_line, = ax2.plot([], [], 'g-', label='R² Score')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True)
        ax1.legend()
        plt.savefig('training_plots.png', bbox_inches='tight', dpi=300)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Over Time')
        ax2.grid(True)
        ax2.legend()
        
        # Coordinate descent
        print("\nStarting coordinate descent...")
        pbar = tqdm(range(self.max_iterations))
        
        for iteration in pbar:
            old_alpha = self.alpha.copy()
            
            # Shuffle coordinates for randomized coordinate descent
            coords = np.random.permutation(n_samples)
            
            for i in coords:
                # Current predictions
                predictions = np.sum(self.alpha.reshape(-1, 1) * self.K, axis=0) + self.b
                
                # Compute error for current point
                error = predictions[i] - y[i]
                
                if abs(error) > self.epsilon:
                    # Compute coordinate update
                    grad = error * self.K[i,i]
                    
                    # Update alpha with box constraints
                    old_alpha_i = self.alpha[i]
                    self.alpha[i] = np.clip(
                        old_alpha_i - self.learning_rate * grad,
                        -self.C,
                        self.C
                    )
                    
                    # Update bias
                    delta_alpha = self.alpha[i] - old_alpha_i
                    if delta_alpha != 0:
                        self.b -= self.learning_rate * error
            
            # Compute metrics
            predictions = np.sum(self.alpha.reshape(-1, 1) * self.K, axis=0) + self.b
            current_loss = self.compute_loss(predictions, y)
            current_r2 = r2_score(y, predictions)
            
            self.losses.append(current_loss)
            self.r2_scores.append(current_r2)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{current_loss:.2e}',
                'R²': f'{current_r2:.4f}'
            })
            
            # Update live plot
            if iteration % 10 == 0:
                loss_line.set_data(range(len(self.losses)), self.losses)
                r2_line.set_data(range(len(self.r2_scores)), self.r2_scores)
                
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                plt.draw()
                plt.pause(0.01)
            
            # Check convergence
            if np.allclose(self.alpha, old_alpha, rtol=1e-5):
                print(f"\nConverged after {iteration + 1} iterations")
                break
        
        plt.ioff()
        plt.show()
        
        # Store support vectors
        sv_idx = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.support_vectors = X[sv_idx]
        self.support_alpha = self.alpha[sv_idx]
        
        print("\nFinal training metrics:")
        print(f"Loss: {self.losses[-1]:.2e}")
        print(f"R² Score: {self.r2_scores[-1]:.4f}")
        print(f"Number of support vectors: {len(self.support_vectors)}")
        
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            for sv, alpha in zip(self.support_vectors, self.support_alpha):
                predictions[i] += alpha * self.rbf_kernel(x, sv)
        predictions += self.b
        return predictions

# Load and prepare data
df = pd.read_csv('kc_house_data.csv')

# Calculate correlation with price
correlations = df.corr()['price'].abs().sort_values(ascending=False)
print("\nFeature correlations with price:")
print(correlations)

# Get top 2 features (excluding price itself)
top_features = correlations[1:3].index.tolist()
print(f"\nTop 2 features: {top_features}")

# Separate features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Before splitting the data, reset the indices
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# After splitting, make sure train sets have continuous indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Before the model fitting, convert X_train_scaled and y_train to numpy arrays
X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)

# Similarly for test data
X_test_scaled = np.array(X_test_scaled)
y_test = np.array(y_test)

# Create and train the model with different learning rates
learning_rates = [0.1, 0.01, 0.001]
best_r2 = -np.inf
best_model = None
best_lr = None

for lr in learning_rates:
    print(f"\nTrying learning rate: {lr}")
    model = CoordinateDescentSVR(
        learning_rate=lr,
        epsilon=0.1,
        C=100.0,
        max_iterations=1000
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    print(f"R² score: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_lr = lr

print(f"\nBest learning rate: {best_lr}")
print(f"Best R² score: {best_r2:.4f}")

# Make predictions with best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nFinal Model Performance:')
print(f'Mean Squared Error: {mse:,.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mse):,.2f}')
print(f'R-squared Score: {r2:.4f}')

# Replace the 3D visualization with scatter plots
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Optional: Feature-specific scatter plots
plt.figure(figsize=(15, 5))

# Plot for top feature 1
plt.subplot(1, 2, 1)
plt.scatter(X_test[top_features[0]], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test[top_features[0]], y_pred, alpha=0.5, label='Predicted')
plt.xlabel(top_features[0])
plt.ylabel('Price')
plt.title(f'Price vs {top_features[0]}')
plt.legend()

# Plot for top feature 2
plt.subplot(1, 2, 2)
plt.scatter(X_test[top_features[1]], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test[top_features[1]], y_pred, alpha=0.5, label='Predicted')
plt.xlabel(top_features[1])
plt.ylabel('Price')
plt.title(f'Price vs {top_features[1]}')
plt.legend()

plt.tight_layout()
plt.show()

# Test prediction with new house
new_house = np.array([[3, 2, 2000, 5000, 2, 7, 1800, 200, 1990, 0, 47.6062, -122.3321, 1900, 4900, 3]])
new_house_scaled = scaler.transform(new_house)
predicted_price = best_model.predict(new_house_scaled)
print(f'\nPredicted Price: ${predicted_price[0]:,.2f}')
