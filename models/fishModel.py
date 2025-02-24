# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Getting the dataset
df = pd.read_csv('/home/sourav/ml/datasets/Fish.csv')

# Encoding categorical data
te = TargetEncoder()
df['Species_Encoded'] = te.fit_transform(df[['Species']], df['Weight'])

# Drop original Species column
df.drop(columns=['Species'], inplace=True)

# Dropping columns to remove multicollinearity
df = df.drop(columns=['Length2', 'Length3'])

# Dividing features and target variable
X = df.drop(columns=['Weight']).values
y = df['Weight'].values
print("Feature matrix shape:", np.shape(X))

# Plotting histograms
df.hist(figsize=(10, 12), bins=20, grid=False)
plt.tight_layout()
plt.savefig('histogram.png')

# Checking for correlations
plt.figure(figsize=(10, 12))
sns.heatmap(df.corr().abs(), annot=True, cmap="coolwarm")
plt.savefig('correlation_matrix.png')

# Log Transformation (to handle skewness in Weight)
df['Weight'] = np.log1p(df['Weight'])  # log(1 + x) to avoid log(0)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Building and training the model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating performance
print("R² Score:", r2_score(y_test, y_pred))
print("Cross-validation scores:", cross_val_score(model, X, y, cv=5))

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score from GridSearchCV:", grid_search.best_score_)

# Train the model with best parameters
best_rf = grid_search.best_estimator_

# Predict and evaluate with tuned model
y_pred_best = best_rf.predict(X_test)
print("Final R² Score (Tuned Model):", r2_score(y_test, y_pred_best))

# Feature Importance Plot
feature_importances = best_rf.feature_importances_
feature_names = df.drop(columns=['Weight']).columns  
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[sorted_indices], rotation=45, ha="right")
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance in Random Forest')
plt.savefig('feature_importance.png')
