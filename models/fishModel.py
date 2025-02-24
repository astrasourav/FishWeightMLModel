
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.linear_model import LinearRegression 
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Gettign the dataset
df = pd.read_csv('/home/sourav/ml/datasets/Fish.csv')

#Encoding categorical data
te = TargetEncoder()
df['Species_Encoded'] = te.fit_transform(df[['Species']], df['Weight'])

# Drop original Species column
df.drop(columns=['Species'], inplace=True)


#Dropping data to remove multicolinearity 
df = df.drop(columns=['Length2', 'Length3'])
# print(df.head())

# Dividing features and  target variable
x = df.drop(columns=['Weight']).values
y = df['Weight'].values
print(np.shape(x))


# Ploting data
df.hist(figsize=(10, 12), bins=20, grid=False)
plt.tight_layout()
plt.savefig('histograma.png')

# # Checing for corelations
plt.figure(figsize = (10, 12))
sns.heatmap(df.corr().abs(), annot = True)
plt.savefig('corr.png')


# Log Transformation
# print(df.skew())
df['Weight'] = np.log1p(df['Weight'])  # log(1 + x) to avoid log(0)


# Split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# # # # Building and trainig model
# model = LinearRegression()
# model = Ridge(alpha=100)
# model = Lasso()
# model = ElasticNet(random_state=0, alpha=0.5)
model = RandomForestRegressor(n_estimators = 100, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

 # Checking accuracy
print(r2_score(y_test, y_pred))
print(cross_val_score(model, x, y, cv=5))

# print(model.coef_ == 0)



# HyperParametr Tuning 
model2 = RandomForestRegressor(**best_params, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=model2, 
    param_grid=param_grid, 
    cv=5,       # 5-fold cross-validation
    verbose=2,
    n_jobs=-1,
    scoring='r2'
)

grid_search.fit(x_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

# Train model with best parameters
best_rf = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_rf.predict(x_test)
print("Final R² Score:", r2_score(y_test, y_pred))



# # Plotting which is the most significant feature 
feature_importances = model.feature_importances_

# # Get feature names (assuming x is a DataFrame before conversion to numpy array)
feature_names = df.drop(columns=['Weight']).columns  

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]

Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[sorted_indices], rotation=45, ha="right")
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance in Random Forest')
plt.savefig('Feature.png')