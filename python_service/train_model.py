import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import mysql.connector


connection = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    port=3306,
    password='Pavilion227',
    database='nyiso_database'
)

cursor = connection.cursor()
query = "SELECT * FROM generator_prices;"
cursor.execute(query)
results = cursor.fetchall()

df = pd.DataFrame(results, columns=['id', 'time_stamp', 'name', 'ptid', 'lbmp', 'marginal_cost_losses', 'marginal_cost_congestion'])
df.rename(columns={
    'id': 'ID',
    'time_stamp': 'Time Stamp',
    'name': 'Name',
    'ptid': 'PTID',
    'lbmp': 'LBMP ($/MWHr)',
    'marginal_cost_losses': 'Marginal Cost Losses ($/MWHr)',
    'marginal_cost_congestion': 'Marginal Cost Congestion ($/MWHr)'
}, inplace=True)

df = df.drop_duplicates(subset=['Time Stamp', 'PTID'])
# Reset the index
df = df.reset_index(drop=True)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Feature selection
X = df[['LBMP ($/MWHr)', 'Marginal Cost Losses ($/MWHr)', 'Marginal Cost Congestion ($/MWHr)']]
y = df['LBMP ($/MWHr)']  # Target variable is the LBMP price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=200, num=100)],
    'max_depth': [None] + [int(x) for x in np.linspace(start=10, stop=110, num=11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,  # Number of folds in cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=2,  # Verbosity level
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("Best Parameters:", best_params)

# Make predictions on the test set with the best model
y_pred = best_model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Serialize the best model
joblib.dump(best_model, '/Users/danielbrown/Desktop/WebApps/NYISO_PyRustGo/python_service/best_model.pkl')
print("Best model trained and saved to /Users/danielbrown/Desktop/WebApps/NYISO_PyRustGo/ml_model/best_model.pkl")
