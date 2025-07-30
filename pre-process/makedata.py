# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Step 1: Create dummy data
# np.random.seed(42)
# n_samples = 200

# data = pd.DataFrame({
#     'CarValue': np.random.randint(500000, 3000000, size=n_samples),
#     'CarAge': np.random.randint(1, 20, size=n_samples),
#     'OwnerAge': np.random.randint(18, 70, size=n_samples),
#     'VehicleType': np.random.choice(['Sedan', 'SUV', 'Pickup', 'Motorbike'], size=n_samples),
#     'UseType': np.random.choice(['Personal', 'Commercial'], size=n_samples),
#     'OwnershipCount': np.random.randint(1, 5, size=n_samples),
#     'AccidentHistory': np.random.choice([0, 1], size=n_samples),
# })

# # Target: Premium (simulate with some pattern + noise)
# data['Premium'] = (
#     0.02 * data['CarValue'] -
#     1000 * data['CarAge'] +
#     500 * data['AccidentHistory'] +
#     np.random.normal(0, 5000, size=n_samples)
# )

# # Step 2: One-hot encode categorical variables
# data_encoded = pd.get_dummies(data, columns=['VehicleType', 'UseType'], drop_first=True)

# # Step 3: Train-test split
# X = data_encoded.drop('Premium', axis=1)
# y = data_encoded['Premium']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train XGBoost model
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# model.fit(X_train, y_train)

# # Step 5: Evaluate the model
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# mae, rmse, r2, data_encoded.head()



import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
data.plot(kind="scatter", x='CarAge', y='OwnerAge')
plt.show()
print()