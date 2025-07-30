import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#loading the data 
data = pd.read_csv("insurance.csv")

#eliminating outliers if any
Q1 = data["Premium"].quantile(0.25)
Q3 = data["Premium"].quantile(0.75)
IQR = Q3 - Q1
data = data[(data["Premium"] >= Q1 - 1.5*IQR) & (data["Premium"] <= Q3 + 1.5*IQR)]
data["Premium"] = data["Premium"].abs().round(2)

#selecting features for use
categorical_columns=["VehicleType","UseType"]
X= data.drop(columns="Premium")
y = data["Premium"]


#one hot encoding and standard scaling for categorical data 
ct = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(drop="first"),categorical_columns),
    ("scale", StandardScaler(), X.columns.difference(categorical_columns))
])
transformed = ct.fit_transform(X)

#selcting the best features for training the model 
selector = SelectKBest(score_func=f_regression, k=5)
selected = selector.fit_transform(transformed,y)

#splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(selected, y, test_size=0.2, random_state=2)

#inittialising the model
Model = RandomForestRegressor(n_estimators=150, min_samples_split=2,min_samples_leaf=4,max_depth=10,random_state=1)

#training the model
Model.fit(X_train,y_train)

#making the predctions
preds = Model.predict(X_test)

#evaluating the model
r2 = (r2_score(y_test,preds))*100
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Percentage R²: {r2:.2f}% \n Mean Absolute Error: {mae:.2f} \n Root Mean Squared Error: {rmse:.2f}")

#correlation heatmap for the used features
data_copy = data.copy()
data_copy[categorical_columns] = OrdinalEncoder().fit_transform(data_copy[categorical_columns]).astype(int)
corr = data_copy.corr().abs()
sns.heatmap(corr, annot=True)
plt.show()

# joblib.dump(Model, "premium_model.pkl")
# joblib.dump(ct, "Transformer.pkl")
# joblib.dump(selector,"Feature_selector.pkl")
