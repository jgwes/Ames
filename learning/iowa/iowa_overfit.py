# Simple model -- overfit to sample, not a good model
# Four Steps
# 1 - Specify prediction target y
# 2 - Create X the feature data set
# 3 - Specify model and fit to X, y
# 4 - Make predictions

import pandas as pd

i_file_path = "./learning/iowa/data/train.csv"

home_data = pd.read_csv(i_file_path)

home_data.columns

feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

# Step 1 Specify prediction target
y = home_data.SalePrice

# Step 2 Create feature data set
X = home_data[feature_names]

print(X.describe())
print(X.head())

# Step 3 specify model & fit to X & y
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X,y)

# Step 4 make predictions
predictions = iowa_mode.predict(X)
print(predictions.describe())
print(predictions.head())
