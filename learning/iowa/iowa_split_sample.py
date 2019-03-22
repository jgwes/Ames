import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

iowa_file_path = "./data/train.csv"
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_columns = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify model
iowa_model = DecisionTreeRegressor()
# Fit model
iowa_model.fit(X,y)

print('First in-sample predictions:', iowa_model.predict(X.head()))
print('Acutal target values for those homes:', y.head().tolist())

# First in-sample predictions: [208500. 181500. 223500. 140000. 250000.]
# Actual target values for those homes: [208500, 181500, 223500, 140000, 250000]
# Overfit!

# Step 1 - Split home_data
#   Split data into training and test sets
#   To fit model on traning data and predict on test data
# Split example:
# >>> X_train, X_test, y_train, y_test = train_test_split(
# ...     X, y, test_size=0.33, random_state=42)
# >>> X_train
# array([[4, 5],
#        [0, 1],
#        [6, 7]])
# >>> y_train
#  [2, 0, 3]
# >>> X_test
# array([[2, 3],
#       [8, 9]])
# >>> y_test
# [1, 4]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Step 2 - Specify and Fit the model with training data
iowa_model.fit(train_X, train_y)

# Step 3 - Make predictions with validation data
val_predictions = iowa_model.predict(val_X)

# print top validation val_predictions
print('Top validation predictions:', iowa_model.predict(val_X.head()))
# print top prices from validation data
print('Top prices from validation data:', val_y.head())

# Step 4 - Calculate Mean Absolute Error
val_mae = mean_absolute_error(val_y, iowa_model.predict(val_X))
print(val_mae)
