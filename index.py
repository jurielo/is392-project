'''
Python Script for Data Preprocessing, Model Training, and Evaluation of flight data for price prediction.
Uses a cleaned dataset from https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
    - Downloaded and kept in 'archive' folder as 'Clean_Dataset.csv'

Expected outputs:
- Encoded DataFrame head to verify categorical encoding.
- Training and testing data heads.
- R2 Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE) for Linear Regression, Ridge Regression, and Lasso Regression models.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, davies_bouldin_score

#read the dataset
df = pd.read_csv('archive/Clean_Dataset.csv')

'''Encoding Categorical Columns'''

cols_encode = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'] #columns we need to encode (categorical columns)

encoder = OneHotEncoder(sparse_output=False) #intialize OneHotEncoder
encode_data = encoder.fit_transform(df[cols_encode]) #encode the data

df_encoded = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(cols_encode)) #new dataframe with encoding

edf = pd.concat([df.drop(columns=cols_encode), df_encoded], axis=1) #drops categorical and makes new datafram with encoding along columns

print(edf.head()) #Output encoded dataframe head to verify encoding

''' Split the data into training and testing sets'''

X = edf.drop(columns=['price', 'flight']) #training data columns
y = edf.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0) #5% test 95% training split

print(X_test.head()) #print
print(X_train.head())
print(y_test.head())
print(y_train.head())

'''Visualization'''

''' Model Training '''
# Linear Regression Model
print("Training linear regression model...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print("Done training linear regression.")

# Ridge Regression Model
print("Training ridge regression regression model...")
ridge_reg = Ridge(alpha=1) 
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
print("Done training ridge regression.")

# Lasso Regression Model: needs scaled data for faster weight convergences
print("Training lasso regression model...")
lasso_reg = Lasso(max_iter=10000, random_state=42,alpha=0.01)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)
print("Done lasso linear regression.")

'''Metric Analysis'''
# Linear Regression Metrics: RMSE, R2 Score for all linear models
#list of linear model prediction outputs
LinearPredictions =  {
    'Linear Regression Model': y_pred_lin,
    'Ridge Regression Model': y_pred_ridge,
    'Lasso Regression Model': y_pred_lasso
}

for name, y_pred in LinearPredictions.items():
    #print model name
    print("\n",name,":")

    try:
        #mean squared error calculation using y_true and y_pred
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) #root mean squared error conversion

        #print results
        print("MSE: ", mse)
        print("RMSE: ", rmse)
    except Exception as e:
        print("\nError calculating MSE/RMSE: ", e)

    try:
        #r2 score calculation using y_true and y_pred
        r2 = r2_score(y_test, y_pred)

        #print results
        print("R2 Score: ", r2)
    except Exception as e:
        print("\nError calculating R2 Score: ", e)

'''
#Dbies-Bouldin Index Metric (not implemented yet need y from model)
try:
    dbScore = davies_bouldin_score(edf, y) #davies-bouldin index calculation using encoded dataframe and y
    print("\nDavies-Bouldin Index: ", dbScore)
except Exception as e:
    print("\nError calculating Davies-Bouldin Index: ", e)
'''