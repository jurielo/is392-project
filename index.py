import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, davies_bouldin_score

y_pred = None #predicted values placeholder
y_true = None #true values placeholder
df = pd.read_csv('archive/Clean_Dataset.csv') #read the dataset

#Encoding Categorical Columns

cols_encode = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'] #columns we need to encode (categorical columns)

encoder = OneHotEncoder(sparse_output=False) #intialize OneHotEncoder
encode_data = encoder.fit_transform(df[cols_encode]) #encode the data

df_encoded = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(cols_encode)) #new dataframe with encoding

edf = pd.concat([df.drop(columns=cols_encode), df_encoded], axis=1) #drops categorical and makes new datafram with encoding along columns

print(edf.head()) #sanity check

# Split the data into training and testing sets

X = edf.iloc[:, edf.columns != 'price'] #training data columns
y = edf.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0) #5% test 95% training split

print(X_test.head()) #print
print(X_train.head())
print(y_test.head())
print(y_train.head())


# Metric Analysis

# Linear Regression Metrics: RMSE, R2 Score
try:
    mse = mean_squared_error(y_true, y_pred) #mean squared error calculation using y_true and y_pred
    rmse = np.sqrt(mse) #root mean squared error conversion
    print("\nRMSE: ", rmse)
except Exception as e:
    print("\nError calculating MSE/RMSE: ", e)

try:
    r2 = r2_score(y_true, y_pred) #r2 score calculation using y_true and y_pred
    print("\nR2 Score: ", r2)
except Exception as e:
    print("\nError calculating R2 Score: ", e)

#Dbies-Bouldin Index Metric
try:
    dbScore = davies_bouldin_score(edf, y) #davies-bouldin index calculation using encoded dataframe and y
    print("\nDavies-Bouldin Index: ", dbScore)
except Exception as e:
    print("\nError calculating Davies-Bouldin Index: ", e)