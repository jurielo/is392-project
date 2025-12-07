'''
Python Script for Data Preprocessing, Model Training, and Evaluation of flight data for price prediction.
Uses a cleaned dataset from https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
    - Downloaded and kept in 'archive' folder as 'Clean_Dataset.csv'
    - Already cleaned and normalized, encoding categorical variables needed.

Expected outputs:
- Encoded DataFrame head to verify categorical encoding.
- Training and testing data heads.
-Visualizations (not implemented yet)
- R2 Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE) for Linear Regression, Ridge Regression, and Lasso Regression models.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#read the dataset
df = pd.read_csv('archive/Clean_Dataset.csv')

'''Encoding Categorical Columns'''

cols_encode = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'] #columns we need to encode (categorical columns)

encoder = OneHotEncoder(sparse_output=False) #intialize OneHotEncoder
encode_data = encoder.fit_transform(df[cols_encode]) #encode the data

fnames = encoder.get_feature_names_out(cols_encode) #get new column names after encoding
df_encoded = pd.DataFrame(encode_data, columns=fnames, index=df.index) #new dataframe with encoding

edf = pd.concat([df.drop(columns=cols_encode), df_encoded], axis=1) #drops categorical and makes new datafram with encoding along columns

print("Encoded dataframe head:\n", edf.head()) #Output encoded dataframe head to verify encoding

'''Exploratory Data Analysis (EDA)'''

print("\n=== Dataset EDA Before Encoding ===")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

print("\n=== Price Analysis ===")
print(f"Mean price: ${df['price'].mean():.2f}")
print(f"Median price: ${df['price'].median():.2f}")
print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

''' Split the data into training and testing sets'''

X = edf.drop(columns=['price', 'flight', 'Unnamed: 0']) #training data columns
y = edf.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0) #5% test 95% training split

# will be used to evaulate our RMSE
mean_price = np.mean(y_test)
std_price = np.std(y_test)
print("Average Flight Price: $", mean_price)
print("Standard Deviation of Price: $", std_price)

#Output heads of training and testing data
print("\nData splits:\n")
print(X_test.head()) 
print(X_train.head())
print(y_test.head())
print(y_train.head())
print(X.shape)

'''Visualization'''
import matplotlib.pyplot as plt
import seaborn as sns

#price histogram
plt.hist(df['price'], bins=50, edgecolor='white')
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Price Box Plot')
plt.show()

#averag eprice perairlie
aa = df.groupby('airline')['price'].mean()
aa.plot(kind="bar", x = "airline", y = "price");
plt.title('Price Average per Airline')
plt.xlabel('Airline')
plt.ylabel('Price')
plt.show()


#price + stops (class hue)
sns.scatterplot( x = "stops", y = "price", data=df, alpha=0.5, hue="class", s=15, legend="brief")
plt.title("Price based on Stops (Class ver.)")
plt.show()

#price + stops (ariline hue)
sns.scatterplot( x = "stops", y = "price", data=df, alpha=1, hue="airline", s=15, legend="brief")
plt.title("Price based on Stops (Airline ver.)")
plt.show()

#duration hist
plt.hist(df["duration"], bins=50, edgecolor='white')
plt.title('Flight Duration Box Plot')
plt.show()

#scatter plot price + duretion
sns.scatterplot( x = "duration", y = "price",data=df, alpha=0.2, s=15)
plt.title("Price based on duration")
plt.show()

#price + days left scatter plot
sns.scatterplot( x = "days_left", y = "price",data=df, alpha=0.01, s=15)
plt.title("Price based on days left to flight")
plt.show()

#seaborn plot map of histograms based on airline and price
ap = sns.FacetGrid(df, col="airline", col_wrap=2, height=10)
ap.map(sns.histplot, "price", bins=25)
plt.show()

# map hist sourcecity + price 
scp = sns.FacetGrid(df, col="source_city", col_wrap=2, height=15)
scp.map(sns.histplot, "price", bins=45)
plt.show()

# map hist departuretime + price 
dtp = sns.FacetGrid(df, col="departure_time", col_wrap=2, height=10)
dtp.map(sns.histplot, "price", bins=50)
plt.show()

# mh arrivaltime + rpice
atp = sns.FacetGrid(df, col="arrival_time", col_wrap=2, height=15)
atp.map(sns.histplot, "price", bins=40)
plt.show()

#mh destcity + pricee
dcp = sns.FacetGrid(df, col="destination_city", col_wrap=2, height=10)
dcp.map(sns.histplot, "price", bins=50)
plt.show()

#stacked hist of price from business and economy
plt.hist([ df[df['class'] == 'Business']['price'], df[df['class'] == 'Economy']['price']], bins=50, stacked=True, color=["black", "pink"], edgecolor="white")
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Stacked Histogram of Flight Class')
plt.legend(['Business', 'Economy'])
plt.show()

''' Model Training '''
# Linear Regression Model
print("\nTraining linear regression model...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print("Done training linear regression.\n")

# Ridge Regression Model
print("Training ridge regression regression model...")
ridge_reg = Ridge(alpha=1) # arbitrary choice
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
print("Done training ridge regression.\n")

# Lasso Regression Model: needs scaled data for faster weight convergences
# print("Training lasso regression model...")
# lasso_reg = Lasso(max_iter=1000, random_state=42,alpha=0.01)
# lasso_reg.fit(X_train, y_train)
# y_pred_lasso = lasso_reg.predict(X_test)
# print("Done lasso linear regression.\n")

# Polynomial Regression Model: 
# print("Training polynomial regression model...")
# poly_reg = PolynomialFeatures(degree=2) # degree of 3 makes RAM run out
# X_poly_train = poly_reg.fit_transform(X_train)
# lin_reg.fit(X_poly_train, y_train)
# X_poly_test = poly_reg.transform(X_test)
# y_pred_poly= lin_reg.predict(X_poly_test)
# print("Done polynomial regression.\n")

# Random Forest Model
# print("Training random forest model...")
# rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=0, n_jobs=-1)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# print("Done Random Forest.\n")

# XGBoost Model
print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

importances = xgb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
plt.title('XGBoost Feature Importance')
# uncomment this out to view graph
#plt.show() 

'''Metric Analysis'''
# Regression Metrics: RMSE, R2 Score for all regression models
Predictions =  { #list of model prediction outputs
    'Linear Regression Model': y_pred_lin,
    'Ridge Regression Model': y_pred_ridge,
    #'Lasso Regression Model': y_pred_lasso,
    #'Polynomial Regression Model': y_pred_poly,
    #'Random Forest Model': y_pred_rf,
    'XGBoost Model': y_pred_xgb,

}

for name, y_pred in Predictions.items():
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

