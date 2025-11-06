import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('archive/Clean_Dataset.csv') #read the dataset

cols_encode = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'] #columns we need to encode (categorical columns)

encoder = OneHotEncoder(sparse_output=False) #intialize OneHotEncoder
encode_data = encoder.fit_transform(df[cols_encode]) #encode the data

df_encoded = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(cols_encode)) #new dataframe with encoding

edf = pd.concat([df.drop(columns=cols_encode), df_encoded], axis=1) #drops categorical and makes new datafram with encoding along columns

print(edf.head()) #sanity check

X = edf.iloc[:, edf.columns != 'price'] #training data columns
y = edf.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0) #5% test 95% training split

print(X_test.head()) #print
print(X_train.head())
print(y_test.head())
print(y_train.head())