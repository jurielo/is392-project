import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('archive/Clean_Dataset.csv') #read the dataset

cols_encode = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'] #columns we need to encode (categorical columns)

encoder = OneHotEncoder(sparse_output=False) #intialize OneHotEncoder
encode_data = encoder.fit_transform(df[cols_encode]) #encode the data

df_encoded = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(cols_encode)) #new dataframe with encoding

edf = pd.concat([df.drop(columns=cols_encode), df_encoded], axis=1) #drops categorical and makes new datafram with encoding along columns

print(edf.head()) #sanity check