import pandas as pd
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('datasets/df2.csv')
onehot_encoder = OneHotEncoder(sparse=False)
sex_ohe = onehot_encoder.fit_transform(df[['Sex']])
sex_ohe_df = pd.DataFrame(sex_ohe, columns=onehot_encoder.categories_[0])
df = pd.concat([df, sex_ohe_df], axis=1)

df.to_csv('datasets/df2.csv', index=False)
