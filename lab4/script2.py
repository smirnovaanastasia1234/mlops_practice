import pandas as pd


df = pd.read_csv('datasets/df.csv')
df = df[['Pclass', 'Sex', 'Age']]
df.to_csv('datasets/df2.csv', index=False)
