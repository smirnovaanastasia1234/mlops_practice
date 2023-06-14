import pandas as pd
new_df = pd.read_csv('datasets/df2.csv',delimiter=',')
new_df['Age'] = new_df['Age'].fillna(new_df['Age'].mean())
new_df.to_csv('datasets/df2.csv',index=False)
