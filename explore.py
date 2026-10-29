import pandas as pd
df = pd.read_csv("raw data/example_data.csv")
print(df.shape)
df.info()
print(df.isna().sum())
print(df.duplicated().sum())
