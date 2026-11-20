import pandas as pd

from config import RAW_CSV

df = pd.read_csv(RAW_CSV, low_memory=False)
print(df.shape)
df.info()
print(df.isna().sum())
print(df.duplicated().sum())
