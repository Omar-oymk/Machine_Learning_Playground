import numpy as np
import pandas as pd

# load the data
df = pd.read_csv(r"C:\Users\user\Downloads\archive (2)\Pokemon.csv")
print(df.head())
print(df.info())

#region preprocessing

# first drop both the names and the # 
df.drop(columns = ["#", "Name"], inplace = True)

# then fill nulls with the most common type in type 2   (THIS IS NOT OPTIMAL)
# df["Type 2"] = df["Type 2"].fillna(df["Type 2"].mode()[0])
# THIS IS WHY reasearching the data before starting is important cause type 2 only represents like side features
# some of them dont have side features so we cant just add innaccurate side features so we can instead 

# repeat the values from type 1
df["Type 2"] = df["Type 2"].fillna(df["Type 1"])
print(df.info())


# then just one hot encoding both types to ints
df = pd.concat([
                df,
                pd.get_dummies(df["Type 1"]).astype(int),
                pd.get_dummies(df["Type 2"]).astype(int)
                ], axis = 1)

print(df.head(10))
df.drop(columns = ["Type 1", "Type 2"], inplace = True)
# now check for correlation and drop redundant features
print(df.corr())
# since here we have 45x45 so ill just export it to a csv file and check from there
(df.corr()).to_csv(r"C:\Users\user\Downloads\pokemoncorr.csv")
# for simplicity's sake since this is an implemenatation from scratch ill only keep
# the total which is corr by approx 0.5
X = [[df["Total"]]]
Y = [df["Legendary"]]

#endregion