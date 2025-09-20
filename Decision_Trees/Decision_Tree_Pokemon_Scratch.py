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
X = df[["Total"]]
y = df["Legendary"]

#endregion

#region splitting data
def train_test_split(X, y, train_size = None, test_size = None, shuffle = False, random_state = None):
    if(train_size is not None and test_size is not None):
        raise ValueError("NOT ALLOWED TO DEFINE BOTH train_size AND test_size AT THE SAME TIME")
    
    if(train_size is None and test_size is None):
        train_size = 0.75
        test_size = 0.25
    elif(train_size is not None and 0 <= train_size <= 1): test_size = 1 - train_size
    elif(test_size is not None and 0 <= test_size <= 1): train_size = 1 - test_size
    else: raise ValueError("Please keep train/test size a fractional value between 0 and 1 inclusive")

    n = (int)(train_size * len(X))
    if(shuffle or random_state is not None):
        np.random.seed(random_state)
        indices_shuffled = np.random.permutation(len(X))    # the random.perm returns a shuffled n indices from 0 - n-1
        X = X.iloc[indices_shuffled]
        y = y.iloc[indices_shuffled]

    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42)
#endregion

