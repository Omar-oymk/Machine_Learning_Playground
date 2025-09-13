import numpy as np
import pandas as pd
import random
from collections import Counter

# loading and reading dataset
df = pd.read_csv(r"C:\Users\user\Downloads\breast-cancer.csv")
print(df.head())
print(f"\n{df.info()}")
print(F"\n{df.describe()}")

# preprocessing data
df.drop(columns="id", inplace= True)
malcol = pd.get_dummies(df["diagnosis"], drop_first= True).astype(int)
malcol.columns = ["isMalignant?"]
df = pd.concat([df, malcol], axis = 1)
df.drop(columns= "diagnosis", inplace= True)
# df.corr().to_csv("Downloads/cancer_correlation_table.csv")
# after filtering using correlation ill only keep the "concave points_worst" column for simplicity
X = df[["concave points_worst"]]
Y = df["isMalignant?"]

# scaling
def zscoreStandardization(X):
    return (X - X.mean())/X.std()

X = zscoreStandardization(X)

# divide dataset using train_test_split fn from scratch
def train_test_split(X, Y, train_size = None, test_size = None, shuffle = False, random_state = None):
    """
    returns x_train, x_test, y_train, y_test
    defaults to 
    train_size = 0.75 
    test_size = 0.25 
    shuffling = False
    """
    # validation
    if(train_size is not None and test_size is not None):
        raise Exception("You can only declare one either train_size or test_size for simplicity")
    if(train_size is not None and not 0 <= train_size <= 1):
        raise ValueError("Please enter a valid train size fractional value between 0 and 1")
    if(test_size is not None and not 0 <= test_size <= 1):
        raise ValueError("Please enter a valid test size fractional value between 0 and 1")
    
    # assigning default values
    if(train_size is not None): test_size = 1 - train_size
    elif(test_size is not None): train_size = 1 - test_size
    else: train_size, test_size = 0.75, 0.25    
    
    # start splitting the data
    if(shuffle):
        list_of_indices = list(range(X.shape[0]))
        random.seed(random_state)
        random.shuffle(list_of_indices)
        X = X.iloc[list_of_indices]
        Y = Y.iloc[list_of_indices]

    n = (int)(len(X) * train_size)
    return X.iloc[:n], X.iloc[n:], Y.iloc[:n], Y.iloc[n:]       # learned that numpy slicing is better and faster than head, tail

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.8)

# start training
class KNN:

    def __init__(self, k):
        self.k = k

    def Fit(self, x_train, y_train):
        self.x_train = x_train.values
        self.y_train = y_train.values

    def predict(self, x):
        distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis = 1))
        distancesOrdered = np.argsort(distances)
        pickedDistances = distancesOrdered[:self.k]
        pickedLabels = self.y_train[pickedDistances]
        most_common = Counter(pickedLabels).most_common(1)[0][0]
        
        return most_common
    


model = KNN(3)

model.Fit(x_train, y_train)

print(model.predict(1))