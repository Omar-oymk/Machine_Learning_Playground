import numpy as np
import pandas as pd
import random

# loading data
df = pd.read_csv(r"C:\Users\user\Downloads\train.csv")
print(f"{df.head()}\n")
print(f"{df.info()}\n")

# region PREPROCESSING DATA

# 1) drop the obvious unrelated categorical features
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis= 1, inplace= True)


# 2) handle NAs

# MISTAKE: THIS DROPS LOTS OF ROWS CAUSE THE DATASET IS FULL OF NAs
# df.dropna(inplace= True)

# so instead fill the NAs with values
df["Age"] = df["Age"].fillna(df["Age"].median())      # we fill with median and not the mean cause median will use the middle value (robust to outliers)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) # get mode() (the most common values) then if there is a tie in values take the first one (5od el awal mesh el awal mokarar)


# 3) Feature engineering

# now for the one hot encoding for the gender
# MISTAKE : THIS WILL GENERATE TWO MORE FEATURES ONE OF THEM WILL BE REDUNDANT CAUSE THEY REPRESENT THE SAME THING NO MALE WILL BE FEMALE AT THE SAME TIME
# gender_column = pd.get_dummies(df["Sex"]).astype(int)

# make it binary encoding instead for to make the model less complex
male_column = pd.get_dummies(df["Sex"], drop_first= True, prefix= "Is").astype(int)
embarked_column = pd.get_dummies(df["Embarked"], prefix= "EmbarkedAt").astype(int)
df = pd.concat([df, male_column, embarked_column], axis = 1)

# now we can get rid of the sex and embarked columns
df.drop(["Sex", "Embarked"], axis = 1, inplace= True)


# 4) check for correlations and handle redundant features
# now for the correlation to know which ones are redundant 
print(df.corr())

# THIS SHOWED THAT [PCLASS, ISMALE, FARE] ARE THE ONLY ONES CORRELATED WITH SURVIVED
# now check for the corr between them to remove ones that very corr with each other
# well the pclass is only corr with ismale by 0.15 so we can take both for now
# pclass and fare are highly corr with each other by 0.55 we can use only one of them to make things simpler
# df.drop(["Age", "SibSp", "Parch", "Fare", "EmbarkedAt_S", "EmbarkedAt_C", "EmbarkedAt_Q"], axis = 1, inplace= True)
df.drop(["Age", "SibSp", "Parch", "EmbarkedAt_S", "EmbarkedAt_Q"], axis = 1, inplace= True)

print(f"\n{df.head()}")
# endregion

# region DEFINING FUNCTIONS

def handle_default_values(train_size, test_size, shuffle, random_state):
    # now handle the default actions
    # train/test size
    if (train_size is not None and test_size is not None):
        raise Exception("U cannot declare both test and train size at the same time")
    
    if (train_size is None and test_size is None):
        train_size = 0.75
        test_size = 0.25
    elif (train_size is not None):  test_size = 1 - train_size
    elif (test_size is not None):   train_size = 1 - test_size
    # shuffling and random_state
    if (shuffle is None and random_state is not None): shuffle = True
    elif (shuffle is None): shuffle = False
    
    return train_size, test_size, shuffle, random_state

def handle_shuffling(X, Y, train_size, shuffle, random_state):
     # only dividing dataset when shuffling = false
    if(shuffle == False):
        # calculate the number of rows 
        n = (int)(len(X) * train_size)
        return X.head(n), X.tail(len(X) - n), Y.head(n), Y.tail(len(Y) - n)     # take the top then the rest from bottom
    
    # shuffling indicies first when shuffling = true
    elif(shuffle == True):
        # alright so after doing some research on the scikitlearn version of this fn apparently it makes a list of indicies then shuffles them based on seed
        # then it returns it onto the x and y 
        list_of_indicies = []
        rows, columns = X.shape
        for index in range(rows):
            list_of_indicies.append(index)
        
        random.seed(random_state)       # set the random seed to shuffle with the exact seed the user declared
        random.shuffle(list_of_indicies)    # this will shuffle using random seed the user declared

        shuffled_x = X.iloc[list_of_indicies]
        shuffled_y = Y.iloc[list_of_indicies]
        
        # now split the data
        n = (int)(len(X) * train_size)
        return shuffled_x.head(n), shuffled_x.tail(len(X) - n), shuffled_y.head(n), shuffled_y.tail(len(Y) - n)     # take the top then the rest from bottom

# 1) creating the traintestsplit fn
# the one from scikit learn has multiple features 
# it allows u to pick urself if u want to declare the test size or train size 
# if u dont declare them it will automatically make test size = 25% of the whole dataset
# another features it to decide whether u want it to shuffle randomly and based on a pseudorandom number generator or not

# to do so we declare parameters values = none so that the user isnt obliged to input them
def train_test_split(X : pd.DataFrame, Y : pd.DataFrame, train_size = None, test_size = None, shuffle = None, random_state = None):

    train_size, test_size, shuffle, random_state = handle_default_values(train_size, test_size, shuffle, random_state)
    return handle_shuffling(X, Y, train_size, shuffle, random_state)

def zscoreStandardization(x_train):
    xscaled = (x_train - np.mean(x_train, axis = 0))/np.std(x_train, axis = 0)
    return xscaled

X = df[["Pclass", "Is_male", "Fare"]]
Y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, random_state= 42)

x_train = zscoreStandardization(x_train)
y_train = y_train.values
x_test = zscoreStandardization(x_test)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

#endregion

# region DEFINING TRAINING FUNCTIONS

# initial weights

# first get number of columns (features)
rows, cols, = x_train.shape

w = np.zeros(cols)          # THIS CREATES AN ARRAY OF WEIGHTS ALL INITIALIZED AS 0 (so for example feature 1 has w1 = 0, feature 2 has w2 = 0, etc....)
                                        # why as numpy array? well cause i want to dot product anyways between the w and features
b = 0

def linear(X):
    # first turn x from dataframe ---> numpy array
    z = np.dot(X, w) + b
    return z

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def loss_function(x_train, y_train):
    
    sum = np.sum(y_train * np.log(np.clip(sigmoid(linear(x_train)), 1e-15, 1 - 1e-15)) + (1 - y_train) * np.log(1 - np.clip(sigmoid(linear(x_train)), 1e-15, 1 - 1e-15)))
    return -(1/rows) * sum

def gradient(X, Y):
    y_hat = sigmoid(linear(X))
    DcostDw = (1/rows)* np.dot(X.T, (y_hat - Y))    # this will hold w[0], w[1] and so on
    DcostDb = (1/rows)* np.sum(y_hat - Y)           # get the sum of this array first cause this is how it is it is (sigma)

    return DcostDw, DcostDb

def gradient_descent(DcostDw, DcostDb, learning_rate):
    new_w = w - learning_rate*DcostDw
    new_b = b - learning_rate*DcostDb

    return new_w, new_b

#endregion

# region training loop
def LogisticRegressionfit(x_train, y_train, x_test, y_test, epochs, learning_rate):

    global w, b
    for i in range(epochs):
        
        if(i % 100 == 0):
            print(loss_function(x_test.values, y_test.values))
    
        dw, db = gradient(x_train, y_train)
        w, b = gradient_descent(dw, db, learning_rate)
#endregion

LogisticRegressionfit(x_train, y_train, x_test, y_test, 10000, 0.01)